"""Streaming helpers for OpenRouter chat-completion responses.

This module owns acceptance criterion AC-6: token streaming reaches the
GitHub Action logs. It parses an OpenRouter Server-Sent-Events stream
(OpenAI chat-completions compatible) and writes each content delta to
stdout *as it arrives*, flushing aggressively so the runner forwards
each chunk to the live job log instead of buffering until process exit.

Design notes
------------
* OpenRouter's streaming responses are OpenAI-compatible SSE: each line
  starts with ``data: `` and is followed by either a JSON chunk or the
  ``[DONE]`` sentinel.  We tolerate ``: keep-alive`` comments and blank
  separator lines as recommended by the SSE spec.
* GitHub Actions buffers per-line: a token delta written without a
  trailing newline is held in the runner's pipe until the next newline
  or until the process exits. We therefore flush stdout after *every*
  delta, and emit a trailing newline once the stream completes so the
  final partial line is committed to the log.
* ``PYTHONUNBUFFERED=1`` is set by ``action.yml`` so that the Python
  process itself does not introduce additional buffering on top of
  Python's default line-buffered stdout.
* The printer is decoupled from any specific HTTP client. Pass it any
  iterable of raw SSE bytes (``httpx.Response.iter_bytes()``,
  ``requests.Response.iter_content()``, a list of bytes in tests, etc.).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import IO, Any

# Sentinel emitted by OpenRouter / OpenAI-compatible servers to signal
# the end of a chat-completion stream.
_SSE_DONE_SENTINEL = "[DONE]"

# Prefix on real data frames in the SSE protocol.
_SSE_DATA_PREFIX = "data:"

# Prefix used by SSE servers (including OpenRouter) for keep-alive
# comments. We must not interpret these as JSON.
_SSE_COMMENT_PREFIX = ":"


@dataclass
class StreamingResult:
    """Final aggregate produced by consuming an OpenRouter SSE stream."""

    text: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    # Tool-call accumulation keyed by index: the OpenAI/OpenRouter wire
    # format streams tool calls as deltas with stable indices, so the
    # consumer must merge by index. We expose the merged map for the
    # schema-enforced JSON path; AC-6 itself only requires text
    # streaming, but tracking tool-call deltas here keeps the streaming
    # layer reusable for the schema turn.
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class _DeltaSinks:
    """Bundle of optional sinks the printer dispatches deltas through."""

    on_text: Callable[[str], None] | None = None
    on_reasoning: Callable[[str], None] | None = None


class OpenRouterStreamPrinter:
    """Consume an OpenRouter SSE chat-completion stream and stream tokens.

    The printer is intentionally narrow. It does no HTTP work; the
    caller hands in either an iterable of raw byte chunks (the typical
    ``response.iter_bytes()`` shape) or an iterable of already-parsed
    SSE event payloads. The printer guarantees:

    * Every text-content delta is written to ``stdout`` and flushed
      before returning control to the caller.
    * If ``stream_to_logs`` is False, the deltas are still aggregated
      into the returned ``StreamingResult`` but no live output is
      produced. This mirrors the legacy ``suppress_stream`` flag used
      by the schema-enforced JSON turn.
    * A trailing newline is emitted after the final delta when streaming
      to logs, so GitHub's per-line log buffer commits the last partial
      line.
    """

    def __init__(
        self,
        *,
        stream_to_logs: bool = True,
        stdout: IO[str] | None = None,
        debug: Callable[[int, str], None] | None = None,
        on_text_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._stream_to_logs = stream_to_logs
        self._on_usage = on_usage
        self._stdout: IO[str] = stdout if stdout is not None else sys.stdout
        self._debug = debug or (lambda _level, _message: None)
        self._sinks = _DeltaSinks(
            on_text=on_text_delta,
            on_reasoning=on_reasoning_delta,
        )

    # ------------------------------------------------------------------
    # Public API

    def consume_bytes(self, byte_chunks: Iterable[bytes]) -> StreamingResult:
        """Consume an iterable of raw SSE byte chunks.

        Most HTTP clients (httpx, requests) expose a ``iter_bytes`` /
        ``iter_content`` method that yields arbitrarily-sized chunks
        of the response body. We re-frame those into SSE events here.
        """
        return self.consume_events(_iter_sse_events(byte_chunks))

    def consume_events(self, events: Iterable[dict[str, Any]]) -> StreamingResult:
        """Consume already-decoded SSE JSON events.

        Each event is the dict that was after ``data: `` (i.e. the
        raw JSON body). Sentinel ``[DONE]`` lines are handled by the
        framer in :func:`_iter_sse_events`, so callers feeding events
        directly should simply omit the sentinel.
        """
        result = StreamingResult(text="")
        text_buffer: list[str] = []
        emitted_anything = False

        try:
            for event in events:
                emitted_anything = (
                    self._handle_event(
                        event=event,
                        result=result,
                        text_buffer=text_buffer,
                    )
                    or emitted_anything
                )
        finally:
            # Report usage even when the stream breaks after the usage frame arrived.
            if self._on_usage is not None and result.usage is not None:
                self._on_usage(result.usage)
            if self._stream_to_logs and emitted_anything:
                # Commit the final partial line so the GH Actions log
                # buffer flushes the last token.
                self._stdout.write("\n")
                self._stdout.flush()

        result.text = "".join(text_buffer)
        return result

    # ------------------------------------------------------------------
    # Event dispatch

    def _handle_event(
        self,
        *,
        event: dict[str, Any],
        result: StreamingResult,
        text_buffer: list[str],
    ) -> bool:
        """Process a single decoded SSE event. Returns True if it
        produced any visible token output (used so we know whether to
        emit a trailing newline)."""
        emitted = False

        usage = event.get("usage")
        if isinstance(usage, dict):
            result.usage = usage

        choices = event.get("choices")
        if not isinstance(choices, list):
            return emitted

        for choice in choices:
            if not isinstance(choice, dict):
                continue

            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                result.finish_reason = finish_reason

            delta = choice.get("delta")
            if not isinstance(delta, dict):
                # Some providers emit a final ``message`` payload on
                # the last event. Treat ``message.content`` as a
                # one-shot text delta so we never lose tokens.
                message = choice.get("message")
                if isinstance(message, dict):
                    text = message.get("content")
                    if isinstance(text, str) and text:
                        emitted = self._emit_text_delta(text, text_buffer) or emitted
                continue

            # Reasoning deltas (Anthropic / DeepSeek style "thinking"
            # tokens) are surfaced as a separate stream so they do not
            # corrupt the schema-enforced JSON turn but still appear in
            # the live log when callers opt in.
            reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                self._emit_reasoning_delta(reasoning)

            content = delta.get("content")
            if isinstance(content, str) and content:
                emitted = self._emit_text_delta(content, text_buffer) or emitted

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                _merge_tool_call_deltas(result.tool_calls, tool_calls)

        return emitted

    def _emit_text_delta(self, chunk: str, text_buffer: list[str]) -> bool:
        text_buffer.append(chunk)
        if self._sinks.on_text is not None:
            self._sinks.on_text(chunk)
        if self._stream_to_logs:
            self._stdout.write(chunk)
            # Flushing per-delta is what gets the chunk into the GH
            # Actions log buffer immediately. Without this, the runner
            # only sees output after a newline or after the process
            # exits.
            self._stdout.flush()
            return True
        return False

    def _emit_reasoning_delta(self, chunk: str) -> None:
        if self._sinks.on_reasoning is not None:
            self._sinks.on_reasoning(chunk)
        # Reasoning is intentionally *not* mirrored to stdout by
        # default — most reviewers want token output, not raw chain of
        # thought, in CI logs. Debug level >= 2 surfaces it via the
        # debug callback the caller installs through ``on_reasoning``.
        self._debug(2, f"[openrouter-reasoning] chars={len(chunk)}")


# ----------------------------------------------------------------------
# Internal SSE framer
# ----------------------------------------------------------------------


def _iter_sse_events(byte_chunks: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Frame a raw byte stream into a sequence of decoded SSE events.

    Implements just enough of the SSE wire format to handle OpenRouter
    chat completions:

    * Lines are separated by ``\\n``. CRLF is normalized to LF.
    * Blank lines are event separators (we ignore them — each ``data:``
      line is a complete JSON object on this provider).
    * Lines starting with ``:`` are comments / keep-alives.
    * ``data: [DONE]`` terminates the stream.
    * Anything else after ``data: `` is JSON to decode.

    Malformed JSON is logged and skipped rather than raising so a
    single corrupt frame does not abort the whole review.
    """
    buffer = ""
    for raw_chunk in byte_chunks:
        if not raw_chunk:
            continue
        if isinstance(raw_chunk, (bytes, bytearray)):
            text = raw_chunk.decode("utf-8", errors="replace")
        else:
            # Defensive: some test fakes may yield strings directly.
            text = str(raw_chunk)
        buffer += text.replace("\r\n", "\n").replace("\r", "\n")

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            event = _decode_sse_line(line)
            if event is _SENTINEL_DONE:
                return
            if event is None:
                continue
            yield event

    # Drain any final line that lacked a trailing newline.
    if buffer.strip():
        event = _decode_sse_line(buffer)
        if event is _SENTINEL_DONE:
            return
        if event is not None:
            yield event


# Marker object distinguishable from "no event" (None) and from a real
# event dict. Using a unique object identity sidesteps any chance of a
# real payload happening to compare equal.
_SENTINEL_DONE: Any = object()


def _decode_sse_line(line: str) -> dict[str, Any] | None | Any:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(_SSE_COMMENT_PREFIX) and not stripped.startswith(_SSE_DATA_PREFIX):
        # Comment / keep-alive line.
        return None
    if not stripped.startswith(_SSE_DATA_PREFIX):
        # Some providers emit ``event: ...`` lines we do not need.
        return None
    payload = stripped[len(_SSE_DATA_PREFIX) :].lstrip()
    if payload == _SSE_DONE_SENTINEL:
        return _SENTINEL_DONE
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        # Skip malformed frames — never let one bad chunk kill the
        # review.
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def _merge_tool_call_deltas(
    accumulator: dict[int, dict[str, Any]],
    deltas: list[Any],
) -> None:
    """Merge OpenAI-shaped tool_call streaming deltas by index.

    OpenRouter follows the OpenAI streaming contract: each tool call
    arrives as multiple deltas keyed by ``index`` with the function
    ``arguments`` field arriving as a sequence of partial strings that
    must be concatenated.
    """
    for raw in deltas:
        if not isinstance(raw, dict):
            continue
        index_value = raw.get("index")
        if not isinstance(index_value, int):
            continue
        slot = accumulator.setdefault(index_value, {})

        for top_level_field in ("id", "type"):
            value = raw.get(top_level_field)
            if isinstance(value, str) and value:
                slot[top_level_field] = value

        function_delta = raw.get("function")
        if not isinstance(function_delta, dict):
            continue
        function_slot = slot.setdefault("function", {})
        name = function_delta.get("name")
        if isinstance(name, str) and name:
            function_slot["name"] = name
        arguments = function_delta.get("arguments")
        if isinstance(arguments, str) and arguments:
            current_arguments = function_slot.get("arguments")
            function_slot["arguments"] = (
                arguments
                if not isinstance(current_arguments, str)
                else current_arguments + arguments
            )
