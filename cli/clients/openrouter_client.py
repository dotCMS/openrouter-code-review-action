"""OpenRouter chat-completions client with a tool-calling agent loop.

Drop-in replacement for :class:`~cli.clients.codex_client.CodexClient`
from the workflows' point of view: the same ``execute_text`` /
``execute_structured`` surface, backed by OpenRouter's
OpenAI-compatible chat-completions endpoint instead of the Codex SDK.

Agentic parity
--------------
The Codex SDK bundled an agent runtime (tool loop + sandbox). OpenRouter
only gives us chat completions, so this client implements the loop:

1. Send the prompt plus a small tool surface (``read_file``,
   ``write_file``, ``run_command`` — see :mod:`.openrouter_tools`).
2. When the model responds with tool calls, execute them locally and
   feed the results back as ``role: "tool"`` messages.
3. Repeat until the model produces a plain-text answer or the
   iteration cap is hit (after which the model is asked to answer
   without further tools).

Structured output parity
------------------------
``execute_structured`` mirrors the Codex two-turn contract: an agentic
turn first (tools enabled, streaming to logs), then a schema-enforced
turn that replays the whole conversation with OpenRouter's strict
``response_format`` JSON-schema enforcement and no tools.

Resume parity
-------------
Conversations are persisted through
:class:`~cli.clients.openrouter_thread_store.OpenRouterThreadStore` so
``resume_thread_id`` continues a prior review thread exactly like the
Codex ``CODEX_HOME`` thread resume did.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from ..core.config import ReviewConfig, make_debug
from ..core.exceptions import ConfigurationError, DotBotExecutionError
from ..core.models import build_openrouter_response_format
from .openrouter_stream import OpenRouterStreamPrinter, StreamingResult
from .openrouter_thread_store import OpenRouterThreadStore
from .openrouter_tools import TOOL_SPECS, execute_tool

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = "https://github.com/dotcms/openrouter-code-review-action"
OPENROUTER_TITLE = "openrouter-code-review-action"
OPENROUTER_USER_AGENT = "openrouter-code-review-action/1.0"
OPENROUTER_TIMEOUT_SECONDS = 300

DEFAULT_MAX_TOOL_ITERATIONS = 30
_ONLINE_SUFFIX = ":online"
_VALID_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})

Transport = Callable[[dict[str, Any]], dict[str, Any]]
StreamTransport = Callable[[dict[str, Any]], Iterable[bytes]]

_AGENT_SYSTEM_PROMPT = (
    "You are an autonomous coding agent operating on a git repository "
    "checked out at the current working directory. Use the provided tools "
    "(read_file, write_file, run_command) to inspect and, when asked, "
    "modify the repository. Do not fabricate file contents — read them. "
    "When your task is complete, respond with your final answer as plain "
    "text without calling any more tools."
)

_TOOL_ITERATION_LIMIT_MESSAGE = (
    "Tool iteration limit reached. Do not call any more tools; respond with your final answer now."
)

# Cap on the one-line tool-argument summary printed at debug level 1.
_TOOL_SUMMARY_ARG_LIMIT = 120


def _summarize_tool_call(name: str, arguments_json: str) -> str:
    """Render a short, readable summary of a tool call for CI logs.

    Shows what the model is doing (the git command, the file being read,
    the file being written) without dumping full file contents.
    """
    try:
        arguments = json.loads(arguments_json) if arguments_json.strip() else {}
    except json.JSONDecodeError:
        # Malformed arguments: show the raw payload (truncated) rather than
        # claiming the argument was simply missing.
        detail = arguments_json
        if len(detail) > _TOOL_SUMMARY_ARG_LIMIT:
            detail = detail[: _TOOL_SUMMARY_ARG_LIMIT - 3] + "..."
        return detail
    if not isinstance(arguments, dict):
        arguments = {}
    if name == "run_command":
        command = arguments.get("command")
        detail = command if isinstance(command, str) else "<missing command>"
    elif name == "read_file":
        detail = f"path={arguments.get('path', '<missing>')!r}"
    elif name == "write_file":
        path = arguments.get("path")
        content = arguments.get("content")
        length = len(content) if isinstance(content, str) else "?"
        detail = f"path={path!r} ({length} chars)"
    else:
        detail = arguments_json
    if len(detail) > _TOOL_SUMMARY_ARG_LIMIT:
        detail = detail[: _TOOL_SUMMARY_ARG_LIMIT - 3] + "..."
    return detail


class OpenRouterClient:
    """Client for executing OpenRouter models with typed streaming and tool use."""

    def __init__(
        self,
        config: ReviewConfig,
        *,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        thread_store: OpenRouterThreadStore | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ) -> None:
        self.config = config
        self._debug = make_debug(config)
        self._transport: Transport | None = transport
        self._stream_transport: StreamTransport | None = stream_transport
        self._thread_store = thread_store or OpenRouterThreadStore(
            OpenRouterThreadStore.default_directory()
        )
        self._max_tool_iterations = max(1, max_tool_iterations)
        self._usage_totals: dict[str, int] = {}
        self._request_count = 0

    # ------------------------------------------------------------------
    # Observability (usage / request timing)
    # ------------------------------------------------------------------

    def _begin_usage_window(self) -> None:
        self._usage_totals = {}
        self._request_count = 0

    def _accumulate_usage(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                self._usage_totals[key] = self._usage_totals.get(key, 0) + value

    def _log_request(
        self, result: StreamingResult, *, model_name: str | None, elapsed: float
    ) -> None:
        self._request_count += 1
        self._accumulate_usage(result.usage)
        self._debug(
            1,
            f"[openrouter-request] model={self._resolved_model(model_name, online_suffix=False)} "
            f"request#{self._request_count} elapsed={elapsed:.1f}s "
            f"finish={result.finish_reason or 'n/a'}",
        )

    def _log_usage_summary(self, *, model_name: str | None) -> None:
        if not self._usage_totals:
            return
        model = self._resolved_model(model_name, online_suffix=False)
        self._debug(
            1,
            f"[openrouter-usage] model={model} requests={self._request_count} "
            f"prompt_tokens={self._usage_totals.get('prompt_tokens', 0)} "
            f"completion_tokens={self._usage_totals.get('completion_tokens', 0)} "
            f"total_tokens={self._usage_totals.get('total_tokens', 0)}",
        )

    # ------------------------------------------------------------------
    # Public API (CodexClient parity)
    # ------------------------------------------------------------------

    def execute_text(
        self,
        prompt: str,
        *,
        model_name: str | None = None,
        reasoning_effort: str | None = None,
        suppress_stream: bool = False,
        sandbox_mode: str = "read-only",
        resume_thread_id: str | None = None,
    ) -> str:
        """Run an agentic turn and return the final agent text."""
        self._begin_usage_window()
        messages, thread_id = self._start_thread(resume_thread_id)
        messages.append({"role": "user", "content": prompt})

        stream_enabled = self._should_stream(suppress_stream)
        final_text = self._agent_loop(
            messages,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            sandbox_mode=sandbox_mode,
            stream_enabled=stream_enabled,
        )
        self._save_thread(thread_id, messages, model_name=model_name)

        if not final_text.strip():
            raise DotBotExecutionError("OpenRouter did not return an agent message.")
        self._log_usage_summary(model_name=model_name)
        return final_text

    def execute_structured(
        self,
        prompt: str,
        *,
        output_schema: dict[str, object],
        schema_prompt: str = "Produce the JSON output now.",
        model_name: str | None = None,
        reasoning_effort: str | None = None,
        suppress_stream: bool = False,
        sandbox_mode: str = "read-only",
        resume_thread_id: str | None = None,
    ) -> str:
        """Run an agentic turn followed by a schema-enforced output turn."""
        self._begin_usage_window()
        messages, thread_id = self._start_thread(resume_thread_id)
        messages.append({"role": "user", "content": prompt})

        stream_enabled = self._should_stream(suppress_stream)
        self._agent_loop(
            messages,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            sandbox_mode=sandbox_mode,
            stream_enabled=stream_enabled,
        )
        messages.append({"role": "user", "content": schema_prompt})
        content = self._schema_turn(
            messages,
            output_schema=output_schema,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        )
        # Persist the schema-turn answer too so a resumed thread carries
        # the full review history, including the last structured output.
        if content.strip():
            messages.append({"role": "assistant", "content": content})
        self._save_thread(thread_id, messages, model_name=model_name)

        if not content.strip():
            raise DotBotExecutionError("OpenRouter did not return structured output on turn 2.")
        self._log_usage_summary(model_name=model_name)
        return content

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    def _agent_loop(
        self,
        messages: list[dict[str, Any]],
        *,
        model_name: str | None,
        reasoning_effort: str | None,
        sandbox_mode: str,
        stream_enabled: bool,
    ) -> str:
        for _ in range(self._max_tool_iterations):
            started = time.monotonic()
            result = self._agent_request(
                messages,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                stream_enabled=stream_enabled,
            )
            self._log_request(result, model_name=model_name, elapsed=time.monotonic() - started)
            tool_calls = _tool_calls_list(result)

            if not tool_calls:
                messages.append(_assistant_message(text=result.text, tool_calls=None))
                return result.text

            messages.append(_assistant_message(text=result.text, tool_calls=tool_calls))
            for call in tool_calls:
                tool_name = _tool_call_name(call)
                self._debug(
                    1,
                    f"[openrouter-tool] {tool_name}: {_summarize_tool_call(tool_name, _tool_call_arguments(call))}",
                )
                output = execute_tool(
                    tool_name,
                    _tool_call_arguments(call),
                    repo_root=self.config.resolved_repo_root,
                    sandbox_mode=sandbox_mode,
                )
                self._debug(2, f"[openrouter-tool] {tool_name} -> {len(output)} chars")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _tool_call_id(call),
                        "content": output,
                    }
                )

        self._debug(
            1,
            f"OpenRouter agent hit the {self._max_tool_iterations}-iteration tool cap; "
            "forcing a final answer",
        )
        messages.append({"role": "user", "content": _TOOL_ITERATION_LIMIT_MESSAGE})
        result = self._agent_request(
            messages,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            stream_enabled=stream_enabled,
            include_tools=False,
        )
        messages.append(_assistant_message(text=result.text, tool_calls=None))
        return result.text

    def _agent_request(
        self,
        messages: list[dict[str, Any]],
        *,
        model_name: str | None,
        reasoning_effort: str | None,
        stream_enabled: bool,
        include_tools: bool = True,
    ) -> StreamingResult:
        payload: dict[str, Any] = {
            "model": self._resolved_model(model_name, online_suffix=True),
            "messages": messages,
            "reasoning": {"effort": self._resolve_effort(reasoning_effort)},
        }
        if include_tools:
            payload["tools"] = TOOL_SPECS
            payload["tool_choice"] = "auto"
        if stream_enabled:
            return self._request_stream(payload)
        return _result_from_response(self._request(payload))

    def _schema_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        output_schema: dict[str, object],
        model_name: str | None,
        reasoning_effort: str | None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._resolved_model(model_name, online_suffix=False),
            "messages": messages,
            "response_format": build_openrouter_response_format(
                schema=output_schema,
                name="structured_output",
            ),
            "reasoning": {"effort": self._resolve_effort(reasoning_effort)},
            "stream": False,
        }
        started = time.monotonic()
        response = self._request(payload)
        elapsed = time.monotonic() - started
        usage = response.get("usage")
        if isinstance(usage, dict):
            self._accumulate_usage(usage)
        self._request_count += 1
        self._debug(
            1,
            f"[openrouter-request] model={self._resolved_model(model_name, online_suffix=False)} "
            f"request#{self._request_count} (schema turn) elapsed={elapsed:.1f}s",
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DotBotExecutionError("OpenRouter response had no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise DotBotExecutionError("OpenRouter response had no message")
        content = message.get("content")
        if not isinstance(content, str):
            return ""
        return content

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_api_key()
        if self._transport is not None:
            response = self._transport(payload)
            if not isinstance(response, dict):
                raise DotBotExecutionError("OpenRouter transport returned a non-object response")
            return response

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=body,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(request, timeout=OPENROUTER_TIMEOUT_SECONDS) as resp:  # nosec B310 — fixed https URL
                decoded = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DotBotExecutionError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise DotBotExecutionError(f"OpenRouter request failed: {exc}") from exc
        if not isinstance(decoded, dict):
            raise DotBotExecutionError("OpenRouter response was not a JSON object")
        return decoded

    def _request_stream(self, payload: dict[str, Any]) -> StreamingResult:
        self._require_api_key()
        if self._stream_transport is not None:
            chunks = self._stream_transport({**payload, "stream": True})
        else:
            chunks = self._urllib_stream_chunks({**payload, "stream": True})
        reasoning_chars = 0

        def _on_reasoning(chunk: str) -> None:
            nonlocal reasoning_chars
            reasoning_chars += len(chunk)

        printer = OpenRouterStreamPrinter(
            stream_to_logs=True,
            debug=self._debug,
            on_reasoning_delta=_on_reasoning,
        )
        result = printer.consume_bytes(chunks)
        if reasoning_chars:
            self._debug(
                1,
                f"[openrouter-reasoning] {reasoning_chars} chars of reasoning tokens this request "
                "(hidden from CI logs; rerun with DEBUG_CODEREVIEW=2 for per-delta traces)",
            )
        return result

    def _urllib_stream_chunks(self, payload: dict[str, Any]) -> Iterable[bytes]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=body,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(request, timeout=OPENROUTER_TIMEOUT_SECONDS) as resp:  # nosec B310 — fixed https URL
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    yield chunk
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DotBotExecutionError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise DotBotExecutionError(f"OpenRouter request failed: {exc}") from exc

    def _require_api_key(self) -> None:
        if not self.config.openrouter_api_key.strip():
            raise ConfigurationError("Missing OPENROUTER_API_KEY for model provider 'openrouter'")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.openrouter_api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
            "User-Agent": OPENROUTER_USER_AGENT,
        }

    # ------------------------------------------------------------------
    # Threads, models, effort
    # ------------------------------------------------------------------

    def _start_thread(self, resume_thread_id: str | None) -> tuple[list[dict[str, Any]], str]:
        if resume_thread_id:
            stored = self._thread_store.load(resume_thread_id)
            if stored is not None:
                self._debug(1, f"Resuming OpenRouter thread {resume_thread_id}")
                return list(stored), resume_thread_id
            self._debug(
                1,
                f"Failed to resume OpenRouter thread {resume_thread_id}; starting fresh",
            )
        return [{"role": "system", "content": _AGENT_SYSTEM_PROMPT}], uuid.uuid4().hex

    def _save_thread(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        *,
        model_name: str | None,
    ) -> None:
        try:
            self._thread_store.save(
                thread_id,
                messages,
                repository=self.config.repository,
                pr_number=self.config.pr_number,
                model=self._resolved_model(model_name, online_suffix=False),
            )
        except OSError as exc:
            print(
                f"Warning: failed to persist OpenRouter thread {thread_id}: {exc}",
                file=sys.stderr,
            )
            self._debug(1, f"Thread persistence failed for {thread_id}: {exc}")

    def _should_stream(self, suppress_stream: bool) -> bool:
        return bool(self.config.stream_output and not suppress_stream)

    def _resolve_effort(self, reasoning_effort: str | None) -> str:
        value = (reasoning_effort or self.config.reasoning_effort or "medium").strip().lower()
        if value in _VALID_REASONING_EFFORTS:
            return value
        self._debug(1, f"Invalid reasoning effort '{value}', falling back to 'medium'")
        return "medium"

    def _resolved_model(self, model_name: str | None, *, online_suffix: bool) -> str:
        model = (model_name or self.config.selected_model or "").strip()
        if not model:
            raise ConfigurationError("No OpenRouter model configured")
        if (
            online_suffix
            and self.config.web_search_mode == "live"
            and not model.endswith(_ONLINE_SUFFIX)
        ):
            return f"{model}{_ONLINE_SUFFIX}"
        return model


# ----------------------------------------------------------------------
# Response helpers
# ----------------------------------------------------------------------


def _result_from_response(response: dict[str, Any]) -> StreamingResult:
    """Convert a non-streaming chat-completion response to a StreamingResult."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DotBotExecutionError("OpenRouter response had no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise DotBotExecutionError("OpenRouter response choice was not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise DotBotExecutionError("OpenRouter response had no message")
    content = message.get("content")
    text = content if isinstance(content, str) else ""

    result = StreamingResult(text=text)
    finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        result.finish_reason = finish_reason
    usage = response.get("usage")
    if isinstance(usage, dict):
        result.usage = usage

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for index, call in enumerate(tool_calls):
            if isinstance(call, dict):
                result.tool_calls[index] = call
    return result


def _tool_calls_list(result: StreamingResult) -> list[dict[str, Any]]:
    """Return the merged tool calls as OpenAI wire-format list entries."""
    calls: list[dict[str, Any]] = []
    for index in sorted(result.tool_calls):
        slot = result.tool_calls[index]
        call_id = slot.get("id")
        function = slot.get("function")
        if not isinstance(call_id, str) or not call_id:
            continue
        if not isinstance(function, dict):
            continue
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "",
                },
            }
        )
    return calls


def _assistant_message(
    *,
    text: str,
    tool_calls: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _tool_call_id(call: dict[str, Any]) -> str:
    return str(call.get("id") or "")


def _tool_call_name(call: dict[str, Any]) -> str:
    function = call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return ""


def _tool_call_arguments(call: dict[str, Any]) -> str:
    function = call.get("function")
    if isinstance(function, dict):
        return str(function.get("arguments") or "")
    return ""
