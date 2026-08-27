from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cli.clients.model_client import ModelClientProtocol, create_model_client
from cli.clients.openrouter_client import OpenRouterClient
from cli.clients.openrouter_thread_store import OpenRouterThreadStore
from cli.core.config import ReviewConfig
from cli.core.exceptions import ConfigurationError, DotBotExecutionError


def _config(tmp_path: Path, **overrides: Any) -> ReviewConfig:
    values: dict[str, Any] = dict(
        github_token="token",
        repository="owner/repo",
        pr_number=7,
        model_provider="openrouter",
        openrouter_api_key="test-key",
        review_model="anthropic/claude-opus-4.7",
        act_model="anthropic/claude-opus-4.7",
        stream_output=False,
        debug_level=0,
        web_search_mode="disabled",
        repo_root=tmp_path,
    )
    values.update(overrides)
    return ReviewConfig(**values)


class _ScriptedTransport:
    """Fake non-streaming transport recording payloads, returning queued responses."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(copy.deepcopy(payload))
        if not self.responses:
            raise AssertionError("unexpected extra model request")
        return self.responses.pop(0)


def _text_response(text: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
    }


def _tool_response(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def _client(
    tmp_path: Path,
    transport: _ScriptedTransport,
    **overrides: Any,
) -> OpenRouterClient:
    return OpenRouterClient(
        _config(tmp_path, **overrides),
        transport=transport,
        thread_store=OpenRouterThreadStore(tmp_path / "threads"),
    )


def _load_single_thread(tmp_path: Path) -> list[dict[str, Any]] | None:
    store = OpenRouterThreadStore(tmp_path / "threads")
    files = list(store.directory.iterdir())
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    return payload.get("messages")


def test_execute_text_returns_plain_answer(tmp_path: Path) -> None:
    transport = _ScriptedTransport([_text_response("all done")])
    client = _client(tmp_path, transport)

    assert client.execute_text("inspect the repo") == "all done"

    assert len(transport.payloads) == 1
    payload = transport.payloads[0]
    assert payload["model"] == "anthropic/claude-opus-4.7"
    assert payload["tools"][0]["function"]["name"] == "read_file"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][-1] == {"role": "user", "content": "inspect the repo"}
    assert payload["reasoning"] == {"effort": "medium"}


def test_execute_text_runs_tool_then_answers(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("file-content", encoding="utf-8")
    transport = _ScriptedTransport(
        [
            _tool_response("call-1", "read_file", json.dumps({"path": "a.txt"})),
            _text_response("the file says file-content"),
        ]
    )
    client = _client(tmp_path, transport)

    assert client.execute_text("read a.txt") == "the file says file-content"

    second_payload = transport.payloads[1]
    tool_messages = [m for m in second_payload["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call-1"
    assert tool_messages[0]["content"] == "file-content"
    assistant_with_tools = second_payload["messages"][-2]
    assert assistant_with_tools["tool_calls"][0]["id"] == "call-1"


def test_execute_text_empty_answer_raises(tmp_path: Path) -> None:
    transport = _ScriptedTransport([_text_response("   ")])
    client = _client(tmp_path, transport)

    with pytest.raises(DotBotExecutionError, match="did not return an agent message"):
        client.execute_text("hello")


def test_execute_text_missing_api_key_raises(tmp_path: Path) -> None:
    transport = _ScriptedTransport([])
    client = _client(tmp_path, transport, openrouter_api_key="")

    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        client.execute_text("hello")


def test_execute_structured_returns_schema_turn_output(tmp_path: Path) -> None:
    review_json = json.dumps({"findings": [], "overall_correctness": "patch is correct"})
    transport = _ScriptedTransport(
        [
            _text_response("I inspected everything."),
            _text_response(review_json),
        ]
    )
    client = _client(tmp_path, transport)

    output = client.execute_structured(
        "review this PR",
        output_schema={"type": "object"},
        schema_prompt="Produce the JSON review output now.",
    )
    assert output == review_json

    agent_payload, schema_payload = transport.payloads
    assert "tools" in agent_payload
    assert "response_format" not in agent_payload
    assert "tools" not in schema_payload
    assert schema_payload["response_format"]["type"] == "json_schema"
    assert schema_payload["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert schema_payload["messages"][-1] == {
        "role": "user",
        "content": "Produce the JSON review output now.",
    }

    # The schema-turn answer is persisted on the thread so resumed
    # conversations carry the final structured output.
    saved = _load_single_thread(tmp_path)
    assert saved is not None
    assert saved[-1] == {"role": "assistant", "content": review_json}


def test_execute_structured_empty_output_raises(tmp_path: Path) -> None:
    transport = _ScriptedTransport([_text_response("ok"), _text_response("")])
    client = _client(tmp_path, transport)

    with pytest.raises(DotBotExecutionError, match="structured output"):
        client.execute_structured("review", output_schema={"type": "object"})


def test_tool_iteration_cap_forces_final_answer(tmp_path: Path) -> None:
    transport = _ScriptedTransport(
        [
            _tool_response("c1", "read_file", '{"path": "a.txt"}'),
            _tool_response("c2", "read_file", '{"path": "a.txt"}'),
            _text_response("forced final answer"),
        ]
    )
    client = OpenRouterClient(
        _config(tmp_path),
        transport=transport,
        thread_store=OpenRouterThreadStore(tmp_path / "threads"),
        max_tool_iterations=2,
    )

    assert client.execute_text("loop forever") == "forced final answer"

    forced_payload = transport.payloads[2]
    assert "tools" not in forced_payload
    assert forced_payload["messages"][-1]["content"].startswith("Tool iteration limit reached")


def test_resume_continues_stored_thread(tmp_path: Path) -> None:
    store = OpenRouterThreadStore(tmp_path / "threads")
    stored_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prior prompt"},
        {"role": "assistant", "content": "prior answer"},
    ]
    store.save(
        "thread-1",
        stored_messages,
        repository="owner/repo",
        pr_number=7,
        model="anthropic/claude-opus-4.7",
    )
    transport = _ScriptedTransport([_text_response("resumed answer")])
    client = OpenRouterClient(
        _config(tmp_path),
        transport=transport,
        thread_store=store,
    )

    assert client.execute_text("next prompt", resume_thread_id="thread-1") == "resumed answer"

    messages = transport.payloads[0]["messages"]
    assert messages[:3] == stored_messages
    assert messages[3] == {"role": "user", "content": "next prompt"}

    updated = store.load("thread-1")
    assert updated is not None
    assert updated[-1] == {"role": "assistant", "content": "resumed answer"}


def test_resume_with_unknown_thread_starts_fresh(tmp_path: Path) -> None:
    transport = _ScriptedTransport([_text_response("fresh answer")])
    client = _client(tmp_path, transport)

    assert client.execute_text("hello", resume_thread_id="missing") == "fresh answer"
    messages = transport.payloads[0]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"


def test_live_web_search_mode_appends_online_suffix_to_agent_turns(tmp_path: Path) -> None:
    transport = _ScriptedTransport(
        [
            _text_response("agent"),
            _text_response("{}"),
        ]
    )
    client = _client(tmp_path, transport, web_search_mode="live")

    client.execute_structured("review", output_schema={"type": "object"})

    agent_model = transport.payloads[0]["model"]
    schema_model = transport.payloads[1]["model"]
    assert agent_model == "anthropic/claude-opus-4.7:online"
    assert schema_model == "anthropic/claude-opus-4.7"


def test_model_name_argument_overrides_config(tmp_path: Path) -> None:
    transport = _ScriptedTransport([_text_response("ok")])
    client = _client(tmp_path, transport)

    client.execute_text("hello", model_name="openai/gpt-5")

    assert transport.payloads[0]["model"] == "openai/gpt-5"


def test_thread_persisted_after_execute_text(tmp_path: Path) -> None:
    store = OpenRouterThreadStore(tmp_path / "threads")
    transport = _ScriptedTransport([_text_response("answer")])
    client = OpenRouterClient(_config(tmp_path), transport=transport, thread_store=store)

    client.execute_text("hello")

    files = list(store.directory.iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("openrouter-thread-v1-owner-repo-pr-7-")


def test_create_model_client_returns_openrouter_by_default(tmp_path: Path) -> None:
    client = create_model_client(_config(tmp_path))
    assert isinstance(client, OpenRouterClient)
    assert isinstance(client, ModelClientProtocol)


def test_invalid_reasoning_effort_falls_back_to_medium(tmp_path: Path) -> None:
    transport = _ScriptedTransport([_text_response("ok")])
    client = _client(tmp_path, transport)

    client.execute_text("hello", reasoning_effort="extreme")

    assert transport.payloads[0]["reasoning"] == {"effort": "medium"}


def test_summarize_tool_call_formats_common_tools() -> None:
    from cli.clients.openrouter_client import _summarize_tool_call

    assert (
        _summarize_tool_call("run_command", json.dumps({"command": "git diff main...HEAD"}))
        == "git diff main...HEAD"
    )
    assert (
        _summarize_tool_call("read_file", json.dumps({"path": "src/main.py"}))
        == "path='src/main.py'"
    )
    assert (
        _summarize_tool_call("write_file", json.dumps({"path": "a.py", "content": "x" * 10}))
        == "path='a.py' (10 chars)"
    )
    # Long commands are truncated.
    long_command = "git log " + "a" * 500
    summary = _summarize_tool_call("run_command", json.dumps({"command": long_command}))
    assert len(summary) <= 120
    assert summary.endswith("...")
    # Malformed JSON degrades to a raw-arguments dump, never raises.
    assert _summarize_tool_call("run_command", "{not json") == "{not json"


def test_debug_level_1_logs_tool_calls_and_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = _ScriptedTransport(
        [
            _tool_response("call-1", "run_command", json.dumps({"command": "git diff main"})),
            _text_response("all done"),
        ]
    )
    # Attach usage to both responses.
    transport.responses[0]["usage"] = {"prompt_tokens": 100, "completion_tokens": 40}
    transport.responses[1]["usage"] = {"prompt_tokens": 200, "completion_tokens": 60}
    client = _client(tmp_path, transport, debug_level=1)

    client.execute_text("inspect the repo")

    stderr = capsys.readouterr().err
    assert "[openrouter-tool] run_command: git diff main" in stderr
    assert "[openrouter-request] model=anthropic/claude-opus-4.7 request#1 elapsed=" in stderr
    assert (
        "[openrouter-usage] model=anthropic/claude-opus-4.7 requests=2 "
        "prompt_tokens=300 completion_tokens=100 total_tokens=0"
    ) in stderr


def test_no_usage_summary_without_usage_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = _ScriptedTransport([_text_response("all done")])
    client = _client(tmp_path, transport, debug_level=1)

    client.execute_text("inspect the repo")

    assert "[openrouter-usage]" not in capsys.readouterr().err


def _reasoning_chunk(reasoning: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"reasoning": reasoning}}]}


def test_reasoning_chars_summarized_in_stream_logs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def _frame(payload: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode()

    def _content(text: str, finish: str) -> dict[str, Any]:
        return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}

    def _stream_transport(payload: dict[str, Any]) -> Any:
        yield _frame(_reasoning_chunk("thinking hard..."))
        yield _frame(_content("final answer", "stop"))

    config = _config(tmp_path, debug_level=1, stream_output=True)
    client = OpenRouterClient(
        config,
        stream_transport=_stream_transport,
        thread_store=OpenRouterThreadStore(tmp_path / "threads"),
    )

    assert client.execute_text("inspect the repo") == "final answer"

    stderr = capsys.readouterr().err
    assert "[openrouter-reasoning] 16 chars of reasoning tokens this request" in stderr


def test_schema_turn_logs_elapsed_and_accumulates_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = _ScriptedTransport(
        [
            _text_response("thinking"),
            _text_response('{"overall_correctness": "patch is correct"}'),
        ]
    )
    transport.responses[1]["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}
    client = _client(tmp_path, transport, debug_level=1)

    client.execute_structured("review", output_schema={"type": "object"})

    stderr = capsys.readouterr().err
    assert "(schema turn) elapsed=" in stderr
    assert "prompt_tokens=10" in stderr
