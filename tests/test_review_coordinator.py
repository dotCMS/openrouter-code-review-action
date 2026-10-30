from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, cast

import pytest

from cli.core.config import ReviewConfig
from cli.core.exceptions import DotBotExecutionError, ReviewContractError
from cli.core.models import PriorDotBotReviewComment
from cli.review.coordinator import PanelTurn, ReviewCoordinator
from cli.review.review_prompt import load_guidelines, load_panel_instructions


def _config(tmp_path: Path, *, policy: str = "unanimous") -> ReviewConfig:
    return ReviewConfig(
        github_token="SECRET-GITHUB",
        repository="owner/repo",
        pr_number=7,
        review_model="primary",
        review_models=("secondary",),
        repo_root=tmp_path,
        review_decision_policy=policy,
        openrouter_api_key="SECRET-MODEL",
        model_timeout_seconds=3,
    )


def _ledger(prompt: str) -> list[dict[str, Any]]:
    return json.loads(
        prompt.split("Issue ledger (data):\n", 1)[1].split("\nPrevious completed rounds", 1)[0]
    )


def _history(prompt: str) -> list[dict[str, Any]]:
    return json.loads(
        prompt.split("Previous completed rounds (data):\n", 1)[1].split(
            "\nYour reviewer identity:", 1
        )[0]
    )


def _finding(tmp_path: Path, *, title: str = "[P1] Validate authorization") -> dict[str, Any]:
    return {
        "title": title,
        "body": "An unauthorized caller reaches the privileged operation.",
        "confidence_score": 0.9,
        "priority": 1,
        "code_location": {
            "absolute_file_path": str(tmp_path / "src.py"),
            "line_range": {"start": 1, "end": 1},
        },
    }


def _turn(
    *,
    findings: list[dict[str, Any]] | None = None,
    votes: dict[str, bool] | None = None,
    correct: bool | None = None,
    message: str = "Check the authorization path in src.py:1.",
) -> str:
    findings = findings or []
    votes = votes or {}
    if correct is None:
        correct = not findings and not any(votes.values())
    return json.dumps(
        {
            "overall_correctness": "patch is correct" if correct else "patch is incorrect",
            "overall_explanation": "Final assessment of the checked code.",
            "overall_confidence_score": 0.9,
            "findings": findings,
            "carried_forward": [],
            "message": message,
            "security_summary": "Checked authorization; no runtime tests performed.",
            "dispositions": {
                key: {
                    "action": "retain" if value else "reject",
                    "reason": "The caller path supports this position.",
                    "evidence": "src.py:1: authorize(user)",
                }
                for key, value in votes.items()
            },
        }
    )


class _Client:
    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute_structured(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append((prompt, kwargs))
        return self.handler(prompt, kwargs)


def _panel(config: ReviewConfig, handler: Any) -> tuple[ReviewCoordinator, dict[str, _Client]]:
    clients: dict[str, _Client] = {}

    def factory(isolated: ReviewConfig) -> Any:
        assert isolated is not config
        assert isolated.stream_output is False
        assert isolated.debug_level == 0
        client = _Client(handler)
        clients[isolated.review_model] = client
        return client

    return ReviewCoordinator(config, factory), clients


def _run(
    panel: ReviewCoordinator, tmp_path: Path, prior: list[PriorDotBotReviewComment] | None = None
) -> Any:
    return panel.run(
        "Pinned diff",
        prior_comments=prior or [],
        transcript_path=tmp_path / "discussion.json",
        head_sha="a" * 40,
    )


def test_investigations_overlap_and_rounds_broadcast_complete_ordered_history(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2, timeout=2)

    def handler(prompt: str, kwargs: dict[str, Any]) -> str:
        history = _history(prompt)
        assert len(history) in {0, 2, 4}
        if not history:
            barrier.wait()
        else:
            assert [entry["model"] for entry in history[:2]] == ["primary", "secondary"]
        assert "authorization, trust boundaries, injection" in prompt
        assert kwargs["suppress_stream"] is True
        assert kwargs["sandbox_mode"] == "read-only"
        assert kwargs["resume_thread_id"] is None
        return _turn()

    config = _config(tmp_path)
    panel, clients = _panel(config, handler)
    result = _run(panel, tmp_path)
    assert result.overall_correctness == "patch is correct"
    assert all(len(client.calls) == 3 for client in clients.values())
    assert config.review_model == "primary"
    assert config.stream_output is True
    assert [entry["model"] for entry in panel.transcript] == ["primary", "secondary"] * 3


@pytest.mark.parametrize("retract", [False, True])
def test_union_preserves_minority_findings_but_all_can_retract(
    tmp_path: Path, retract: bool
) -> None:
    def handler(prompt: str, kwargs: dict[str, Any]) -> str:
        ledger = _ledger(prompt)
        if not _history(prompt):
            # Identical independent findings get one stable ID, not duplicate comments.
            return _turn(findings=[_finding(tmp_path)])
        keep = kwargs["model_name"] == "primary" and not (
            retract and "Phase: discussion-2." in prompt
        )
        return _turn(votes={item["issue_id"]: keep for item in ledger})

    panel, _ = _panel(_config(tmp_path), handler)
    result = _run(panel, tmp_path)
    assert len(panel.issues) == 1
    assert len(result.findings) == (0 if retract else 1)
    assert result.overall_correctness == ("patch is correct" if retract else "patch is incorrect")
    assert panel.transcript[0]["new_issue_ids"] == ["F1"]
    assert panel.transcript[1]["new_issue_ids"] == ["F1"]


def test_distinct_issues_at_same_line_are_not_collapsed(tmp_path: Path) -> None:
    def handler(prompt: str, kwargs: dict[str, Any]) -> str:
        if not _history(prompt):
            return _turn(findings=[_finding(tmp_path, title=kwargs["model_name"])])
        return _turn(votes={item["issue_id"]: True for item in _ledger(prompt)})

    panel, _ = _panel(_config(tmp_path), handler)
    assert len(_run(panel, tmp_path).findings) == 2


def test_lead_can_reject_disputed_finding_after_evidence_based_adjudication(tmp_path: Path) -> None:
    def handler(prompt: str, kwargs: dict[str, Any]) -> str:
        if not _history(prompt):
            return _turn(
                findings=[_finding(tmp_path)] if kwargs["model_name"] == "secondary" else []
            )
        keep = kwargs["model_name"] == "secondary"
        return _turn(votes={item["issue_id"]: keep for item in _ledger(prompt)})

    panel, clients = _panel(_config(tmp_path, policy="lead"), handler)
    result = _run(panel, tmp_path)
    assert result.overall_correctness == "patch is correct"
    assert result.findings == []
    assert len(clients["primary"].calls) == 4
    assert len(clients["secondary"].calls) == 3
    assert panel.transcript[-1]["phase"] == "adjudication"
    assert "all reviewer models" not in result.overall_explanation


def test_missing_lead_disposition_fails_instead_of_implicitly_approving(tmp_path: Path) -> None:
    def handler(prompt: str, kwargs: dict[str, Any]) -> str:
        if "Phase: adjudication." in prompt:
            return _turn()
        if not _history(prompt):
            return _turn(findings=[_finding(tmp_path)])
        return _turn(votes={item["issue_id"]: True for item in _ledger(prompt)})

    panel, _ = _panel(_config(tmp_path, policy="lead"), handler)
    with pytest.raises(DotBotExecutionError, match="adjudication failed for primary"):
        _run(panel, tmp_path)


def test_prior_comments_require_explicit_dispositions(tmp_path: Path) -> None:
    prior = PriorDotBotReviewComment(
        id="comment-1",
        thread_id="thread-1",
        path="src.py",
        line=1,
        body="prior issue",
        current_code="authorize(user)",
        is_currently_applicable=True,
    )
    panel, _ = _panel(
        _config(tmp_path),
        lambda prompt, kwargs: _turn(votes={item["issue_id"]: True for item in _ledger(prompt)}),
    )
    result = _run(panel, tmp_path, [prior])
    assert result.carried_forward_comment_ids == ["comment-1"]
    assert result.overall_correctness == "patch is incorrect"


@pytest.mark.parametrize("failure", ["invalid-json", "exception", "timeout"])
def test_failed_reviewer_never_becomes_an_approval_vote(tmp_path: Path, failure: str) -> None:
    release = threading.Event()

    def handler(prompt: str, kwargs: dict[str, Any]) -> str:
        if kwargs["model_name"] == "secondary":
            if failure == "invalid-json":
                return "not json"
            if failure == "exception":
                raise RuntimeError("failed")
            release.wait(5)
        return _turn()

    config = _config(tmp_path, policy="lead")
    config.model_timeout_seconds = 1
    panel, _ = _panel(config, handler)
    try:
        with pytest.raises(DotBotExecutionError, match="investigation.*secondary"):
            _run(panel, tmp_path)
        assert not panel.transcript
    finally:
        release.set()


def test_transcript_redacts_credentials_and_only_logs_explicit_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    panel, _ = _panel(
        _config(tmp_path),
        lambda prompt, kwargs: _turn(message="Question SECRET-GITHUB SECRET-MODEL"),
    )
    _run(panel, tmp_path)
    logs = capsys.readouterr().out
    saved = (tmp_path / "discussion.json").read_text()
    for output in (logs, saved):
        assert "SECRET-GITHUB" not in output
        assert "SECRET-MODEL" not in output
        assert "[REDACTED]" in output
        assert "private_reasoning" not in output
    config = _config(tmp_path)
    config.stream_output = False
    panel, _ = _panel(config, lambda prompt, kwargs: _turn())
    _run(panel, tmp_path)
    assert capsys.readouterr().out == ""
    assert json.loads((tmp_path / "discussion.json").read_text())["turns"]


def test_panel_never_resumes_a_cached_verdict_and_roster_changes_start_fresh(
    tmp_path: Path,
) -> None:
    seen: list[int] = []

    def handler(prompt: str, kwargs: dict[str, Any]) -> str:
        seen.append(len(_history(prompt)))
        return _turn()

    config = _config(tmp_path)
    for policy in ("unanimous", "lead"):
        config.review_decision_policy = policy
        panel, _ = _panel(config, handler)
        _run(panel, tmp_path)
    assert seen.count(0) == 4


@pytest.mark.parametrize(
    "change",
    ["missing", "array", "unknown", "empty-evidence", "contradiction", "unknown-field"],
)
def test_panel_output_validation_is_fail_closed(change: str) -> None:
    payload = json.loads(_turn(votes={"F1": True}))
    if change == "missing":
        payload["dispositions"] = {}
    elif change == "array":
        payload["dispositions"] = []
    elif change == "unknown":
        payload["dispositions"]["F999"] = payload["dispositions"].pop("F1")
    elif change == "empty-evidence":
        payload["dispositions"]["F1"]["evidence"] = " "
    elif change == "contradiction":
        payload["overall_correctness"] = "patch is correct"
    else:
        payload["raw_reasoning"] = "not allowed"
    with pytest.raises(ReviewContractError):
        PanelTurn.parse(json.dumps(payload), {"F1"})


def test_provider_schema_requires_every_ledger_id_and_stays_round_local(tmp_path: Path) -> None:
    def handler(prompt: str, kwargs: dict[str, Any]) -> str:
        ids = {item["issue_id"] for item in _ledger(prompt)}
        schema = kwargs["output_schema"]["properties"]["dispositions"]
        assert schema["type"] == "object"
        assert schema["required"] == sorted(ids)
        assert set(schema["properties"]) == ids
        assert schema["additionalProperties"] is False
        return _turn(
            votes={issue_id: True for issue_id in ids},
            findings=[_finding(tmp_path)] if not ids else [],
        )

    panel, clients = _panel(_config(tmp_path), handler)
    _run(panel, tmp_path)
    for client in clients.values():
        first_schema = client.calls[0][1]["output_schema"]["properties"]["dispositions"]
        assert first_schema["properties"] == {}
        assert first_schema["required"] == []
        assert client.calls[-1][1]["output_schema"]["properties"]["dispositions"]["required"] == [
            "F1"
        ]


def test_duplicate_json_dispositions_fail_closed() -> None:
    output = _turn(votes={"F1": True})
    position = json.dumps(json.loads(output)["dispositions"]["F1"])
    output = output.replace('"F1":', f'"F1": {position}, "F1":', 1)
    with pytest.raises(ReviewContractError, match="duplicate field"):
        PanelTurn.parse(output, {"F1"})


def test_contract_failure_diagnostics_do_not_expose_credentials(tmp_path: Path) -> None:
    def handler(prompt: str, kwargs: dict[str, Any]) -> str:
        raise ReviewContractError("Invalid position SECRET-MODEL")

    panel, _ = _panel(_config(tmp_path), handler)
    with pytest.raises(DotBotExecutionError, match="Invalid position ") as caught:
        _run(panel, tmp_path)
    assert "SECRET-MODEL" not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


@pytest.mark.parametrize("correction", ["valid", "drop-finding", "change-vote", "still-invalid"])
def test_real_client_repairs_one_turn_without_erasing_findings_or_votes(
    tmp_path: Path,
    correction: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cli.clients.openrouter_client import OpenRouterClient
    from cli.clients.openrouter_thread_store import OpenRouterThreadStore

    config = _config(tmp_path)
    schema_calls = {"primary": 0, "secondary": 0}
    initial_calls = {"primary": 0, "secondary": 0}
    finding = _finding(tmp_path, title="[P2] Preserve the actionable finding")
    finding["priority"] = 2
    prior = PriorDotBotReviewComment(
        id="prior",
        thread_id="thread",
        path="src.py",
        line=1,
        body="Prior issue",
        current_code="authorize(user)",
        is_currently_applicable=True,
    )

    def factory(isolated: ReviewConfig) -> Any:
        model = isolated.review_model

        def transport(payload: dict[str, Any]) -> dict[str, Any]:
            text = "Investigation complete"
            if "response_format" in payload:
                schema_calls[model] += 1
                prompt = payload["messages"][1]["content"]
                votes = {item["issue_id"]: True for item in _ledger(prompt)}
                if "Phase: investigation." in prompt and model == "primary":
                    initial_calls[model] += 1
                    if initial_calls[model] == 1 or correction == "still-invalid":
                        text = _turn(findings=[finding], votes=votes, correct=True)
                    else:
                        text = _turn(
                            findings=[] if correction == "drop-finding" else [finding],
                            votes={key: False for key in votes}
                            if correction == "change-vote"
                            else votes,
                            correct=False,
                        )
                else:
                    text = _turn(votes=votes)
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                ]
            }

        return OpenRouterClient(
            isolated, transport=transport, thread_store=OpenRouterThreadStore(tmp_path / model)
        )

    panel = ReviewCoordinator(config, factory)
    if correction == "valid":
        result = _run(panel, tmp_path, [prior])
        assert result.overall_correctness == "patch is incorrect"
        assert len(result.findings) == 1
        assert result.findings[0].priority == 2
        assert result.carried_forward_comment_ids == ["prior"]
        assert len(panel.transcript) == 6
        assert schema_calls == {"primary": 4, "secondary": 3}
    else:
        with pytest.raises(DotBotExecutionError, match="investigation failed for primary"):
            _run(panel, tmp_path, [prior])
        assert not panel.transcript
        assert initial_calls["primary"] == 2
    assert capsys.readouterr().out.count("[review-panel-correction]") == 1


def test_shared_client_is_rejected(tmp_path: Path) -> None:
    client = _Client(lambda prompt, kwargs: _turn())
    with pytest.raises(ReviewContractError, match="isolated model clients"):
        ReviewCoordinator(_config(tmp_path), lambda config: cast(Any, client))


@pytest.mark.parametrize("emojis", [("", ""), ("⚪ ", "🔴 "), ("⚪️ ", "🔴 ")])
def test_duplicates_preserve_highest_severity_and_redact_published_content(
    tmp_path: Path,
    emojis: tuple[str, str],
) -> None:
    def handler(prompt: str, kwargs: dict[str, Any]) -> str:
        if not _history(prompt):
            priority = 3 if kwargs["model_name"] == "primary" else 1
            emoji = emojis[0] if kwargs["model_name"] == "primary" else emojis[1]
            finding = _finding(tmp_path, title=f"{emoji}[P{priority}] Validate authorization")
            finding["priority"] = priority
            finding["body"] += " SECRET-GITHUB"
            return _turn(findings=[finding])
        return _turn(votes={item["issue_id"]: True for item in _ledger(prompt)})

    panel, _ = _panel(_config(tmp_path), handler)
    result = _run(panel, tmp_path)
    assert len(result.findings) == 1
    assert result.findings[0].priority == 1
    assert "SECRET-GITHUB" not in result.findings[0].body
    assert "[REDACTED]" in result.findings[0].body


@pytest.mark.parametrize("priority", [2, 3])
def test_panel_policy_overrides_nonblocking_verdict_guidance(tmp_path: Path, priority: int) -> None:
    finding = _finding(tmp_path)
    finding["priority"] = priority
    config = _config(tmp_path)
    for guidelines in (load_guidelines(config), load_panel_instructions(config)):
        assert "including P2/P3" in guidelines
        assert "at least one P0 or P1" not in guidelines
    with pytest.raises(ReviewContractError, match="contradicts retained findings"):
        PanelTurn.parse(_turn(findings=[finding], correct=True), set())
    assert (
        PanelTurn.parse(_turn(findings=[finding]), set()).review.overall_correctness
        == "patch is incorrect"
    )
