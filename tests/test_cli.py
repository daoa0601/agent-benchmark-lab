from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_benchmark_lab.cli import main
from agent_benchmark_lab.gate import GateSummary
from agent_benchmark_lab.providers.zai import ZAIChatResponse, ZAIClient, ZAIProviderError
from agenttrace.models import TokenUsage
from agenttrace.storage import FileObjectStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evidence_command_prints_only_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("private observation", encoding="utf-8")

    assert main(["evidence", str(artifact), "--uri", "object://example/artifact.txt"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["uri"] == "object://example/artifact.txt"
    assert "private observation" not in captured.out


def test_gate_command_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = tmp_path / "report.md"

    assert (
        main(
            [
                "gate",
                str(PROJECT_ROOT / "examples/auditable-session.example.json"),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "passed"
    assert report.exists()


def test_gate_command_uses_distinct_blocked_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = GateSummary(
        session_id="blocked-session",
        session_digest="a" * 64,
        status="blocked",
        eligible_for_headline=False,
        headline_claim_ids=("claim",),
        failed_checks=("review",),
        report=str(tmp_path / "report.md"),
    )
    monkeypatch.setattr("agent_benchmark_lab.cli.gate_session", lambda *args, **kwargs: summary)

    assert main(["gate", "session.json", "--report", "report.md"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_command_failure_discloses_only_error_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "sensitive-name.txt"

    assert main(["evidence", str(missing)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"status": "invalid", "error_type": "ValueError"}
    assert "sensitive-name" not in captured.err


EXAMPLE_TASKS = PROJECT_ROOT / "examples/tasks"
SUPPORT_ARGS = [
    str(EXAMPLE_TASKS / "support-routing.yaml"),
    str(EXAMPLE_TASKS / "support-routing.samples.jsonl"),
]
ROUTED = '{"intent":"billing_refund","route":"billing","urgency":"normal"}'


def test_zai_api_command_prints_only_a_safe_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "runtime-only-secret")

    def fake_complete(self, messages, *, max_tokens=64, temperature=0.0):
        del self, messages, max_tokens, temperature
        return ZAIChatResponse(
            content=ROUTED,
            model="glm-5.2",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=3, output_tokens=2),
            request_id="req-safe",
        )

    monkeypatch.setattr(ZAIClient, "complete", fake_complete)
    exit_code = main(
        ["benchmark", "zai-api", *SUPPORT_ARGS, "--output-root", str(tmp_path / "objects")]
    )

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["summary"]["required_pass_count"] == 1
    assert payload["content_printed"] is False
    assert FileObjectStore(tmp_path / "objects").exists(payload["manifest_key"])
    assert "billing_refund" not in captured.out
    assert "runtime-only-secret" not in captured.out + captured.err


def test_zai_api_command_exits_2_when_provider_calls_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "runtime-only-secret")

    def failing_complete(self, messages, *, max_tokens=64, temperature=0.0):
        del self, messages, max_tokens, temperature
        raise ZAIProviderError("denied", status_code=401, request_id=None)

    monkeypatch.setattr(ZAIClient, "complete", failing_complete)
    exit_code = main(
        ["benchmark", "zai-api", *SUPPORT_ARGS, "--output-root", str(tmp_path / "objects")]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["summary"]["provider_error_count"] == 1


def test_zai_api_command_rejects_invalid_reasoning_options(tmp_path: Path) -> None:
    exit_code = main(
        [
            "benchmark",
            "zai-api",
            *SUPPORT_ARGS,
            "--thinking-mode",
            "sometimes",
            "--output-root",
            str(tmp_path / "objects"),
        ]
    )

    assert exit_code == 2
    assert not (tmp_path / "objects").exists()


def test_zai_opencode_command_runs_one_isolated_process_per_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "real-coding-secret-123456")
    exit_code = main(
        [
            "benchmark",
            "zai-opencode",
            str(EXAMPLE_TASKS / "code-review.yaml"),
            str(EXAMPLE_TASKS / "code-review.samples.jsonl"),
            "--executable",
            "true",
            "--watchdog-seconds",
            "30",
            "--session-id",
            "cli-session",
            "--output-root",
            str(tmp_path / "objects"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["transport"] == "opencode"
    assert payload["session_id"] == "cli-session"
    assert payload["summary"]["sample_count"] == 1
    assert FileObjectStore(tmp_path / "objects").exists(payload["manifest_key"])
    assert "real-coding-secret" not in captured.out + captured.err


@pytest.mark.parametrize("command", ["zai-api", "zai-opencode"])
def test_benchmark_commands_report_missing_credentials_by_type_only(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_API_KEY_FILE", raising=False)
    task_args = (
        SUPPORT_ARGS
        if command == "zai-api"
        else [
            str(EXAMPLE_TASKS / "code-review.yaml"),
            str(EXAMPLE_TASKS / "code-review.samples.jsonl"),
        ]
    )

    exit_code = main(["benchmark", command, *task_args, "--output-root", str(tmp_path / "objects")])

    error = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert error["ok"] is False
    assert error["error_type"] == "ValueError"
    assert error["content_printed"] is False
