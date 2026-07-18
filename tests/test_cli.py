from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_benchmark_lab.cli import main
from agent_benchmark_lab.gate import GateSummary

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
