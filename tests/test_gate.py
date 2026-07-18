from __future__ import annotations

from pathlib import Path

import pytest

from agent_benchmark_lab import gate_session
from agent_benchmark_lab.gate import _persist_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_example_passes_and_renders_findings(tmp_path: Path) -> None:
    report = tmp_path / "audit-report.md"
    summary = gate_session(
        PROJECT_ROOT / "examples/auditable-session.example.json",
        report_path=report,
    )

    assert summary.status == "passed"
    assert summary.eligible_for_headline is True
    assert summary.bundle_key is None
    assert "## Audit findings" in report.read_text(encoding="utf-8")


def test_agenttrace_is_loaded_only_when_persistence_is_requested(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="agenttrace-platform"):
        gate_session(
            PROJECT_ROOT / "examples/auditable-session.example.json",
            report_path=tmp_path / "audit-report.md",
            object_root=tmp_path / "objects",
        )


def test_agenttrace_persistence_returns_the_bundle_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Stored:
        key = "audits/bundles/example.json"

    class Audits:
        @staticmethod
        def persist_audit_bundle(store: object, bundle: object) -> Stored:
            return Stored()

    class Objects:
        class FileObjectStore:
            def __init__(self, root: Path) -> None:
                self.root = root

    modules = {"agenttrace.audits": Audits, "agenttrace.storage.objects": Objects}
    monkeypatch.setattr("agent_benchmark_lab.gate.import_module", modules.__getitem__)

    assert _persist_bundle(object(), tmp_path) == "audits/bundles/example.json"
