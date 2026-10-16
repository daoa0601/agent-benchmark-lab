from __future__ import annotations

from pathlib import Path

from agent_benchmark_lab import gate_session
from agenttrace.storage import FileObjectStore

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


def test_gate_persists_an_immutable_bundle_in_the_agenttrace_object_store(
    tmp_path: Path,
) -> None:
    manifest = PROJECT_ROOT / "examples/auditable-session.example.json"
    object_root = tmp_path / "objects"

    first = gate_session(manifest, report_path=tmp_path / "first.md", object_root=object_root)
    second = gate_session(manifest, report_path=tmp_path / "second.md", object_root=object_root)

    assert first.status == "passed"
    assert first.bundle_key is not None
    assert first.bundle_key.startswith("audits/bundles/")
    assert second.bundle_key == first.bundle_key
    assert len(list(object_root.glob("audits/bundles/*.json"))) == 1
    assert FileObjectStore(object_root).exists(first.bundle_key)
