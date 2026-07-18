"""Apply the portable benchmark audit gate and optionally persist its immutable bundle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from benchmark_audit_core import AuditableSession, build_audit_bundle, render_audit_report


@dataclass(frozen=True, slots=True)
class GateSummary:
    """Safe command output for one gated benchmark session."""

    session_id: str
    session_digest: str
    status: str
    eligible_for_headline: bool
    headline_claim_ids: tuple[str, ...]
    failed_checks: tuple[str, ...]
    report: str
    bundle_key: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _persist_bundle(bundle: object, object_root: Path) -> str:
    """Load AgentTrace storage only when the caller requests persistence."""

    try:
        audits = import_module("agenttrace.audits")
        objects = import_module("agenttrace.storage.objects")
    except ImportError as error:
        raise RuntimeError(
            "object-store persistence requires the sibling agenttrace-platform source"
        ) from error
    stored = audits.persist_audit_bundle(objects.FileObjectStore(object_root), bundle)
    return str(stored.key)


def gate_session(
    manifest_path: str | Path,
    *,
    report_path: str | Path,
    object_root: str | Path | None = None,
) -> GateSummary:
    """Validate, gate, render, and optionally persist one session manifest."""

    manifest = Path(manifest_path)
    report = Path(report_path)
    session = AuditableSession.model_validate_json(manifest.read_text(encoding="utf-8"))
    bundle = build_audit_bundle(session)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_audit_report(bundle), encoding="utf-8")
    bundle_key = _persist_bundle(bundle, Path(object_root)) if object_root is not None else None
    gate = bundle.headline_gate
    return GateSummary(
        session_id=session.session_id,
        session_digest=gate.session_digest,
        status=gate.status.value,
        eligible_for_headline=gate.eligible_for_headline,
        headline_claim_ids=tuple(gate.headline_claim_ids),
        failed_checks=tuple(check.check_id for check in gate.checks if not check.passed),
        report=str(report.resolve()),
        bundle_key=bundle_key,
    )
