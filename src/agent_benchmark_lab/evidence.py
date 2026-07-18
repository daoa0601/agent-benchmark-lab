"""Content-addressed evidence descriptions that never copy or print artifact contents."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from benchmark_audit_core import evidence_ref_from_file


class EvidenceDescription(TypedDict):
    """Portable, serializable metadata for one local evidence file."""

    uri: str
    sha256: str
    media_type: str
    byte_size: int


def describe_evidence(
    path: str | Path,
    *,
    uri: str | None = None,
    media_type: str | None = None,
) -> EvidenceDescription:
    """Hash a regular file and return only its portable evidence metadata."""

    evidence = evidence_ref_from_file(path, uri=uri, media_type=media_type)
    if evidence.byte_size is None:
        raise RuntimeError("file-backed evidence did not include a byte count")
    return {
        "uri": evidence.uri,
        "sha256": evidence.sha256,
        "media_type": evidence.media_type,
        "byte_size": evidence.byte_size,
    }
