from __future__ import annotations

import json
from pathlib import Path

from agent_benchmark_lab import describe_evidence


def test_describe_evidence_hashes_without_copying_content(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('{"latency_us":[12.0,13.0]}\n', encoding="utf-8")

    evidence = describe_evidence(artifact, uri="object://audit/samples.json")

    assert evidence["uri"] == "object://audit/samples.json"
    assert len(evidence["sha256"]) == 64
    assert evidence["byte_size"] == artifact.stat().st_size
    assert evidence["media_type"] == "application/json"
    assert "latency_us" not in json.dumps(evidence)
