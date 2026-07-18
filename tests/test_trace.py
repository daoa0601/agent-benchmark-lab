from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_benchmark_lab import JsonlTraceSink, TraceOverflowError, sanitize_trace_value


def test_trace_redacts_secret_content_and_private_reasoning(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    token = "sk-" + "A" * 24
    with JsonlTraceSink(trace_path) as trace:
        trace.emit(
            event="model.call",
            path="runner.step",
            data={
                "authorization": "Bearer private-token-value",
                "output": token,
                "chain_of_thought": "hidden",
            },
        )

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    serialized = json.dumps(record)
    assert token not in serialized
    assert "private-token-value" not in serialized
    assert "hidden" not in serialized
    assert record["data"]["_sanitized"] is True


def test_trace_fails_closed_before_exceeding_limit(tmp_path: Path) -> None:
    trace = JsonlTraceSink(tmp_path / "trace.jsonl", max_file_bytes=16)
    with pytest.raises(TraceOverflowError):
        trace.emit(event="large", path="runner", data={"value": "x"})
    trace.close()


def test_sanitizer_preserves_safe_primitive() -> None:
    assert sanitize_trace_value(42) == (42, False)


def test_sanitizer_handles_collections_bytes_and_unknown_objects() -> None:
    assert sanitize_trace_value(["safe", b"bytes"]) == (["safe", "bytes"], False)
    cleaned, changed = sanitize_trace_value(object())
    assert isinstance(cleaned, str)
    assert changed is True


def test_trace_rejects_invalid_limits_and_symbolic_links(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        JsonlTraceSink(tmp_path / "trace.jsonl", max_file_bytes=0)

    target = tmp_path / "target.jsonl"
    target.touch()
    link = tmp_path / "link.jsonl"
    os.symlink(target, link)
    with pytest.raises(ValueError, match="symbolic link"):
        JsonlTraceSink(link)


def test_trace_close_is_idempotent(tmp_path: Path) -> None:
    trace = JsonlTraceSink(tmp_path / "trace.jsonl")
    trace.close()
    trace.close()
