"""Bounded JSONL trace capture using AgentTrace's secret and private-reasoning redaction rules."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from agenttrace.redaction import PRIVATE_REASONING_KEYS, SECRET_KEYS, SECRET_PATTERNS

_FIELD_CAP = 8 * 1024
_DEFAULT_FILE_CAP = 50 * 1024 * 1024


class TraceOverflowError(RuntimeError):
    """Raised before a trace record would exceed the configured hard cap."""


def sanitize_trace_value(value: Any, *, key: str | None = None) -> tuple[Any, bool]:
    """Return a JSON-safe value and whether it was redacted, truncated, or coerced."""

    normalized = key.casefold() if key else None
    if normalized in SECRET_KEYS:
        return "[REDACTED_SECRET]", True
    if normalized in PRIVATE_REASONING_KEYS:
        return "[PRIVATE_REASONING_REMOVED]", True
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        changed = False
        for child_key, child_value in value.items():
            cleaned, child_changed = sanitize_trace_value(child_value, key=str(child_key))
            output[str(child_key)] = cleaned
            changed = changed or child_changed
        return output, changed
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        output_list: list[Any] = []
        changed = False
        for child in value:
            cleaned, child_changed = sanitize_trace_value(child)
            output_list.append(cleaned)
            changed = changed or child_changed
        return output_list, changed
    if isinstance(value, bytes | bytearray):
        cleaned = bytes(value[:_FIELD_CAP]).decode("utf-8", errors="replace")
        return cleaned, len(value) > _FIELD_CAP
    if isinstance(value, str):
        cleaned = value[:_FIELD_CAP]
        changed = len(value) > _FIELD_CAP
        for pattern in SECRET_PATTERNS:
            cleaned, replacements = pattern.subn("[REDACTED_SECRET]", cleaned)
            changed = changed or replacements > 0
        return cleaned, changed
    if value is None or isinstance(value, bool | int | float):
        return value, False
    return str(value)[:_FIELD_CAP], True


class JsonlTraceSink:
    """Thread-safe, append-only JSONL sink with a fail-closed size limit."""

    def __init__(self, path: str | Path, *, max_file_bytes: int = _DEFAULT_FILE_CAP) -> None:
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("trace path must not be a symbolic link")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            self._file: TextIO = os.fdopen(descriptor, "a", encoding="utf-8")
        except Exception:
            os.close(descriptor)
            raise
        self._bytes_written = self.path.stat().st_size
        self._max_file_bytes = max_file_bytes
        self._lock = threading.Lock()

    def emit(self, *, event: str, path: str, data: Mapping[str, Any]) -> None:
        """Append one sanitized event or raise before exceeding the file cap."""

        safe_data, changed = sanitize_trace_value(data)
        if not isinstance(safe_data, dict):
            raise TypeError("trace data must sanitize to a mapping")
        if changed:
            safe_data["_sanitized"] = True
        record = {"ts": time.time(), "event": event, "path": path, "data": safe_data}
        line = json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        encoded_size = len(line.encode("utf-8"))
        with self._lock:
            if self._bytes_written + encoded_size > self._max_file_bytes:
                raise TraceOverflowError(
                    f"trace file exceeded its configured {self._max_file_bytes}-byte limit"
                )
            self._file.write(line)
            self._file.flush()
            self._bytes_written += encoded_size

    def close(self) -> None:
        """Close the trace file exactly once."""

        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> JsonlTraceSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
