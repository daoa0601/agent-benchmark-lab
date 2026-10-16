"""Fresh per-sample workspace snapshots and change manifests."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agenttrace.storage import ObjectStore

EXCLUDED_NAMES = {
    ".git",
    ".opencode",
    ".env",
    ".env.local",
    "opencode.json",
    "opencode.jsonc",
    "secrets",
}


class WorkspaceChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    status: Literal["added", "modified", "deleted"]
    before_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    after_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    after_ref: str | None = None
    after_bytes: int | None = Field(default=None, ge=0)
    # Git-style entry mode: 100644 file, 100755 executable file, 120000 symlink (content is the
    # link target, never what it points to).
    after_mode: str | None = Field(default=None, pattern=r"^(100644|100755|120000)$")


@dataclass(frozen=True)
class IsolatedWorkspace:
    path: Path
    source: Path | None
    excluded_names: tuple[str, ...]


def _excluded(name: str) -> bool:
    return name in EXCLUDED_NAMES or name.startswith(".env.")


def _validate_symlinks(source: Path) -> None:
    for root, directories, files in os.walk(source, followlinks=False):
        directories[:] = [name for name in directories if not _excluded(name)]
        for name in [*directories, *files]:
            path = Path(root) / name
            if not path.is_symlink():
                continue
            target = path.readlink()
            if target.is_absolute():
                raise ValueError(f"workspace contains an absolute symlink: {path}")
            resolved = (path.parent / target).resolve()
            if source.resolve() not in resolved.parents and resolved != source.resolve():
                raise ValueError(f"workspace symlink escapes the source tree: {path}")


@contextmanager
def isolated_workspace(
    source: Path | None, *, exclude: tuple[Path, ...] = ()
) -> Iterator[IsolatedWorkspace]:
    """Copy a source tree without VCS, local agent config, or common secret files.

    ``exclude`` names directories, such as an output root inside the source, left out entirely.
    """
    excluded_paths = {path.expanduser().resolve() for path in exclude}

    def ignore(directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if _excluded(name) or (Path(directory) / name).resolve() in excluded_paths
        }

    with TemporaryDirectory(prefix="agenttrace-opencode-workspace-") as temp_dir:
        root = Path(temp_dir)
        workspace = root / "workspace"
        if source is None:
            workspace.mkdir()
            resolved_source = None
        else:
            resolved_source = source.expanduser().resolve()
            if not resolved_source.is_dir():
                raise ValueError(f"workspace source is not a directory: {resolved_source}")
            _validate_symlinks(resolved_source)
            shutil.copytree(
                resolved_source,
                workspace,
                symlinks=True,
                ignore=ignore,
            )
        yield IsolatedWorkspace(
            path=workspace,
            source=resolved_source,
            excluded_names=tuple(sorted(EXCLUDED_NAMES)),
        )


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


Snapshot = dict[str, tuple[str, int, str]]
"""Relative path to (SHA-256, byte count, Git-style mode) of every file and symlink."""


def _entry(path: Path) -> tuple[str, int, str] | None:
    if path.is_symlink():
        target = os.readlink(path).encode("utf-8", "surrogateescape")
        return hashlib.sha256(target).hexdigest(), len(target), "120000"
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        return None
    digest, size = _sha256(path)
    return digest, size, "100755" if metadata.st_mode & 0o111 else "100644"


def snapshot_workspace(workspace: Path) -> Snapshot:
    snapshot: Snapshot = {}
    for root, directories, files in os.walk(workspace, followlinks=False):
        # A symlinked directory is recorded as a link and never descended into.
        linked = [name for name in directories if (Path(root) / name).is_symlink()]
        directories[:] = sorted(
            name for name in directories if not _excluded(name) and name not in linked
        )
        for name in sorted([*files, *linked]):
            if _excluded(name):
                continue
            path = Path(root) / name
            entry = _entry(path)
            if entry is not None:
                snapshot[path.relative_to(workspace).as_posix()] = entry
    return snapshot


def workspace_changes(
    *,
    workspace: Path,
    before: Snapshot,
    objects: ObjectStore,
) -> list[WorkspaceChange]:
    after = snapshot_workspace(workspace)
    changes: list[WorkspaceChange] = []
    for relative in sorted(before.keys() | after.keys()):
        previous = before.get(relative)
        current = after.get(relative)
        if previous == current:
            continue
        if previous is None:
            status = "added"
        elif current is None:
            status = "deleted"
        else:
            status = "modified"
        after_ref = None
        mode: str | None = None
        if current is not None:
            digest, size, mode = current
            path = workspace / relative
            content = (
                os.readlink(path).encode("utf-8", "surrogateescape")
                if mode == "120000"
                else path.read_bytes()
            )
            key = f"benchmarks/workspaces/sha256/{digest[:2]}/{digest}.bin"
            objects.put_bytes(key, content, if_absent=True)
            after_ref = f"object://{key}"
        else:
            digest, size = None, None
        changes.append(
            WorkspaceChange(
                path=relative,
                status=status,
                before_sha256=previous[0] if previous is not None else None,
                after_sha256=digest,
                after_ref=after_ref,
                after_bytes=size,
                after_mode=mode,
            )
        )
    return changes
