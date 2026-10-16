"""Shared deterministic fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenttrace.tasks import TaskDefinition, load_task

EXAMPLE_TASKS = Path(__file__).resolve().parents[1] / "examples/tasks"


@pytest.fixture
def support_task() -> TaskDefinition:
    return load_task(EXAMPLE_TASKS / "support-routing.yaml")
