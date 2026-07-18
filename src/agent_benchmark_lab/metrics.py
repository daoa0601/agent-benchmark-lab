"""Small outcome summaries with explicit harness-versus-judge semantics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Outcome:
    """One benchmark outcome.

    `submitted` means the harness produced a candidate. `resolved` is only true or false when an
    explicit judge supplied that result; `None` means the result is still unknown.
    """

    submitted: bool
    resolved: bool | None = None


@dataclass(frozen=True, slots=True)
class OutcomeSummary:
    """Aggregate counts and rates without treating submission as success."""

    total: int
    submitted: int
    resolved: int
    unresolved: int
    resolution_unknown: int

    @property
    def submission_rate(self) -> float | None:
        return self.submitted / self.total if self.total else None

    @property
    def resolution_rate(self) -> float | None:
        judged = self.resolved + self.unresolved
        return self.resolved / judged if judged else None


def summarize_outcomes(outcomes: Iterable[Outcome]) -> OutcomeSummary:
    """Summarize outcomes while preserving unknown judge results."""

    values = tuple(outcomes)
    return OutcomeSummary(
        total=len(values),
        submitted=sum(outcome.submitted for outcome in values),
        resolved=sum(outcome.resolved is True for outcome in values),
        unresolved=sum(outcome.resolved is False for outcome in values),
        resolution_unknown=sum(outcome.resolved is None for outcome in values),
    )
