from agent_benchmark_lab import Outcome, summarize_outcomes


def test_submission_is_not_treated_as_resolution() -> None:
    summary = summarize_outcomes(
        [Outcome(submitted=True), Outcome(submitted=False, resolved=False)]
    )

    assert summary.total == 2
    assert summary.submitted == 1
    assert summary.resolved == 0
    assert summary.unresolved == 1
    assert summary.resolution_unknown == 1
    assert summary.submission_rate == 0.5
    assert summary.resolution_rate == 0.0


def test_empty_outcomes_have_undefined_rates() -> None:
    summary = summarize_outcomes([])

    assert summary.submission_rate is None
    assert summary.resolution_rate is None
