"""Inline direct-runner evaluators over provider-neutral task samples."""

from __future__ import annotations

from typing import Any

import pytest

from agent_benchmark_lab.benchmarks.evaluation import evaluate_response, validate_direct_task
from agenttrace.tasks import TaskDefinition, TaskSample


def _task(support_task: TaskDefinition, *evaluators: dict[str, Any], **updates: Any):
    payload = support_task.model_dump(mode="json")
    payload["evaluators"] = [{"version": "1.0.0", **spec} for spec in evaluators]
    payload.update(updates)
    return TaskDefinition.model_validate(payload)


def _sample(target: Any = None) -> TaskSample:
    return TaskSample(sample_id="sample-1", input={"message": "hello"}, target=target)


def test_direct_runner_rejects_required_environment_evaluators(support_task) -> None:
    custom = _task(support_task, {"evaluator_id": "hidden", "kind": "custom"})
    with pytest.raises(ValueError, match="cannot execute required evaluators"):
        validate_direct_task(custom)

    docker = _task(
        support_task,
        {"evaluator_id": "answer", "kind": "exact_match"},
        environment={"kind": "docker", "image": "sandbox:latest"},
    )
    with pytest.raises(ValueError, match="environment.kind=none"):
        validate_direct_task(docker)


def test_text_evaluators_score_matches_and_skip_environment_checks(support_task) -> None:
    task = _task(
        support_task,
        {
            "evaluator_id": "answer",
            "kind": "exact_match",
            "config": {"expected": "Billing", "case_sensitive": False},
        },
        {"evaluator_id": "terms", "kind": "contains", "config": {"values": ["bill", "refund"]}},
        {
            "evaluator_id": "shape",
            "kind": "regex",
            "config": {"pattern": "BILL", "ignore_case": True},
        },
        {"evaluator_id": "sandbox", "kind": "custom", "required": False},
    )

    outcomes, aggregate, required_passed = evaluate_response(
        task, _sample(), "  billing  ", latency_ms=1.0
    )

    by_id = {outcome.evaluator_id: outcome for outcome in outcomes}
    assert by_id["answer"].passed is True
    assert by_id["terms"].score == 0.5
    assert by_id["terms"].passed is False
    assert by_id["shape"].passed is True
    assert by_id["sandbox"].status == "skipped"
    assert required_passed is False
    assert aggregate == pytest.approx(2.5 / 3)


def test_reference_answer_is_the_fallback_expectation(support_task) -> None:
    task = _task(support_task, {"evaluator_id": "answer", "kind": "exact_match"})
    reference = '{"intent":"billing_refund","route":"billing","urgency":"normal"}'

    outcomes, _, required_passed = evaluate_response(task, _sample(), reference, latency_ms=1.0)

    assert outcomes[0].passed is True
    assert required_passed is True


def test_json_schema_evaluator_reports_invalid_and_non_json_output(support_task) -> None:
    task = _task(support_task, {"evaluator_id": "schema", "kind": "json_schema"})

    not_json, _, _ = evaluate_response(task, _sample(), "plain text", latency_ms=1.0)
    wrong_shape, _, _ = evaluate_response(task, _sample(), '{"intent": 1}', latency_ms=1.0)

    assert not_json[0].details == {"validation_error": "output is not valid JSON"}
    assert wrong_shape[0].passed is False
    assert "validation_error" in wrong_shape[0].details


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"evaluator_id": "shape", "kind": "regex"}, "requires config.pattern"),
        (
            {"evaluator_id": "fast", "kind": "latency_budget", "config": {"max_latency_ms": "1"}},
            "numeric config.max_latency_ms",
        ),
        (
            {"evaluator_id": "fast", "kind": "latency_budget", "config": {"max_latency_ms": 0}},
            "positive config.max_latency_ms",
        ),
    ],
)
def test_misconfigured_evaluators_fail_closed(
    support_task, spec: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_response(_task(support_task, spec), _sample(), "output", latency_ms=1.0)


def test_json_schema_evaluator_requires_a_schema(support_task) -> None:
    task = _task(
        support_task,
        {"evaluator_id": "schema", "kind": "json_schema"},
        expected_output_schema=None,
    )
    with pytest.raises(ValueError, match="requires a schema"):
        evaluate_response(task, _sample(), "{}", latency_ms=1.0)
