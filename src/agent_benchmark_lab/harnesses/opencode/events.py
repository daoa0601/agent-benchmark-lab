"""Version-tolerant OpenCode JSONL sanitization, parsing, and trace translation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agenttrace.integrations import AgentRun
from agenttrace.redaction import RedactionStats, Redactor


class OpenCodeToolEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    name: str
    input: JsonValue = None
    output: JsonValue = None
    status: str | None = None


class OpenCodeModelEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    # Position where the step opened; the model call precedes the tools it requested.
    started_sequence: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    output: str | None = None


@dataclass(frozen=True)
class ParsedOpenCodeEvents:
    event_count: int
    invalid_line_count: int
    tools: tuple[OpenCodeToolEvent, ...]
    models: tuple[OpenCodeModelEvent, ...]
    final_text: str
    """Redacted terminal answer; safe to persist and export."""
    answer_text: str
    """Unredacted terminal answer, used only for scoring and never persisted."""
    redaction_stats: RedactionStats


def _integer(value: object) -> int:
    return (
        int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
    )


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def sanitize_and_parse_jsonl(
    raw: str,
    *,
    known_secrets: tuple[str, ...] = (),
    redaction_mode: str = "strict",
) -> tuple[bytes, ParsedOpenCodeEvents]:
    """Redact raw events before persistence and omit private reasoning event payloads."""
    redactor = Redactor(cast(Any, redaction_mode))
    sanitized_lines: list[str] = []
    payloads: list[dict[str, JsonValue]] = []
    # Same events with known secrets removed but no pattern redaction, so scoring sees the
    # answer the agent actually gave. Only the terminal answer text is read from these.
    unredacted: list[dict[str, JsonValue]] = []
    invalid = 0
    total_stats = RedactionStats()
    for sequence, raw_line in enumerate(raw.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        unredacted_payload: dict[str, JsonValue] | None = None
        known_replacements = 0
        for secret in known_secrets:
            if secret:
                count = line.count(secret)
                line = line.replace(secret, "[REDACTED_SECRET]")
                known_replacements += count
        if known_replacements:
            total_stats = total_stats.add(RedactionStats(secret_values=known_replacements))
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            payload = {
                "type": "agenttrace.invalid_event",
                "sequence": sequence,
                "sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                "bytes": len(line.encode("utf-8")),
            }
        else:
            if not isinstance(decoded, dict):
                invalid += 1
                payload = {
                    "type": "agenttrace.invalid_event",
                    "sequence": sequence,
                    "sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                    "bytes": len(line.encode("utf-8")),
                }
            elif str(decoded.get("type", "")).casefold() in {
                "reasoning",
                "thinking",
                "reasoning_delta",
            } or (
                isinstance(decoded.get("part"), dict)
                and str(decoded["part"].get("type", "")).casefold()
                in {"reasoning", "thinking", "reasoning_delta"}
            ):
                payload = {
                    "type": "agenttrace.private_reasoning_removed",
                    "sequence": sequence,
                }
                total_stats = total_stats.add(RedactionStats(private_reasoning=1))
            else:
                cleaned, stats = redactor.redact(cast(JsonValue, decoded))
                total_stats = total_stats.add(stats)
                payload = cast(dict[str, JsonValue], cleaned)
                unredacted_payload = cast(dict[str, JsonValue], decoded)
        payloads.append(payload)
        unredacted.append(payload if unredacted_payload is None else unredacted_payload)
        sanitized_lines.append(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )

    tools: list[OpenCodeToolEvent] = []
    models: list[OpenCodeModelEvent] = []
    # Text is grouped per step and the answer is the final step's text, even when that step has
    # none (tools only, or interrupted): earlier narration ("let me look at the file") never is.
    steps: list[list[int]] = []
    current_step: list[int] = []
    step_opened = 0
    step_is_open = False
    for sequence, payload in enumerate(payloads):
        event_type = str(payload.get("type") or "")
        raw_part = payload.get("part")
        part = raw_part if isinstance(raw_part, dict) else {}
        if event_type == "tool_use":
            raw_state = part.get("state")
            state = raw_state if isinstance(raw_state, dict) else {}
            tools.append(
                OpenCodeToolEvent(
                    sequence=sequence,
                    name=str(part.get("tool") or "tool"),
                    input=cast(JsonValue, state.get("input")),
                    output=cast(JsonValue, state.get("output")),
                    status=str(state.get("status")) if state.get("status") is not None else None,
                )
            )
        normalized = str(event_type or part.get("type") or "").replace("-", "_")
        if normalized == "step_start":
            # Text outside any step, or a step that never finished, still counts as a step.
            if current_step or step_is_open:
                steps.append(current_step)
            current_step = []
            step_is_open = True
            step_opened = sequence
        if event_type == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                current_step.append(sequence)
        if normalized == "step_finish":
            steps.append(current_step)
            current_step = []
            step_is_open = False
            raw_tokens = part.get("tokens")
            tokens = raw_tokens if isinstance(raw_tokens, dict) else {}
            raw_cache = tokens.get("cache")
            cache = raw_cache if isinstance(raw_cache, dict) else {}
            models.append(
                OpenCodeModelEvent(
                    sequence=sequence,
                    started_sequence=step_opened,
                    input_tokens=_integer(tokens.get("input")),
                    output_tokens=_integer(tokens.get("output")),
                    cached_input_tokens=_integer(cache.get("read")),
                    reasoning_tokens=_integer(tokens.get("reasoning")),
                    cost_usd=_number(part.get("cost")),
                )
            )
            step_opened = sequence + 1
    if current_step or step_is_open:
        steps.append(current_step)
    answer_sequences = steps[-1] if steps else []
    encoded = ("\n".join(sanitized_lines) + ("\n" if sanitized_lines else "")).encode("utf-8")
    final_text = _join_text(payloads, answer_sequences)
    answer_text = _join_text(unredacted, answer_sequences)
    if models and final_text:
        models[-1] = models[-1].model_copy(update={"output": final_text})
    return encoded, ParsedOpenCodeEvents(
        event_count=len(payloads),
        invalid_line_count=invalid,
        tools=tuple(tools),
        models=tuple(models),
        final_text=final_text,
        answer_text=answer_text,
        redaction_stats=total_stats,
    )


def _join_text(payloads: list[dict[str, JsonValue]], sequences: list[int]) -> str:
    parts: list[str] = []
    for sequence in sequences:
        raw_part = payloads[sequence].get("part")
        text = raw_part.get("text") if isinstance(raw_part, dict) else None
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


def emit_opencode_spans(run: AgentRun, events: ParsedOpenCodeEvents, *, model_name: str) -> None:
    """Emit model and tool spans in the order OpenCode produced them."""
    ordered: list[OpenCodeModelEvent | OpenCodeToolEvent] = sorted(
        [*events.models, *events.tools],
        key=lambda event: (
            (event.started_sequence, 0)
            if isinstance(event, OpenCodeModelEvent)
            else (event.sequence, 1)
        ),
    )
    for event in ordered:
        if isinstance(event, OpenCodeModelEvent):
            _emit_model_span(run, event, model_name=model_name)
        else:
            _emit_tool_span(run, event)


def _emit_model_span(run: AgentRun, model: OpenCodeModelEvent, *, model_name: str) -> None:
    with run.span(
        "opencode.model_generation",
        step_type="model_generation",
        kind="LLM",
        input={"event_sequence": model.sequence},
        attributes={
            "gen_ai.provider.name": "zai",
            "gen_ai.request.model": model_name,
            "agenttrace.transport": "opencode",
        },
    ) as result:
        result.output = model.output or {"event_sequence": model.sequence}
        result.input_tokens = model.input_tokens
        result.output_tokens = model.output_tokens
        result.cached_input_tokens = model.cached_input_tokens
        result.reasoning_tokens = model.reasoning_tokens
        result.cost_usd = model.cost_usd


def _emit_tool_span(run: AgentRun, tool: OpenCodeToolEvent) -> None:
    with run.span(
        f"opencode.tool.{tool.name}",
        step_type="tool_call",
        kind="TOOL",
        input=tool.input,
        attributes={
            "tool.name": tool.name,
            "agenttrace.opencode.event_sequence": tool.sequence,
        },
    ) as result:
        result.output = tool.output
        result.attributes["tool.status"] = tool.status or "unknown"
        if tool.status is not None and tool.status.casefold() == "error":
            result.error_type = "tool_error"
