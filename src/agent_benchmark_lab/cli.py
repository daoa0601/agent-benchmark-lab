"""Command-line interface for the benchmark workbench."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from pydantic import BaseModel, ValidationError

from agent_benchmark_lab.evidence import describe_evidence
from agent_benchmark_lab.gate import gate_session
from agenttrace.storage import FileObjectStore
from agenttrace.tasks import TaskDefinition, TaskSample, describe_validation_error, load_task

app = typer.Typer(
    name="agent-benchmark",
    help="Run, describe, and gate auditable agent benchmarks.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
provider_app = typer.Typer(help="Run securely configured model-provider harnesses.")
benchmark_app = typer.Typer(help="Run provider-neutral tasks with traced model adapters.")
app.add_typer(provider_app, name="provider")
app.add_typer(benchmark_app, name="benchmark")


def _invalid(error: Exception) -> typer.Exit:
    """Report a failure by type only, so artifact contents and paths are never echoed."""

    typer.echo(json.dumps({"status": "invalid", "error_type": type(error).__name__}), err=True)
    return typer.Exit(code=1)


EnumT = TypeVar("EnumT", bound=Enum)

# Models whose Pydantic errors loaders wrap, keyed by the error title they produce.
_VALIDATED_MODELS: dict[str, type[BaseModel]] = {
    model.__name__: model for model in (TaskDefinition, TaskSample)
}


def _bad_parameter(error: ValueError, model: type[BaseModel] | None = None) -> typer.BadParameter:
    """Report invalid input by field and rule; exception text can quote the offending values.

    ``model`` names the fields that may appear in the location; other location parts are masked.
    A loader error wrapping a Pydantic error is described the same way, and any other error by
    type only.
    """
    cause = error if isinstance(error, ValidationError) else error.__cause__
    if isinstance(cause, ValidationError):
        cause_model = _VALIDATED_MODELS.get(cause.title, model)
        return typer.BadParameter(f"{cause.title}: {describe_validation_error(cause, cause_model)}")
    return typer.BadParameter(f"invalid input ({type(error).__name__})")


def _choice(kind: type[EnumT], value: str, option: str) -> EnumT:
    """Parse an option value, naming the allowed values rather than echoing the given one."""
    try:
        return kind(value)
    except ValueError as error:
        allowed = ", ".join(str(member.value) for member in kind)
        raise typer.BadParameter(f"{option} must be one of: {allowed}") from error


@app.command()
def evidence(
    path: Path,
    uri: Annotated[str | None, typer.Option()] = None,
    media_type: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Describe one artifact by digest."""
    try:
        output = describe_evidence(path, uri=uri, media_type=media_type)
    except Exception as error:
        raise _invalid(error) from error
    typer.echo(json.dumps(output, indent=2, sort_keys=True))


@app.command()
def gate(
    manifest: Path,
    report: Annotated[Path, typer.Option()],
    object_root: Annotated[
        Path | None,
        typer.Option(help="Persist the immutable audit bundle into this AgentTrace object store."),
    ] = None,
) -> None:
    """Gate one auditable session manifest; exit 2 when a headline is blocked."""
    try:
        summary = gate_session(manifest, report_path=report, object_root=object_root)
    except Exception as error:
        raise _invalid(error) from error
    typer.echo(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    if summary.status == "blocked":
        raise typer.Exit(code=2)


@benchmark_app.command("zai-api")
def zai_api_benchmark(
    task_path: Annotated[Path, typer.Argument(help="Versioned AgentTrace task YAML.")],
    samples_path: Annotated[Path, typer.Argument(help="Strict TaskSample JSONL file.")],
    model: Annotated[str, typer.Option()] = "glm-5.2",
    api_key_file: Annotated[
        Path | None,
        typer.Option(
            "--api-key-file",
            help="Read the general API credential from a file instead of ZAI_API_KEY.",
        ),
    ] = None,
    max_tokens: Annotated[int, typer.Option(min=1)] = 512,
    temperature: Annotated[float, typer.Option(min=0)] = 0.0,
    thinking_mode: Annotated[
        str,
        typer.Option(help="Turn-level reasoning mode: disabled or enabled."),
    ] = "disabled",
    reasoning_effort: Annotated[
        str,
        typer.Option(
            help="GLM-5.2 reasoning effort: max, xhigh, high, medium, low, minimal, or none."
        ),
    ] = "max",
    experiment_id: Annotated[str, typer.Option()] = "zai-api-benchmark",
    output_root: Annotated[
        Path,
        typer.Option(help="Local immutable benchmark object-store root."),
    ] = Path("outputs/benchmarks"),
    store_outputs: Annotated[
        bool,
        typer.Option(
            "--store-outputs/--metadata-only",
            help="Retain full responses as content-addressed objects; default stores hashes only.",
        ),
    ] = False,
    otlp_endpoint: Annotated[
        str | None,
        typer.Option(help="OTLP base URL or /v1/traces URL. Omit to disable trace export."),
    ] = None,
    capture_content: Annotated[
        str,
        typer.Option(help="OTLP payload policy: none, metadata, or full."),
    ] = "metadata",
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast/--continue-on-provider-error"),
    ] = False,
) -> None:
    """Run general-API GLM eval samples without printing prompts, responses, or credentials."""
    from agent_benchmark_lab.benchmarks import run_zai_api_benchmark
    from agent_benchmark_lab.providers.zai import (
        ZAIEndpointPlan,
        ZAIReasoningEffort,
        ZAISettings,
        ZAIThinkingMode,
    )
    from agenttrace.integrations import CaptureMode, TelemetrySettings, create_telemetry
    from agenttrace.tasks import load_task_samples

    try:
        capture = _choice(CaptureMode, capture_content, "--capture-content")
        thinking = _choice(ZAIThinkingMode, thinking_mode, "--thinking-mode")
        effort = _choice(ZAIReasoningEffort, reasoning_effort, "--reasoning-effort")
        task = load_task(task_path)
        samples = load_task_samples(samples_path)
    except ValueError as error:
        raise _bad_parameter(error, TaskDefinition) from error
    try:
        if api_key_file is None:
            # Leave the key source to the environment (ZAI_API_KEY or ZAI_API_KEY_FILE).
            provider_settings = ZAISettings(
                endpoint_plan=ZAIEndpointPlan.GENERAL,
                model=model,
                thinking_mode=thinking,
                reasoning_effort=effort,
            )
        else:
            # An explicit --api-key-file wins over ZAI_API_KEY.
            provider_settings = ZAISettings(
                api_key=None,
                api_key_file=api_key_file,
                endpoint_plan=ZAIEndpointPlan.GENERAL,
                model=model,
                thinking_mode=thinking,
                reasoning_effort=effort,
            )
    except ValueError as error:
        raise _bad_parameter(error, ZAISettings) from error

    telemetry = create_telemetry(
        TelemetrySettings(
            telemetry_enabled=otlp_endpoint is not None,
            otlp_endpoint=otlp_endpoint or "http://127.0.0.1:4318/v1/traces",
            service_name="agenttrace-zai-api-benchmark",
            capture_content=capture,
            redaction_mode=task.data_policy.redaction_mode,
        )
    )
    try:
        execution = run_zai_api_benchmark(
            task=task,
            samples=samples,
            settings=provider_settings,
            telemetry=telemetry,
            objects=FileObjectStore(output_root),
            experiment_id=experiment_id,
            max_tokens=max_tokens,
            temperature=temperature,
            store_outputs=store_outputs,
            fail_fast=fail_fast,
        )
    except ValueError as error:
        typer.echo(
            json.dumps(
                {
                    "ok": False,
                    "provider": "zai",
                    "endpoint_plan": "general",
                    "error_type": type(error).__name__,
                    "content_printed": False,
                },
                indent=2,
                sort_keys=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from error
    finally:
        telemetry.close()

    summary = execution.manifest.summary
    typer.echo(
        json.dumps(
            {
                "ok": summary.provider_error_count == 0,
                "provider": "zai",
                "endpoint_plan": "general",
                "model": execution.manifest.model,
                "thinking_mode": execution.manifest.thinking_mode,
                "reasoning_effort": execution.manifest.reasoning_effort,
                "experiment_id": execution.manifest.experiment_id,
                "manifest_key": execution.manifest_key,
                "object_root": str(output_root.expanduser().resolve()),
                "summary": summary.model_dump(mode="json"),
                "content_printed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if summary.provider_error_count:
        raise typer.Exit(code=2)


@benchmark_app.command("zai-opencode")
def zai_opencode_benchmark(
    task_path: Annotated[Path, typer.Argument(help="Versioned AgentTrace task YAML.")],
    samples_path: Annotated[Path, typer.Argument(help="Strict TaskSample JSONL file.")],
    workspace: Annotated[
        Path | None,
        typer.Option(help="Optional source tree copied into a fresh workspace for every sample."),
    ] = None,
    model: Annotated[str, typer.Option()] = "glm-5.2",
    reasoning_effort: Annotated[
        str,
        typer.Option(
            help="GLM-5.2 reasoning effort: max, xhigh, high, medium, low, minimal, or none."
        ),
    ] = "max",
    small_model: Annotated[
        str | None,
        typer.Option(
            help="Optional lightweight model; defaults to --model for a one-model session."
        ),
    ] = None,
    api_key_file: Annotated[
        Path | None,
        typer.Option(
            "--api-key-file",
            help="Parent-only Coding Plan credential file; otherwise use ZAI_API_KEY.",
        ),
    ] = None,
    executable: Annotated[str, typer.Option(help="OpenCode executable name or path.")] = "opencode",
    watchdog_seconds: Annotated[
        int | None,
        typer.Option(min=0, help="0 disables the watchdog; omitted uses the task policy."),
    ] = None,
    max_model_calls: Annotated[int, typer.Option(min=1)] = 256,
    experiment_id: Annotated[str, typer.Option()] = "zai-opencode-benchmark",
    session_id: Annotated[
        str | None,
        typer.Option(
            help="Portable OTLP session ID copied to every span; Langfuse groups matching traces."
        ),
    ] = None,
    output_root: Annotated[
        Path,
        typer.Option(help="Local immutable benchmark object-store root."),
    ] = Path("outputs/benchmarks"),
    store_outputs: Annotated[
        bool,
        typer.Option("--store-outputs/--metadata-only"),
    ] = False,
    otlp_endpoint: Annotated[
        str | None,
        typer.Option(help="OTLP base URL or /v1/traces URL. Omit to disable trace export."),
    ] = None,
    capture_content: Annotated[
        str,
        typer.Option(help="OTLP payload policy: none, metadata, or full."),
    ] = "metadata",
) -> None:
    """Run isolated OpenCode sessions without exposing the Coding Plan key to agent tools."""
    from agent_benchmark_lab.harnesses.opencode import ZAICredentialSource, run_opencode_benchmark
    from agent_benchmark_lab.providers.zai import ZAIReasoningEffort
    from agenttrace.integrations import CaptureMode, TelemetrySettings, create_telemetry
    from agenttrace.tasks import load_task_samples

    try:
        capture = _choice(CaptureMode, capture_content, "--capture-content")
        effort = _choice(ZAIReasoningEffort, reasoning_effort, "--reasoning-effort")
        task = load_task(task_path)
        samples = load_task_samples(samples_path)
        if watchdog_seconds is None:
            timeout = (
                task.session_policy.watchdog_seconds
                if task.session_policy is not None
                else task.environment.resources.timeout_seconds
            )
        else:
            timeout = watchdog_seconds or None
    except ValueError as error:
        raise _bad_parameter(error, TaskDefinition) from error
    telemetry = create_telemetry(
        TelemetrySettings(
            telemetry_enabled=otlp_endpoint is not None,
            otlp_endpoint=otlp_endpoint or "http://127.0.0.1:4318/v1/traces",
            service_name="agenttrace-zai-opencode-benchmark",
            capture_content=capture,
            redaction_mode=task.data_policy.redaction_mode,
        )
    )
    try:
        execution = run_opencode_benchmark(
            task=task,
            samples=samples,
            credential_source=ZAICredentialSource(api_key_file=api_key_file),
            telemetry=telemetry,
            objects=FileObjectStore(output_root),
            experiment_id=experiment_id,
            session_id=session_id,
            workspace_source=workspace,
            model=model,
            reasoning_effort=effort,
            small_model=small_model,
            executable=executable,
            timeout_seconds=timeout,
            max_model_calls=max_model_calls,
            store_outputs=store_outputs,
        )
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(
            json.dumps(
                {
                    "ok": False,
                    "provider": "zai",
                    "transport": "opencode",
                    "endpoint_plan": "coding",
                    "error_type": type(error).__name__,
                    "content_printed": False,
                },
                indent=2,
                sort_keys=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from error
    finally:
        telemetry.close()
    summary = execution.manifest.summary
    typer.echo(
        json.dumps(
            {
                "ok": summary.process_error_count == 0 and summary.timed_out_count == 0,
                "provider": "zai",
                "transport": "opencode",
                "endpoint_plan": "coding",
                "model": execution.manifest.model,
                "thinking_mode": execution.manifest.thinking_mode,
                "reasoning_effort": execution.manifest.reasoning_effort,
                "experiment_id": execution.manifest.experiment_id,
                "session_id": session_id,
                "manifest_key": execution.manifest_key,
                "object_root": str(output_root.expanduser().resolve()),
                "summary": summary.model_dump(mode="json"),
                "content_printed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if summary.process_error_count or summary.timed_out_count:
        raise typer.Exit(code=2)


@provider_app.command("zai-smoke")
def zai_smoke(
    endpoint_plan: Annotated[
        str,
        typer.Option(
            "--endpoint-plan", help="Explicit entitlement/billing endpoint: general or coding."
        ),
    ],
    model: Annotated[str, typer.Option()] = "glm-5.2",
    max_tokens: Annotated[int, typer.Option(min=1, max=256)] = 32,
    otlp_endpoint: Annotated[
        str | None,
        typer.Option(help="OTLP base URL or full /v1/traces URL. Omit to disable export."),
    ] = None,
    experiment_id: Annotated[str, typer.Option()] = "zai-live-smoke",
    capture_content: Annotated[
        str,
        typer.Option(help="Trace payload policy: none, metadata, or full."),
    ] = "metadata",
    thinking_mode: Annotated[
        str,
        typer.Option(help="Turn-level reasoning mode: disabled or enabled."),
    ] = "disabled",
    reasoning_effort: Annotated[
        str,
        typer.Option(
            help="GLM-5.2 reasoning effort: max, xhigh, high, medium, low, minimal, or none."
        ),
    ] = "max",
) -> None:
    """Send a minimal general-API request without printing prompt, response, or credentials."""
    from agent_benchmark_lab.providers.zai import (
        ZAIClient,
        ZAIEndpointPlan,
        ZAIProviderError,
        ZAIReasoningEffort,
        ZAISettings,
        ZAIThinkingMode,
    )
    from agenttrace.integrations import CaptureMode, TelemetrySettings, create_telemetry

    try:
        plan = _choice(ZAIEndpointPlan, endpoint_plan, "--endpoint-plan")
        capture = _choice(CaptureMode, capture_content, "--capture-content")
        thinking = _choice(ZAIThinkingMode, thinking_mode, "--thinking-mode")
        effort = _choice(ZAIReasoningEffort, reasoning_effort, "--reasoning-effort")
    except ValueError as error:
        raise _bad_parameter(error) from error
    if plan == ZAIEndpointPlan.CODING:
        raise typer.BadParameter(
            "custom SDK calls cannot use the Coding Plan; use the supported OpenCode harness"
        )
    telemetry = create_telemetry(
        TelemetrySettings(
            telemetry_enabled=otlp_endpoint is not None,
            otlp_endpoint=otlp_endpoint or "http://127.0.0.1:4318/v1/traces",
            service_name="agenttrace-zai-smoke",
            capture_content=capture,
        )
    )
    try:
        with telemetry.run(
            experiment_id=experiment_id,
            task_id="provider.zai-connectivity",
            task_version="1.0.0",
            task_split="smoke",
            attributes={
                "agenttrace.provider.endpoint_plan": plan.value,
                "agenttrace.provider.thinking_mode": thinking.value,
                "agenttrace.provider.reasoning_effort": effort.value,
            },
        ) as run:
            with ZAIClient(
                ZAISettings(
                    endpoint_plan=plan,
                    model=model,
                    thinking_mode=thinking,
                    reasoning_effort=effort,
                ),
                trace_run=run,
            ) as client:
                response = client.complete(
                    [{"role": "user", "content": "Reply with exactly READY."}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
        completion_valid = response.finish_reason == "stop" and response.content.strip() == "READY"
        typer.echo(
            json.dumps(
                {
                    "ok": completion_valid,
                    "provider": "zai",
                    "endpoint_plan": plan.value,
                    "thinking_mode": thinking.value,
                    "reasoning_effort": effort.value,
                    "model": response.model,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                    "usage": response.usage.model_dump(),
                    "response_sha256": hashlib.sha256(response.content.encode("utf-8")).hexdigest(),
                    "trace_id": run.trace_id,
                    "content_printed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not completion_valid:
            raise typer.Exit(code=2)
    except (ZAIProviderError, ValueError) as error:
        payload = {
            "ok": False,
            "provider": "zai",
            "endpoint_plan": plan.value,
            "error_type": type(error).__name__,
            "status_code": getattr(error, "status_code", None),
            "request_id": getattr(error, "request_id", None),
            "content_printed": False,
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True), err=True)
        raise typer.Exit(code=1) from error
    finally:
        telemetry.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run a workbench command and return its exit code."""

    try:
        app(args=list(argv) if argv is not None else None, prog_name="agent-benchmark")
    except SystemExit as exit_:
        code = exit_.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    return 0
