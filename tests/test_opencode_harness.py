"""Isolated OpenCode transport, credential broker, and trace evidence tests."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr

from agent_benchmark_lab.harnesses.opencode import ZAICredentialSource, run_opencode_benchmark
from agent_benchmark_lab.harnesses.opencode.config import build_opencode_config
from agent_benchmark_lab.harnesses.opencode.events import (
    emit_opencode_spans,
    sanitize_and_parse_jsonl,
)
from agent_benchmark_lab.harnesses.opencode.models import OpenCodeBenchmarkRecord
from agent_benchmark_lab.harnesses.opencode.process import (
    OpenCodeProcessResult,
    isolated_child_environment,
    run_opencode,
)
from agent_benchmark_lab.harnesses.opencode.proxy import CodingPlanCredentialBroker
from agent_benchmark_lab.harnesses.opencode.workspace import isolated_workspace
from agent_benchmark_lab.providers.zai import ZAIReasoningEffort
from agenttrace.integrations import CaptureMode, TelemetrySettings, create_telemetry
from agenttrace.storage import FileObjectStore
from agenttrace.tasks import TaskDefinition, TaskSample, load_task


def test_credential_source_and_child_environment_never_mix_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "environment-secret")
    assert ZAICredentialSource().resolve().get_secret_value() == "environment-secret"

    secret_file = tmp_path / "zai-key"
    secret_file.write_text("file-secret\n")
    with pytest.raises(ValueError, match="only one"):
        ZAICredentialSource(api_key_file=secret_file).resolve()

    child = isolated_child_environment(tmp_path / "runtime")
    assert "ZAI_API_KEY" not in child
    assert "OPENAI_API_KEY" not in child
    assert child["HOME"].startswith(str(tmp_path))


def test_opencode_config_and_process_command_use_only_a_local_capability(tmp_path: Path) -> None:
    payload = build_opencode_config(
        prompt="private benchmark prompt",
        model="glm-5.2",
        small_model="glm-4.7-flash",
        broker_base_url="http://127.0.0.1:12345",
        broker_token="short-lived-capability",
        reasoning_effort=ZAIReasoningEffort.MAX,
    )
    serialized = json.dumps(payload)
    assert "private benchmark prompt" in serialized
    assert "real-provider-secret" not in serialized
    assert payload["share"] == "disabled"
    assert payload["enabled_providers"] == ["zai-coding-plan"]
    assert payload["provider"]["zai-coding-plan"]["npm"] == "@ai-sdk/openai-compatible"
    assert set(payload["provider"]["zai-coding-plan"]["models"]) == {
        "glm-5.2",
        "glm-4.7-flash",
    }
    assert payload["permission"]["webfetch"] == "deny"
    assert payload["provider"]["zai-coding-plan"]["models"]["glm-5.2"]["variants"]["max"] == {
        "reasoningEffort": "max"
    }

    config_path = tmp_path / "opencode.json"
    config_path.write_text(serialized)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = run_opencode(
        executable="true",
        workspace=workspace,
        config_path=config_path,
        runtime_root=tmp_path / "runtime",
        model="glm-5.2",
        reasoning_effort=ZAIReasoningEffort.MAX,
        sample_id="sample-safe",
        timeout_seconds=5,
    )
    assert result.exit_code == 0
    assert "private benchmark prompt" not in " ".join(result.argv)
    assert "short-lived-capability" not in " ".join(result.argv)
    assert result.argv[result.argv.index("--variant") + 1] == "max"


def test_loopback_broker_forwards_only_the_allowed_model_without_revealing_key() -> None:
    captured: dict[str, str] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"choices": []})

    client = httpx.Client(transport=httpx.MockTransport(upstream))
    with CodingPlanCredentialBroker(
        api_key=SecretStr("real-provider-secret"),
        model="glm-5.2",
        max_requests=1,
        http_client=client,
    ) as broker:
        response = httpx.post(
            f"{broker.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {broker.client_token}"},
            json={"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]},
            timeout=5,
        )
        denied = httpx.post(
            f"{broker.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {broker.client_token}"},
            json={"model": "other-model", "messages": []},
            timeout=5,
        )
        over_limit = httpx.post(
            f"{broker.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {broker.client_token}"},
            json={"model": "glm-5.2", "messages": []},
            timeout=5,
        )
        evidence = broker.requests
    client.close()

    assert response.status_code == 200
    assert denied.status_code == 400
    assert over_limit.status_code == 429
    assert captured["authorization"] == "Bearer real-provider-secret"
    assert "real-provider-secret" not in captured["body"]
    forwarded = json.loads(captured["body"])
    assert forwarded["thinking"] == {"type": "enabled"}
    assert forwarded["reasoning_effort"] == "max"
    assert len(evidence) == 1
    assert "real-provider-secret" not in repr(evidence)


def test_opencode_benchmark_collects_sanitized_trace_and_workspace_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_key = "real-coding-secret-123456"
    monkeypatch.setenv("ZAI_API_KEY", real_key)
    root = Path(__file__).resolve().parents[1]
    task = load_task(root / "examples/tasks/code-review.yaml")
    sample = TaskSample(
        sample_id="unsafe-yaml",
        input={
            "language": "python",
            "diff": "- yaml.safe_load(text)\n+ yaml.load(text, Loader=yaml.Loader)",
        },
        target="unsafe YAML deserialization",
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("value = 1\n")
    (source / ".env").write_text("DO_NOT_COPY=secret\n")
    (source / ".opencode").mkdir()
    (source / ".opencode" / "plugin.js").write_text("malicious")

    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "glm-5.2",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    broker_client = httpx.Client(transport=httpx.MockTransport(upstream))

    def fake_process(**kwargs) -> OpenCodeProcessResult:
        workspace = kwargs["workspace"]
        config = json.loads(kwargs["config_path"].read_text())
        serialized_config = json.dumps(config)
        assert real_key not in serialized_config
        assert not (workspace / ".env").exists()
        assert not (workspace / ".opencode").exists()
        assert kwargs["reasoning_effort"] == ZAIReasoningEffort.MAX
        provider = config["provider"]["zai-coding-plan"]["options"]
        response = httpx.post(
            f"{provider['baseURL']}/chat/completions",
            headers={"Authorization": f"Bearer {provider['apiKey']}"},
            json={"model": "glm-5.2", "messages": []},
            timeout=5,
        )
        assert response.status_code == 200
        (workspace / "main.py").write_text("value = 2\n")
        events = [
            {
                "type": "tool_use",
                "part": {
                    "tool": "edit",
                    "state": {
                        "input": {"path": "main.py"},
                        "output": f"changed without exposing {real_key}",
                        "status": "completed",
                    },
                },
            },
            {
                "type": "step_finish",
                "part": {
                    "cost": 0.012,
                    "tokens": {
                        "input": 12,
                        "output": 5,
                        "reasoning": 2,
                        "cache": {"read": 3},
                    },
                },
            },
            {"type": "text", "part": {"text": "unsafe YAML deserialization"}},
        ]
        return OpenCodeProcessResult(
            argv=("opencode", "run", "--command", "agenttrace-task"),
            exit_code=0,
            stdout="\n".join(json.dumps(event) for event in events),
            stderr=f"debug value {real_key}",
            elapsed_seconds=1.25,
            timed_out=False,
        )

    exporter = InMemorySpanExporter()
    telemetry = create_telemetry(
        TelemetrySettings(capture_content=CaptureMode.FULL),
        span_exporter=exporter,
    )
    objects = FileObjectStore(tmp_path / "objects")
    execution = run_opencode_benchmark(
        task=task,
        samples=[sample],
        credential_source=ZAICredentialSource(),
        telemetry=telemetry,
        objects=objects,
        experiment_id="opencode-test",
        session_id="langfuse-observer-test",
        workspace_source=source,
        process_runner=fake_process,
        broker_http_client=broker_client,
    )
    telemetry.close()
    broker_client.close()

    assert execution.manifest.summary.completed_count == 1
    assert execution.manifest.summary.model_call_count == 1
    assert execution.manifest.summary.tool_call_count == 1
    ref = execution.manifest.records[0]
    record = json.loads(objects.get_bytes(ref.key))
    assert record["usage"] == {
        "cached_input_tokens": 3,
        "input_tokens": 12,
        "output_tokens": 5,
        "reasoning_tokens": 2,
    }
    assert record["required_evaluators_passed"] is True
    assert record["model"] == "glm-5.2"
    assert record["thinking_mode"] == "enabled"
    assert record["reasoning_effort"] == "max"
    assert record["broker_requests"][0]["reasoning_effort"] == "max"
    legacy_record = dict(record)
    legacy_record.pop("thinking_mode")
    legacy_record.pop("reasoning_effort")
    legacy_record["broker_requests"] = [dict(record["broker_requests"][0])]
    legacy_record["broker_requests"][0].pop("thinking_mode")
    legacy_record["broker_requests"][0].pop("reasoning_effort")
    parsed_legacy = OpenCodeBenchmarkRecord.model_validate(legacy_record)
    assert parsed_legacy.reasoning_effort is None
    assert record["redaction_count"] >= 1
    assert record["workspace_changes"][0]["path"] == "main.py"
    raw_key = record["raw_events_ref"].removeprefix("object://")
    assert b"[REDACTED_SECRET]" in objects.get_bytes(raw_key)
    stored = b"\n".join(objects.get_bytes(key) for key in objects.list_keys())
    assert real_key.encode("utf-8") not in stored

    spans = list(exporter.get_finished_spans())
    assert sum(span.name == "code.security-review/unsafe-yaml" for span in spans) == 1
    assert any(span.name == "opencode.tool.edit" for span in spans)
    assert any(span.name == "opencode.model_generation" for span in spans)
    assert any(span.name == "credential_broker.forward" for span in spans)
    assert all(
        dict(span.attributes or {})["session.id"] == "langfuse-observer-test" for span in spans
    )
    input_span = next(span for span in spans if span.name == "task.input")
    assert "isolated autonomous benchmark session" in str(
        dict(input_span.attributes or {})["input.value"]
    )
    model_span = next(span for span in spans if span.name == "opencode.model_generation")
    model_attributes = dict(model_span.attributes or {})
    assert model_attributes["gen_ai.request.model"] == "glm-5.2"
    assert model_attributes["llm.cost.total"] == 0.012
    assert "unsafe YAML deserialization" in str(model_attributes["output.value"])
    assert real_key not in str([dict(span.attributes or {}) for span in spans])


def test_workspace_rejects_symlinks_that_escape_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (source / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        with isolated_workspace(source):
            pass


def _jsonl(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events)


def test_only_the_terminal_step_text_is_the_answer() -> None:
    _, parsed = sanitize_and_parse_jsonl(
        _jsonl(
            {"type": "step_start", "part": {}},
            {"type": "text", "part": {"text": "Let me read main.py first."}},
            {"type": "tool_use", "part": {"tool": "read", "state": {"status": "completed"}}},
            {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
            {"type": "step_start", "part": {}},
            {"type": "text", "part": {"text": "unsafe YAML deserialization"}},
            {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
        )
    )

    assert parsed.final_text == "unsafe YAML deserialization"
    assert parsed.answer_text == "unsafe YAML deserialization"


@pytest.mark.parametrize(
    "tail",
    [
        # The final step only ran tools.
        [
            {"type": "step_start", "part": {}},
            {"type": "tool_use", "part": {"tool": "read", "state": {"status": "completed"}}},
            {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
        ],
        # The agent was interrupted after opening its final step.
        [{"type": "step_start", "part": {}}],
    ],
)
def test_earlier_narration_is_never_promoted_to_the_answer(tail: list[dict[str, object]]) -> None:
    _, parsed = sanitize_and_parse_jsonl(
        _jsonl(
            {"type": "step_start", "part": {}},
            {"type": "text", "part": {"text": "unsafe YAML deserialization"}},
            {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
            *tail,
        )
    )

    assert parsed.answer_text == ""
    assert parsed.final_text == ""


def test_strict_redaction_applies_to_evidence_but_not_to_the_scored_answer() -> None:
    encoded, parsed = sanitize_and_parse_jsonl(
        _jsonl({"type": "text", "part": {"text": "Escalate to billing@example.com"}}),
        known_secrets=("real-provider-secret",),
        redaction_mode="strict",
    )

    assert "billing@example.com" not in parsed.final_text
    assert b"billing@example.com" not in encoded
    assert parsed.answer_text == "Escalate to billing@example.com"


def test_opencode_spans_follow_event_order_and_mark_failed_tools() -> None:
    _, parsed = sanitize_and_parse_jsonl(
        _jsonl(
            {"type": "step_start", "part": {}},
            {"type": "tool_use", "part": {"tool": "bash", "state": {"status": "error"}}},
            {"type": "step_finish", "part": {"tokens": {"input": 1, "output": 1}}},
            {"type": "step_start", "part": {}},
            {"type": "tool_use", "part": {"tool": "edit", "state": {"status": "completed"}}},
            {"type": "step_finish", "part": {"tokens": {"input": 2, "output": 2}}},
            # A step without an explicit start opens right after the previous step finished.
            {"type": "tool_use", "part": {"tool": "read", "state": {"status": "completed"}}},
            {"type": "step_finish", "part": {"tokens": {"input": 3, "output": 3}}},
        )
    )
    exporter = InMemorySpanExporter()
    telemetry = create_telemetry(TelemetrySettings(), span_exporter=exporter)
    with telemetry.run(experiment_id="order-test") as run:
        emit_opencode_spans(run, parsed, model_name="glm-5.2")
    telemetry.close()

    spans = [span for span in exporter.get_finished_spans() if span.name.startswith("opencode.")]
    assert [span.name for span in spans] == [
        "opencode.model_generation",
        "opencode.tool.bash",
        "opencode.model_generation",
        "opencode.tool.edit",
        "opencode.model_generation",
        "opencode.tool.read",
    ]
    tokens = [
        (span.attributes or {}).get("llm.token_count.prompt")
        for span in spans
        if "model" in span.name
    ]
    assert tokens == [1, 2, 3]
    assert spans[1].status.status_code == StatusCode.ERROR
    assert spans[3].status.status_code == StatusCode.OK


def test_broker_drain_waits_for_in_flight_forwards_and_records_stragglers() -> None:
    release = threading.Event()
    entered = threading.Event()

    def slow_upstream(_: httpx.Request) -> httpx.Response:
        entered.set()
        release.wait(timeout=10)
        return httpx.Response(200, json={"choices": []})

    client = httpx.Client(transport=httpx.MockTransport(slow_upstream))
    responses: list[int] = []
    with CodingPlanCredentialBroker(
        api_key=SecretStr("real-provider-secret"), model="glm-5.2", http_client=client
    ) as broker:

        def call() -> None:
            response = httpx.post(
                f"{broker.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {broker.client_token}"},
                json={"model": "glm-5.2", "messages": []},
                timeout=10,
            )
            responses.append(response.status_code)

        caller = threading.Thread(target=call)
        caller.start()
        assert entered.wait(timeout=5)
        abandoned = broker.drain(timeout_seconds=0.05)
        assert [(request.completed, request.status_code) for request in abandoned] == [(False, 0)]
        assert abandoned[0].request_bytes > 0

        release.set()
        caller.join(timeout=5)
        assert broker.requests == abandoned
    client.close()
    assert responses == [200]

    quick = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})))
    with CodingPlanCredentialBroker(
        api_key=SecretStr("real-provider-secret"), model="glm-5.2", http_client=quick
    ) as broker:
        httpx.post(
            f"{broker.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {broker.client_token}"},
            json={"model": "glm-5.2", "messages": []},
            timeout=5,
        )
        drained = broker.drain()
    quick.close()
    assert [(request.completed, request.status_code) for request in drained] == [(True, 200)]


def test_opencode_runner_rejects_environments_it_cannot_provision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "real-coding-secret-123456")
    root = Path(__file__).resolve().parents[1]
    task = load_task(root / "examples/tasks/code-review.yaml")
    payload = task.model_dump(mode="json")
    payload["environment"] = {"kind": "docker", "image": "sandbox:latest"}
    docker_task = TaskDefinition.model_validate(payload)

    def never_run(**_: object) -> OpenCodeProcessResult:
        raise AssertionError("OpenCode must not start for an unsupported environment")

    with pytest.raises(ValueError, match="environment.kind=docker"):
        run_opencode_benchmark(
            task=docker_task,
            samples=[TaskSample(sample_id="s1", input={"language": "python", "diff": ""})],
            credential_source=ZAICredentialSource(),
            telemetry=create_telemetry(TelemetrySettings(telemetry_enabled=False)),
            objects=FileObjectStore(tmp_path / "objects"),
            experiment_id="env-test",
            process_runner=never_run,
        )


def test_broker_accounts_for_a_forward_whose_agent_disconnected() -> None:
    import socket
    import threading
    import time

    forwarding = threading.Event()

    def slow_upstream(_: httpx.Request) -> httpx.Response:
        forwarding.set()
        time.sleep(0.3)
        return httpx.Response(200, json={"choices": []})

    client = httpx.Client(transport=httpx.MockTransport(slow_upstream))
    with CodingPlanCredentialBroker(
        api_key=SecretStr("real-provider-secret"), model="glm-5.2", http_client=client
    ) as broker:
        host, port = broker.base_url.removeprefix("http://").split(":")
        body = json.dumps({"model": "glm-5.2", "messages": []}).encode()
        with socket.create_connection((host, int(port))) as connection:
            connection.sendall(
                b"POST /chat/completions HTTP/1.1\r\nHost: x\r\n"
                + f"Authorization: Bearer {broker.client_token}\r\n".encode()
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            assert forwarding.wait(timeout=5)
        started = time.perf_counter()
        drained = broker.drain(timeout_seconds=5)
        assert time.perf_counter() - started < 3
    client.close()
    assert [(request.completed, request.status_code) for request in drained] == [(True, 200)]


def test_broker_rejects_a_request_whose_body_arrives_after_drain() -> None:
    import socket
    import time

    upstream_calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request)
        return httpx.Response(200, json={"choices": []})

    client = httpx.Client(transport=httpx.MockTransport(upstream))
    with CodingPlanCredentialBroker(
        api_key=SecretStr("real-provider-secret"), model="glm-5.2", http_client=client
    ) as broker:
        host, port = broker.base_url.removeprefix("http://").split(":")
        body = json.dumps({"model": "glm-5.2", "messages": []}).encode()
        connection = socket.create_connection((host, int(port)))
        connection.sendall(
            b"POST /chat/completions HTTP/1.1\r\nHost: x\r\n"
            + f"Authorization: Bearer {broker.client_token}\r\n".encode()
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body[:5]
        )
        # Usually lets the handler start reading the body; the assertions hold either way.
        time.sleep(0.05)
        drained = broker.drain(timeout_seconds=1)
        connection.sendall(body[5:])
    # The listener is closed now: an accepted handler answers 503, a still-queued one is reset.
    connection.settimeout(5)
    try:
        reply = connection.recv(4096)
    except ConnectionResetError:
        reply = b""
    finally:
        connection.close()
    client.close()
    assert drained == ()
    assert upstream_calls == []
    assert reply == b"" or reply.split(b" ", 2)[1] == b"503"
