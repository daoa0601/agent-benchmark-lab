"""Loopback credential broker for a supported OpenCode Coding Plan session."""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
from pydantic import SecretStr

from agent_benchmark_lab.providers.zai import (
    CODING_PLAN_API_BASE,
    ZAIReasoningEffort,
    ZAIThinkingMode,
)


@dataclass(frozen=True)
class BrokerRequest:
    sequence: int
    status_code: int
    latency_ms: float
    request_bytes: int
    response_bytes: int
    thinking_mode: str
    reasoning_effort: str
    completed: bool = True
    """False when the forward was still running when the session closed its evidence."""


class BrokerDrainedError(RuntimeError):
    """A request arrived after the broker stopped admitting upstream forwards."""


class CodingPlanCredentialBroker:
    """Inject the real key upstream without placing it in the agent process.

    This is intentionally a narrow forwarder: one model, one endpoint, bounded calls and payloads,
    loopback only. It is not a general-purpose HTTP proxy.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        reasoning_effort: ZAIReasoningEffort = ZAIReasoningEffort.MAX,
        max_requests: int = 256,
        max_request_bytes: int = 4 * 1024 * 1024,
        timeout_seconds: float = 600.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        self._api_key = api_key.get_secret_value()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_requests = max_requests
        self.max_request_bytes = max_request_bytes
        self.timeout_seconds = timeout_seconds
        self.client_token = secrets.token_urlsafe(32)
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = http_client is None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._requests: list[BrokerRequest] = []
        self._claimed_requests = 0
        # sequence -> (perf_counter start, request bytes) for forwards not yet recorded.
        self._in_flight: dict[int, tuple[float, int]] = {}
        self._abandoned: set[int] = set()
        self._admitting = True

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("credential broker is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def requests(self) -> tuple[BrokerRequest, ...]:
        with self._lock:
            return tuple(sorted(self._requests, key=lambda request: request.sequence))

    def _claim_request(self, *, started: float, request_bytes: int) -> int | None:
        """Atomically reserve one upstream request slot and its stable sequence number.

        Raises ``BrokerDrainedError`` once draining has begun, so a handler that was still
        reading its body cannot start billable work after the evidence was returned.
        """
        with self._lock:
            if not self._admitting:
                raise BrokerDrainedError
            if self._claimed_requests >= self.max_requests:
                return None
            sequence = self._claimed_requests
            self._claimed_requests += 1
            self._in_flight[sequence] = (started, request_bytes)
            return sequence

    def _append_request(
        self,
        *,
        sequence: int,
        status_code: int,
        latency_ms: float,
        request_bytes: int,
        response_bytes: int,
        thinking_mode: str,
        reasoning_effort: str,
    ) -> None:
        with self._idle:
            self._in_flight.pop(sequence, None)
            self._idle.notify_all()
            if sequence in self._abandoned:
                return
            self._requests.append(
                BrokerRequest(
                    sequence=sequence,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                    thinking_mode=thinking_mode,
                    reasoning_effort=reasoning_effort,
                )
            )

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        broker = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AgentTraceCredentialBroker/1.0"

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

            def _json_error(self, status: int, message: str) -> None:
                body = json.dumps({"error": {"message": message}}).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                started = time.perf_counter()
                authorization = self.headers.get("Authorization", "")
                if authorization != f"Bearer {broker.client_token}":
                    self._json_error(401, "invalid local broker capability")
                    return
                if not self.path.split("?", 1)[0].endswith("/chat/completions"):
                    self._json_error(404, "unsupported broker endpoint")
                    return
                raw_length = self.headers.get("Content-Length", "0")
                try:
                    content_length = int(raw_length)
                except ValueError:
                    self._json_error(400, "invalid content length")
                    return
                if content_length < 1 or content_length > broker.max_request_bytes:
                    self._json_error(413, "request payload is outside the allowed size")
                    return
                body = self.rfile.read(content_length)
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    self._json_error(400, "request body is not JSON")
                    return
                if not isinstance(payload, dict) or payload.get("model") != broker.model:
                    self._json_error(400, "request model is not allowed for this session")
                    return
                payload["thinking"] = {"type": ZAIThinkingMode.ENABLED.value}
                payload["reasoning_effort"] = broker.reasoning_effort.value
                upstream_body = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                try:
                    sequence = broker._claim_request(started=started, request_bytes=len(body))
                except BrokerDrainedError:
                    self._reply(lambda: self._json_error(503, "broker session has ended"))
                    return
                if sequence is None:
                    self._json_error(429, "session model-call limit reached")
                    return
                try:
                    upstream = broker._client.post(
                        f"{CODING_PLAN_API_BASE}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {broker._api_key}",
                            "Content-Type": "application/json",
                            "Accept": self.headers.get("Accept", "application/json"),
                            "User-Agent": self.headers.get("User-Agent", "opencode"),
                        },
                        content=upstream_body,
                        timeout=broker.timeout_seconds,
                    )
                    response_body = upstream.content
                    status_code = upstream.status_code
                    content_type = upstream.headers.get("Content-Type", "application/json")
                except httpx.HTTPError:
                    # Record before replying: the agent may already be gone.
                    broker._append_request(
                        sequence=sequence,
                        status_code=502,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        request_bytes=len(body),
                        response_bytes=0,
                        thinking_mode=ZAIThinkingMode.ENABLED.value,
                        reasoning_effort=broker.reasoning_effort.value,
                    )
                    self._reply(lambda: self._json_error(502, "upstream model request failed"))
                    return
                # The upstream call is finished billable work whether or not the agent is still
                # connected to receive it, so account for it before delivering the response.
                broker._append_request(
                    sequence=sequence,
                    status_code=status_code,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    request_bytes=len(body),
                    response_bytes=len(response_body),
                    thinking_mode=ZAIThinkingMode.ENABLED.value,
                    reasoning_effort=broker.reasoning_effort.value,
                )

                def deliver() -> None:
                    self.send_response(status_code)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(response_body)))
                    self.end_headers()
                    self.wfile.write(response_body)

                self._reply(deliver)

            def _reply(self, send: Callable[[], None]) -> None:
                """Deliver a response, tolerating an agent that already disconnected."""
                try:
                    send()
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

        return Handler

    def drain(self, timeout_seconds: float | None = None) -> tuple[BrokerRequest, ...]:
        """Stop accepting requests, wait for in-flight forwards, and return the final evidence.

        Call this after the agent process has exited and before reading evidence: a forward that
        started just before a watchdog kill is still billable work. Forwards still running after
        the wait (by default the upstream timeout plus a margin) are recorded as incomplete.
        """
        with self._lock:
            self._admitting = False
        if self._server is not None:
            self._server.shutdown()
        wait = self.timeout_seconds + 5 if timeout_seconds is None else timeout_seconds
        with self._idle:
            self._idle.wait_for(lambda: not self._in_flight, timeout=wait)
            now = time.perf_counter()
            for sequence, (started, request_bytes) in sorted(self._in_flight.items()):
                self._abandoned.add(sequence)
                self._requests.append(
                    BrokerRequest(
                        sequence=sequence,
                        status_code=0,
                        latency_ms=(now - started) * 1000,
                        request_bytes=request_bytes,
                        response_bytes=0,
                        thinking_mode=ZAIThinkingMode.ENABLED.value,
                        reasoning_effort=self.reasoning_effort.value,
                        completed=False,
                    )
                )
            self._in_flight.clear()
        return self.requests

    def __enter__(self) -> CodingPlanCredentialBroker:
        if self._server is not None:
            raise RuntimeError("credential broker cannot be entered twice")
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._owns_client:
            self._client.close()
        self._api_key = ""
        self.client_token = ""
        self._server = None
        self._thread = None
