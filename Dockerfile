# Build with the sibling checkouts as named contexts (compose.zai.yaml does this):
#   docker build \
#     --build-context agent-trace-platform=../agent-trace-platform \
#     --build-context benchmark-audit-core=../benchmark-audit-core .
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/workspace/agent-benchmark-lab/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv==0.9.5

# uv.lock resolves the siblings as ../<name>, so mirror that layout.
COPY --from=benchmark-audit-core pyproject.toml README.md LICENSE /workspace/benchmark-audit-core/
COPY --from=benchmark-audit-core src /workspace/benchmark-audit-core/src
COPY --from=agent-trace-platform pyproject.toml README.md LICENSE /workspace/agent-trace-platform/
COPY --from=agent-trace-platform src /workspace/agent-trace-platform/src

WORKDIR /workspace/agent-benchmark-lab
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 benchmark && chown -R benchmark:benchmark /workspace
USER benchmark

ENTRYPOINT ["agent-benchmark"]
