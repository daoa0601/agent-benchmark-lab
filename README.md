# Agent Benchmark Lab

A workbench for running and gating auditable, reproducible agent benchmarks. It runs tasks through
model-provider transports, records every sample as a trace and immutable evidence, and applies the
deterministic headline gate to an audited session manifest.

This checkout needs the sibling `agent-trace-platform` and `benchmark-audit-core` checkouts next to
it; see [the catalog README](../../README.md#cross-repo-boundaries) for the layout and cross-repo
boundaries.

## Boundaries

- This repository owns the benchmark runners (Z.AI general API and the isolated OpenCode Coding Plan
  harness), safe local trace capture, outcome semantics, evidence-description commands, and the
  single `agent-benchmark gate` CLI.
- Task contracts, telemetry, redaction rules, and immutable object storage come from AgentTrace's
  public Python API (`agenttrace.tasks`, `agenttrace.integrations`, `agenttrace.redaction`,
  `agenttrace.storage`, `agenttrace.audits`).
- Audit schemas and the gate decision come from `benchmark_audit_core`.
- A research repository owns its own task data, provider policy, prompts, and scientific claims.

## Checkout and verify

```bash
uv sync --extra dev
make check
```

## Run benchmarks

Run the bundled support-routing example through the Z.AI general API:

```bash
export ZAI_API_KEY='...'
uv run agent-benchmark benchmark zai-api \
  examples/tasks/support-routing.yaml \
  examples/tasks/support-routing.samples.jsonl \
  --metadata-only \
  --otlp-endpoint http://localhost:4318
```

`agent-benchmark benchmark zai-opencode` runs Coding Plan sessions through an isolated OpenCode
process, and `agent-benchmark provider zai-smoke --endpoint-plan general` checks connectivity. Both
transports, their credential boundaries, and their retained evidence are described in
[model benchmarking](docs/model-benchmarking.md).

## Describe evidence and gate a claim

Describe an artifact by digest without copying or printing its contents:

```bash
uv run agent-benchmark evidence runs/example/full-trace.jsonl \
  --uri object://research/example/full-trace.jsonl
```

Copy `examples/auditable-session.example.json`, replace every synthetic value with observed facts,
complete an independent review, and gate the claim:

```bash
uv run agent-benchmark gate runs/example/session.json \
  --report runs/example/audit-report.md \
  --object-root runs/object-store
```

Exit code `0` means the headline gate passed or did not apply, `2` means a valid session was blocked,
and `1` means the input was invalid. `--object-root` is optional; with it, the content-addressed
audit bundle is written to that AgentTrace object store without overwriting an existing bundle.

Use `JsonlTraceSink` for a bounded append-only local trace and `summarize_outcomes` to keep harness
submission separate from a judge-confirmed resolution. See [the integrity protocol](docs/integrity.md)
and [the provenance note](UPSTREAM.md).

## Privacy

Raw traces, run outputs, `.env` files, and credentials are ignored and must stay outside Git.
