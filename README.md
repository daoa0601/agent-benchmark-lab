# Agent Benchmark Lab

A fresh-start workbench for building auditable and reproducible agent benchmarks. It provides the
small integration layer a research runner needs without inheriting course artifacts, historic result
directories, cluster scripts, or unrelated experiment code.

The repository is source-only and begins at `0.1.0`. It is consumed from a sibling checkout; no
PyPI publication is required.

## Boundaries

- `benchmark-audit-core` owns portable evidence models, canonical digests, audit bundles, and
  deterministic headline gates.
- Agent Benchmark Lab owns safe trace capture, outcome semantics, evidence-description commands,
  and the workflow that applies the core gate to one experiment manifest.
- `agent-trace-platform` optionally persists an immutable audit bundle. It remains the owner of
  telemetry and object storage.
- A research repository owns its task runner, datasets, provider policy, prompts, and scientific
  claims.

## Checkout and verify

Place this repository beside its shared dependencies:

```text
agentic-orch/
├── agent-benchmark-lab/
├── agent-trace-platform/
└── benchmark-audit-core/
```

Then run:

```bash
uv sync --extra dev
make check
make build
```

## Workbench commands

Describe an artifact by digest without copying or printing its contents:

```bash
uv run agent-benchmark evidence runs/example/full-trace.jsonl \
  --uri object://research/example/full-trace.jsonl
```

Copy `examples/auditable-session.example.json`, replace every synthetic value with observed facts,
complete an independent review, and gate the claim:

```bash
uv run agent-benchmark gate runs/example/session.json \
  --report runs/example/audit-report.md
```

Exit code `0` means the headline gate passed, `2` means a valid session was blocked, and `1` means
the input or integration was invalid. Add `--object-root runs/object-store` only when the optional
AgentTrace dependency is installed.

Use `JsonlTraceSink` for a bounded append-only local trace and `summarize_outcomes` to keep harness
submission separate from a judge-confirmed resolution. See [the integrity protocol](docs/integrity.md)
and [the provenance note](UPSTREAM.md).

## Versioning and privacy

Tags use semantic versioning. `v0.1.0` establishes the workbench API; incompatible contract changes
remain possible until `1.0.0`. Raw traces, run outputs, `.env` files, and credentials are ignored and
must stay outside Git. The GitHub repository remains private until a separate public-readiness review.
