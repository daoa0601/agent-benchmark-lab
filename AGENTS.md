# Agent instructions

This repository runs and gates auditable agent benchmarks. It intentionally does not contain course
reports, result archives, slides, cluster scripts, or the Git history of an earlier research artifact.

- Keep portable schemas and deterministic gate logic in the sibling `benchmark-audit-core` package;
  this repository only calls them.
- Benchmark runners, provider transports, agent harnesses, and the `agent-benchmark` gate CLI live
  here. Durable trace storage, task contracts, telemetry, and redaction rules live in the sibling
  `agent-trace-platform`; use only its public modules listed in its README.
- Resolve sibling Python projects with uv path sources; see the catalog README for the layout.
- Never commit `.env` files, credentials, raw traces, run outputs, or private reasoning.
- Run `make check` before committing.
