# Agent instructions

This repository is a fresh, source-only workbench for auditable agent benchmarks. It intentionally
does not contain course reports, result archives, slides, cluster scripts, or the Git history of an
earlier research artifact.

- Keep portable schemas and deterministic gate logic in the sibling `benchmark-audit-core` package.
- Keep durable trace storage and telemetry adapters in the sibling `agent-trace-platform` package.
- Put experiment-specific runners in their own research repositories; this project supplies only the
  reusable workbench layer between a runner and those shared components.
- Resolve sibling Python projects with uv path sources. Do not assume PyPI publication.
- Never commit `.env` files, credentials, raw traces, run outputs, or private reasoning.
- Run `make check` and `make build` before committing.
