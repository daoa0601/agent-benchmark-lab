# Benchmark integrity protocol

Apply these rules prospectively to autonomous benchmark claims:

1. Run one task instance per agent session and record the exact task and protocol versions.
2. Record the source revision, dirty state, observed hardware, runtime versions, and optimization
   scope from the live run.
3. Capture every required compile, check, and benchmark attempt, including failed iterations. Store
   the trace, final artifact, task definition, and protocol as content-addressed evidence.
4. Keep a finite watchdog and record the actual termination reason. A watchdog is a safety boundary,
   not proof that the agent chose to finish.
5. Give an independent reviewer the final artifact, full trace, task definition, and protocol.
   Automation may assist; it does not replace required manual review or empirical re-testing.
6. Re-test in an isolated process or workspace, including cold-cache behavior, input perturbation,
   and complete output validation. Performance work should randomize ordering and verify resource
   isolation.
7. Preserve every finding. An unresolved blocking finding prevents a headline claim even if someone
   accepts the risk.

`benchmark-audit-core` implements the portable schema and deterministic gate. This repository turns
that contract into the `agent-benchmark gate` workflow and supplies safe local trace and outcome
helpers. It does not infer hidden model-gateway attempts: record the configured and resolved provider
routes as evidence. Which layer owns retries and fallbacks is set in
[the catalog README](../../../README.md#cross-repo-boundaries).

The example manifest is synthetic documentation, never research evidence.
