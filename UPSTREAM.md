# Provenance and fresh-start boundary

This project was created with new Git history to house reusable benchmark-integrity ideas that had
previously been exercised inside a course research repository. No course report, result archive,
slide deck, cluster script, submitted artifact, or original repository history is included here.

The machine-readable audit contract is imported from the sibling `benchmark-audit-core` repository,
and trace, task, and storage APIs from `agent-trace-platform`. The Z.AI and OpenCode benchmark runners
were moved here from `agent-trace-platform` so that repository stays a trace lake.

When a research project needs these facilities, it should import this source checkout or reproduce
the small integration at its own boundary. Research-specific claims and data remain in that research
project.
