"""Command-line interface for the benchmark workbench."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from agent_benchmark_lab.evidence import describe_evidence
from agent_benchmark_lab.gate import gate_session


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-benchmark", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    evidence = subcommands.add_parser("evidence", help="describe one artifact by digest")
    evidence.add_argument("path", type=Path)
    evidence.add_argument("--uri")
    evidence.add_argument("--media-type")

    gate = subcommands.add_parser("gate", help="gate one auditable session manifest")
    gate.add_argument("manifest", type=Path)
    gate.add_argument("--report", type=Path, required=True)
    gate.add_argument("--object-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a workbench command without echoing artifact contents on failure."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "evidence":
            output = describe_evidence(args.path, uri=args.uri, media_type=args.media_type)
            exit_code = 0
        else:
            summary = gate_session(
                args.manifest,
                report_path=args.report,
                object_root=args.object_root,
            )
            output = summary.as_dict()
            exit_code = 2 if summary.status == "blocked" else 0
    except Exception as error:
        print(
            json.dumps({"status": "invalid", "error_type": type(error).__name__}), file=sys.stderr
        )
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return exit_code
