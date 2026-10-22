"""Check a C1 graph JSON file against the C1 -> C2 graph contract.

Exit codes: 0 ok (warnings allowed), 1 contract errors, 2 unreadable file or invalid JSON.
"""

import argparse
import json
import sys
from pathlib import Path

from edusync_c2.difficulty.contract_check import ContractReport, check_c1_graph


def format_report(report: ContractReport, path: str) -> str:
    status = "ok" if report.ok else "FAILED"
    stats = report.stats
    lines = [
        f"C1 graph check: {path}",
        f"Detected version: {report.version}   Result: {status}",
        f"Nodes: {stats['nodes']}   Edges: {stats['edges']} "
        f"(prerequisite {stats['prerequisite_edges']}, co_mentioned {stats['co_mentioned_edges']})",
    ]
    for severity in ("error", "warning"):
        for finding in (f for f in report.findings if f.severity == severity):
            where = f" [{finding.location}]" if finding.location else ""
            lines.append(f"  {severity.upper()} {finding.rule}{where}: {finding.message}")
    lines.append(f"Summary: {stats['errors']} error(s), {stats['warnings']} warning(s) - {status}")
    return "\n".join(lines)


def _emit(text: str) -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="Path to the C1 graph JSON file")
    parser.add_argument("--expected", choices=["auto", "v0", "v1"], default="auto")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    args = parser.parse_args(argv)

    try:
        payload = json.loads(Path(args.file).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:  # unreadable file, bad encoding or invalid JSON
        print(f"Cannot read {args.file}: {exc}", file=sys.stderr)
        return 2

    report = check_c1_graph(payload, args.expected)
    if args.json:
        _emit(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        _emit(format_report(report, args.file))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
