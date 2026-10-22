"""Check a filled C2 label CSV against the labelling protocol (docs/edusync/c2-label-collection.md).

Rules are numbered L-001 and up. Errors fail the check; warnings do not. Rows with an empty
difficulty_label are unlabelled sheet rows: they are counted and skipped by the per-row rules.

--sheet SHEET.csv additionally requires every (lecture_id, concept_id) to exist in a sheet
exported by export_label_sheet.py. --agreement adds, for lectures labelled by two or more raters,
the quadratic weighted kappa (Low=0, Medium=1, High=2) and the exact agreement rate between the
first two raters (sorted by rater_id; the adjudicator ADJ is ignored), CannotJudge pairs excluded.
Agreement is a diagnostic for
the go/no-go check in the protocol; it never changes the exit code.

Exit codes: 0 ok (warnings allowed), 1 findings (errors), 2 unreadable or malformed input.
"""

import argparse
import csv
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

REQUIRED_COLUMNS = (
    "lecture_id",
    "concept_id",
    "difficulty_label",
    "rater_id",
    "student_level",
    "confidence",
    "labelled_at",
    "rubric_version",
    "synthetic",
)
LABELS = ("Low", "Medium", "High", "CannotJudge")
ORDINAL = {"Low": 0, "Medium": 1, "High": 2}
STUDENT_LEVELS = ("first_year", "second_year", "third_year", "postgraduate", "mixed")
PSEUDONYM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
KAPPA_BAR = 0.6
ADJUDICATOR = "ADJ"  # rater id of adjudicated rows; never part of the agreement

RULES = {
    "L-001": "required columns are present",
    "L-002": "lecture_id and concept_id are not empty",
    "L-003": "difficulty_label is Low, Medium, High or CannotJudge (case-sensitive)",
    "L-004": "rater_id is not empty on a labelled row",
    "L-005": "student_level is one of " + ", ".join(STUDENT_LEVELS),
    "L-006": "confidence is an integer 1-5 or empty",
    "L-007": "labelled_at is an ISO date (YYYY-MM-DD) or datetime",
    "L-008": "rubric_version is set and the same on every labelled row",
    "L-009": "synthetic is true or false",
    "L-010": "no duplicate (lecture_id, concept_id, rater_id)",
    "L-011": "every (lecture_id, concept_id) exists in the exported sheet (--sheet)",
    "L-012": "warning: unlabelled rows remain",
    "L-013": "warning: synthetic and real rows are mixed in one file",
    "L-014": "warning: lecture_id or rater_id is not a short pseudonymous token",
}


class Findings:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, rule: str, severity: str, location: str, message: str) -> None:
        self.items.append(
            {"rule": rule, "severity": severity, "location": location, "message": message}
        )

    def count(self, severity: str) -> int:
        return sum(1 for f in self.items if f["severity"] == severity)


def _valid_date(text: str) -> bool:
    try:
        if len(text) == 10:
            date.fromisoformat(text)
        else:
            datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def read_rows(path: Path) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    """Header and (line number, stripped row) pairs; blank rows are skipped."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [h.strip() for h in (reader.fieldnames or [])]
        rows = []
        for raw in reader:
            row = {
                (k or "").strip(): (v or "").strip()
                for k, v in raw.items()
                if k is not None and not isinstance(v, list)
            }
            if any(row.values()):
                rows.append((reader.line_num, row))
    return header, rows


def check_rows(
    header: list[str],
    rows: list[tuple[int, dict[str, str]]],
    sheet_keys: set[tuple[str, str]] | None = None,
) -> tuple[Findings, list[dict[str, str]]]:
    """Apply the rules; return the findings and the labelled rows (valid or not)."""
    findings = Findings()
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        findings.add("L-001", "error", "header", "missing column(s): " + ", ".join(missing))
        return findings, []

    labelled: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    unlabelled = 0
    rubrics: set[str] = set()
    synthetic_values: set[str] = set()
    for line, row in rows:
        where = f"row {line}"
        lecture, concept = row["lecture_id"], row["concept_id"]
        if not lecture or not concept:
            findings.add("L-002", "error", where, "lecture_id and concept_id must not be empty")
        if sheet_keys is not None and (lecture, concept) not in sheet_keys:
            findings.add(
                "L-011", "error", where, f"{concept!r} of lecture {lecture!r} is not in the sheet"
            )
        if row["synthetic"] not in ("true", "false"):
            findings.add(
                "L-009",
                "error",
                where,
                f"synthetic must be true or false, got {row['synthetic']!r}",
            )
        else:
            synthetic_values.add(row["synthetic"])
        label = row["difficulty_label"]
        if not label:
            unlabelled += 1
            continue
        labelled.append(row)
        if label not in LABELS:
            findings.add(
                "L-003", "error", where, f"difficulty_label {label!r} is not one of {list(LABELS)}"
            )
        rater = row["rater_id"]
        if not rater:
            findings.add("L-004", "error", where, "rater_id is empty on a labelled row")
        elif not PSEUDONYM.match(rater):
            findings.add("L-014", "warning", where, "rater_id is not a short pseudonymous token")
        if lecture and not PSEUDONYM.match(lecture):
            findings.add("L-014", "warning", where, "lecture_id is not a short pseudonymous token")
        if row["student_level"] not in STUDENT_LEVELS:
            findings.add(
                "L-005",
                "error",
                where,
                f"student_level {row['student_level']!r} is not one of {list(STUDENT_LEVELS)}",
            )
        conf = row["confidence"]
        if conf and not (re.fullmatch(r"[0-9]+", conf) and 1 <= int(conf) <= 5):
            findings.add("L-006", "error", where, f"confidence {conf!r} is not an integer 1-5")
        if not _valid_date(row["labelled_at"]):
            findings.add(
                "L-007", "error", where, f"labelled_at {row['labelled_at']!r} is not an ISO date"
            )
        rubrics.add(row["rubric_version"])
        key = (lecture, concept, rater)
        if key in seen:
            findings.add(
                "L-010", "error", where, f"duplicate label for {concept!r} by rater {rater!r}"
            )
        seen.add(key)

    if "" in rubrics:
        findings.add("L-008", "error", "file", "rubric_version is empty on a labelled row")
    if len(rubrics - {""}) > 1:
        findings.add(
            "L-008", "error", "file", f"mixed rubric_version values: {sorted(rubrics - {''})}"
        )
    if unlabelled:
        findings.add("L-012", "warning", "file", f"{unlabelled} unlabelled row(s) remain")
    if len(synthetic_values) > 1:
        findings.add(
            "L-013", "warning", "file", "file mixes synthetic=true and synthetic=false rows"
        )
    return findings, labelled


def summarise(rows: list[tuple[int, dict[str, str]]], labelled: list[dict[str, str]]) -> dict:
    distribution = dict.fromkeys(LABELS, 0)
    for row in labelled:
        if row["difficulty_label"] in distribution:
            distribution[row["difficulty_label"]] += 1
    share = round(distribution["CannotJudge"] / len(labelled), 4) if labelled else None
    return {
        "rows": len(rows),
        "labelled_rows": len(labelled),
        "lectures": len({r["lecture_id"] for _, r in rows if r["lecture_id"]}),
        "raters": len({r["rater_id"] for r in labelled if r["rater_id"]}),
        "label_distribution": distribution,
        "cannot_judge_share": share,
    }


def quadratic_weighted_kappa(pairs: list[tuple[int, int]], categories: int = 3) -> float | None:
    """Quadratic weighted kappa of (rater 1, rater 2) category pairs; None when undefined.

    kappa = 1 - sum(w * observed) / sum(w * expected), w = (i - j)^2 / (categories - 1)^2, where
    expected is the product of the two raters' marginal proportions. It is undefined for no pairs
    or when chance disagreement is zero (both raters used one single category).
    """
    n = len(pairs)
    if n == 0:
        return None
    observed = [[0.0] * categories for _ in range(categories)]
    for a, b in pairs:
        observed[a][b] += 1.0 / n
    row = [sum(observed[i]) for i in range(categories)]
    col = [sum(observed[i][j] for i in range(categories)) for j in range(categories)]
    num = den = 0.0
    for i in range(categories):
        for j in range(categories):
            weight = (i - j) ** 2 / (categories - 1) ** 2
            num += weight * observed[i][j]
            den += weight * row[i] * col[j]
    if den == 0:
        return None
    return round(1 - num / den, 4)


def agreement(labelled: list[dict[str, str]]) -> dict:
    """Per-lecture and pooled agreement between the first two raters of each lecture."""
    by_lecture: dict[str, dict[str, dict[str, str]]] = {}
    for row in labelled:
        if not row["rater_id"] or row["rater_id"] == ADJUDICATOR:
            continue
        by_lecture.setdefault(row["lecture_id"], {}).setdefault(row["rater_id"], {}).setdefault(
            row["concept_id"], row["difficulty_label"]
        )
    lectures = []
    pooled: list[tuple[int, int]] = []
    for lecture in sorted(by_lecture):
        raters = sorted(by_lecture[lecture])
        if len(raters) < 2:
            continue
        first, second = raters[0], raters[1]
        pairs = [
            (ORDINAL[a], ORDINAL[by_lecture[lecture][second][concept]])
            for concept, a in sorted(by_lecture[lecture][first].items())
            if a in ORDINAL and by_lecture[lecture][second].get(concept) in ORDINAL
        ]
        pooled.extend(pairs)
        lectures.append({"lecture_id": lecture, "raters": [first, second], **_pair_stats(pairs)})
    return {"lectures": lectures, "pooled": _pair_stats(pooled), "kappa_bar": KAPPA_BAR}


def _pair_stats(pairs: list[tuple[int, int]]) -> dict:
    exact = round(sum(1 for a, b in pairs if a == b) / len(pairs), 4) if pairs else None
    return {"pairs": len(pairs), "exact_agreement": exact, "qwk": quadratic_weighted_kappa(pairs)}


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_report(path: str, findings: Findings, summary: dict, agree: dict | None) -> str:
    status = "ok" if findings.count("error") == 0 else "FAILED"
    dist = summary["label_distribution"]
    lines = [
        f"Label check: {path}   Result: {status}",
        f"Rows: {summary['rows']} (labelled {summary['labelled_rows']})   "
        f"Lectures: {summary['lectures']}   Raters: {summary['raters']}",
        "Labels: "
        + ", ".join(f"{k}={v}" for k, v in dist.items())
        + "   CannotJudge share: "
        + _fmt(summary["cannot_judge_share"]),
    ]
    for severity in ("error", "warning"):
        for f in (f for f in findings.items if f["severity"] == severity):
            lines.append(f"  {severity.upper()} {f['rule']} [{f['location']}]: {f['message']}")
    if agree is not None:
        lines.append("Agreement (diagnostic, first two raters per lecture, CannotJudge excluded):")
        if not agree["lectures"]:
            lines.append("  no lecture has two or more raters")
        for item in agree["lectures"]:
            lines.append(
                f"  {item['lecture_id']} {item['raters'][0]}/{item['raters'][1]}: "
                f"pairs={item['pairs']} exact={_fmt(item['exact_agreement'])} "
                f"qwk={_fmt(item['qwk'])}"
            )
        pooled = agree["pooled"]
        lines.append(
            f"  pooled: pairs={pooled['pairs']} exact={_fmt(pooled['exact_agreement'])} "
            f"qwk={_fmt(pooled['qwk'])} (protocol bar: qwk >= {KAPPA_BAR})"
        )
    lines.append(
        f"Summary: {findings.count('error')} error(s), {findings.count('warning')} warning(s) - "
        f"{status}"
    )
    return "\n".join(lines)


def _emit(text: str) -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def main(argv: list[str] | None = None) -> int:
    """Validate the file; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="Filled label CSV")
    parser.add_argument("--sheet", help="Exported sheet CSV: every concept id must exist in it")
    parser.add_argument("--agreement", action="store_true", help="Report rater agreement")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments, 0 on --help
        return int(exc.code or 0)

    try:
        header, rows = read_rows(Path(args.file))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        print(f"Cannot read {args.file}: {exc}", file=sys.stderr)
        return 2
    if not header:
        print(f"Cannot read {args.file}: no header row", file=sys.stderr)
        return 2

    sheet_keys = None
    if args.sheet:
        try:
            sheet_header, sheet_rows = read_rows(Path(args.sheet))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            print(f"Cannot read {args.sheet}: {exc}", file=sys.stderr)
            return 2
        if "lecture_id" not in sheet_header or "concept_id" not in sheet_header:
            print(
                f"Cannot use {args.sheet}: needs lecture_id and concept_id columns", file=sys.stderr
            )
            return 2
        sheet_keys = {(r["lecture_id"], r["concept_id"]) for _, r in sheet_rows}

    findings, labelled = check_rows(header, rows, sheet_keys)
    summary = summarise(rows, labelled)
    agree = agreement(labelled) if args.agreement else None
    ok = findings.count("error") == 0
    if args.json:
        payload = {"ok": ok, "summary": summary, "findings": findings.items}
        if agree is not None:
            payload["agreement"] = agree
        _emit(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        _emit(format_report(args.file, findings, summary, agree))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
