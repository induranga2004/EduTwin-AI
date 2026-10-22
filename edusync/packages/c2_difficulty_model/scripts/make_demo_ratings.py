"""Write the ILLUSTRATIVE demo ratings for the Lecture 07 snippet sheet (deck id L06).

These ratings are NOT ground truth. They are one assistant's reading of each snippet, made for the
demo so the evaluation path can be shown end to end. They are not lecturer or student labels and
must never be reported as validation of the model.

RUBRIC (rating = how hard the passage is for a typical second-year undergraduate meeting it for
the first time):
  Low    plain definitions, overview, agenda or housekeeping text; few unfamiliar terms.
  Medium needs some prior terms or follows a worked mechanism / comparison of named algorithms.
  High   dense formal or mathematical content, or multi-step algorithm detail with many unfamiliar
         terms.
The distribution is whatever the reading gave; it is deliberately not balanced.

Only snippet ids are stored below, never slide text. Input: the sheet written by the demo export
(data/real/spot_L06.csv, git-ignored). Output: data/real/ratings_L06_demo.csv (git-ignored),
rater_id demo-assistant, column illustrative = true, '#' comment header with the rubric.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RATER_ID = "demo-assistant"
RUBRIC = (
    "ILLUSTRATIVE demo ratings, not ground truth and not lecturer or student labels.",
    "rating = how hard the passage is for a typical second-year undergraduate meeting it for the "
    "first time.",
    "Low = plain definitions/overview; Medium = needs some prior terms or a worked mechanism; "
    "High = dense formal/mathematical or multi-step algorithm detail with many unfamiliar terms.",
)
COLUMNS = ("snippet_id", "lecture_id", "slide_index", "rater_id", "rating", "illustrative")

_L, _M, _H = "Low", "Medium", "High"
RATINGS: dict[str, str] = {
    "D001": _L, "D002": _L, "D003": _L, "D004": _M, "D005": _L,
    "D006": _M, "D007": _M, "D008": _M, "D009": _L, "D010": _L,
    "D011": _M, "D012": _M, "D013": _M, "D014": _M, "D015": _M,
    "D016": _L, "D017": _M, "D018": _M, "D019": _H, "D020": _H,
    "D021": _M, "D022": _L, "D023": _M, "D024": _M, "D025": _M,
    "D026": _M, "D027": _L, "D028": _M, "D029": _M, "D030": _M,
    "D031": _M, "D032": _L, "D033": _M, "D034": _H, "D035": _H,
    "D036": _M, "D037": _M, "D038": _M, "D039": _M, "D040": _L,
}  # fmt: skip


def build_rows(sheet: Path) -> list[dict[str, str]]:
    """Ratings rows for every snippet id in the sheet that has a rating; ids only."""
    with open(sheet, encoding="utf-8-sig", newline="") as handle:
        sheet_rows = list(csv.DictReader(handle))
    return [
        {
            "snippet_id": row["snippet_id"],
            "lecture_id": row["lecture_id"],
            "slide_index": row["slide_index"],
            "rater_id": RATER_ID,
            "rating": RATINGS[row["snippet_id"]],
            "illustrative": "true",
        }
        for row in sheet_rows
        if row["snippet_id"] in RATINGS
    ]


def write_ratings(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "x", encoding="utf-8", newline="") as handle:
        for line in RUBRIC:
            handle.write(f"# {line}\n")
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    data = PACKAGE_ROOT / "data" / "real"
    parser.add_argument("--sheet", type=Path, default=data / "spot_L06.csv")
    parser.add_argument("--out", type=Path, default=data / "ratings_L06_demo.csv")
    args = parser.parse_args(argv)
    try:
        rows = build_rows(args.sheet)
        write_ratings(args.out, rows)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    counts = Counter(r["rating"] for r in rows)
    print(f"wrote {len(rows)} illustrative ratings to {args.out}: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
