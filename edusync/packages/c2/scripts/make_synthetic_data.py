"""Write deterministic synthetic lecture files, C1 graphs and rule-based labels.

SYNTHETIC DATA: the labels come from a rule over the same properties the decks expose, so any
model score on them is circular and must never be reported as accuracy.
"""

import argparse
import json
import sys
from pathlib import Path

from edusync_c2.difficulty.synthetic import (
    build_c1_graph,
    build_labels_csv,
    build_pdf,
    build_pptx,
    make_deck_spec,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic"

README = """\
Synthetic lecture data for the EduSync C2 difficulty pipeline.

- Everything here is synthetic: no real course material is used.
- The difficulty labels in labels.csv come from a transparent rule over the same properties the
  generated decks expose (see synthetic_label in edusync_c2/difficulty/synthetic.py).
- Therefore any model score measured against these labels is circular. Do not use it for
  accuracy claims; the data only exercises the pipeline end to end.

Regenerate (from the repository root, with the workspace environment active):

    python edusync/packages/c2/scripts/make_synthetic_data.py \
        --seed {seed} --lectures {lectures} --concepts {concepts}

Files per lecture k: lecture_<k>.pptx, lecture_<k>.pdf, lecture_<k>_graph_v0.json,
lecture_<k>_graph_v1.json. Shared: labels.csv, README.txt.
"""


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Generate the files; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lectures", type=int, default=3)
    parser.add_argument("--concepts", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--vary-layout",
        action="store_true",
        help="Vary slide geometry, fonts and diagram elements (needs an explicit --out)",
    )
    args = parser.parse_args(argv)
    if args.vary_layout and args.out.resolve() == DEFAULT_OUT.resolve():
        print(
            "--vary-layout would overwrite the committed data/synthetic files; "
            "pass an explicit --out directory elsewhere",
            file=sys.stderr,
        )
        return 2
    if args.lectures < 1:
        print("--lectures must be at least 1", file=sys.stderr)
        return 2
    try:
        specs = [
            make_deck_spec(args.seed + k, args.concepts, lecture_id=f"synthetic-lecture-{k}")
            for k in range(1, args.lectures + 1)
        ]
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    for k, spec in enumerate(specs, start=1):
        (args.out / f"lecture_{k}.pptx").write_bytes(build_pptx(spec, vary_layout=args.vary_layout))
        (args.out / f"lecture_{k}.pdf").write_bytes(build_pdf(spec))
        for version in ("v0", "v1"):
            _write_json(
                args.out / f"lecture_{k}_graph_{version}.json", build_c1_graph(spec, version)
            )
    (args.out / "labels.csv").write_text(build_labels_csv(specs), encoding="utf-8", newline="")
    readme = README.format(seed=args.seed, lectures=args.lectures, concepts=args.concepts)
    (args.out / "README.txt").write_text(readme, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
