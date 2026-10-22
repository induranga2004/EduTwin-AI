"""Write the bundled sample C1 graph (contract v1) used by the ``file`` graph provider.

SYNTHETIC DATA: produced by the synthetic generator, not by C1. Deterministic: running this again
rewrites the committed file byte for byte.
"""

import argparse
import json
import sys
from pathlib import Path

from edusync_c2.difficulty.synthetic import build_c1_graph, make_deck_spec

DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "edusync_c2"
    / "difficulty"
    / "resources"
    / "sample_c1_graph_v1.json"
)
SEED = 7
CONCEPTS = 6
LECTURE_ID = "sample-lecture"


def render() -> str:
    graph = build_c1_graph(make_deck_spec(SEED, CONCEPTS, lecture_id=LECTURE_ID), "v1")
    return json.dumps(graph, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render())
    sys.stdout.write(f"wrote {args.out.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
