"""Export the spot-check rating sheet (seeded sample of lecture snippets) to data/real/.

Reads a lecture manifest (see ``manifest.py``), extracts features through the cache (label-free)
and writes ``<out>`` as CSV with blank rater_id, rating and notes columns. The sheet contains
copyrighted lecture text: keep it in the git-ignored data/real folder. Existing sheets are never
overwritten.
"""

import argparse
import sys
from pathlib import Path

from edusync_c2_model.extract import ExtractError, extract_all
from edusync_c2_model.manifest import ManifestError, read_manifest
from edusync_c2_model.spot_set import DEFAULT_N, SpotSetError, sample_snippets, write_sheet

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=PACKAGE_ROOT / "cache" / "real")
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT / "data" / "real" / "spot_set.csv")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        rows, _ = extract_all(read_manifest(args.manifest), args.cache_dir)
        snippets = sample_snippets(rows, n=args.n, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_sheet(args.out, snippets)
    except (ManifestError, ExtractError, SpotSetError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(snippets)} snippets to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
