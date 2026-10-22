"""Download CLEAR and OneStopEnglish into the git-ignored data/public folder and write the manifest.

Records sha256 and row counts in ``DATA_LICENCES.md`` (package root). Refuses to overwrite existing
files without --force. Exit codes: 0 ok, 1 download/read failure.
"""

import argparse
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

from edusync_c2_model.public_data.clear import load_clear
from edusync_c2_model.public_data.licences import (
    LICENCE_REMINDER,
    SOURCES,
    DownloadError,
    download,
    fetch_url,
    render_licences,
)
from edusync_c2_model.public_data.onestop import load_onestop

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def run(
    folder: Path,
    licences_path: Path,
    *,
    force: bool = False,
    fetch: Callable[[str], bytes] | None = None,
    today: str | None = None,
) -> dict[str, int]:
    """Download every source, count its rows, write the licence manifest; return the counts."""
    print(LICENCE_REMINDER)
    counts: dict[str, int] = {}
    loaders = {"clear": load_clear, "onestop": load_onestop}
    for source in SOURCES:
        path = download(source, folder, force=force, fetch=fetch or fetch_url)
        counts[source.key] = len(loaders[source.key](path))
        print(f"{source.key}: {path} ({counts[source.key]} rows)")
    text = render_licences(SOURCES, folder, counts, today or date.today().isoformat())
    licences_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {licences_path}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=PACKAGE_ROOT / "data" / "public")
    parser.add_argument("--licences", type=Path, default=PACKAGE_ROOT / "DATA_LICENCES.md")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)
    try:
        run(args.folder, args.licences, force=args.force)
    except (DownloadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
