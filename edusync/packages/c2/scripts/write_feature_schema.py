"""Write feature_schema.json for a trained C2 model package from the feature registry.

The file holds schema_version, names (registry order) and hash, exactly what
OnnxFusionModel.from_artifacts checks. Write it at training time, next to the ONNX graph, so a
later registry change is detected. With --check PATH an existing file is compared with the current
registry instead of being written.

Exit codes: 0 ok, 1 --check found a mismatch, 2 bad arguments, unreadable file or unwritable path.
"""

import argparse
import json
import sys
from pathlib import Path

from edusync_c2.difficulty.registry import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    feature_schema_hash,
)

DEFAULT_OUT = Path("feature_schema.json")


def schema_payload() -> dict:
    """The registry as the artifact file expects it."""
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "names": list(FEATURE_NAMES),
        "hash": feature_schema_hash(),
    }


def schema_text() -> str:
    return json.dumps(schema_payload(), indent=1) + "\n"


def differences(existing: object) -> list[str]:
    """Names of the fields in ``existing`` that differ from the registry (empty when equal)."""
    expected = schema_payload()
    if not isinstance(existing, dict):
        return ["file is not a JSON object"]
    return [
        f"{field} differs (file {existing.get(field)!r:.80}, registry {value!r:.80})"
        if field != "names"
        else _names_difference(existing.get("names"))
        for field, value in expected.items()
        if existing.get(field) != value
    ]


def _names_difference(names: object) -> str:
    expected = list(FEATURE_NAMES)
    if not isinstance(names, list):
        return "names differs (not a list)"
    missing = [n for n in expected if n not in names]
    extra = [n for n in names if n not in expected]
    if missing or extra:
        return f"names differs (missing {missing[:3]}, unexpected {extra[:3]})"
    return "names differs (same names, different order)"


def _emit(text: str) -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def main(argv: list[str] | None = None) -> int:
    """Write or check the file; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where to write the file")
    parser.add_argument("--check", metavar="PATH", help="Compare an existing file, do not write")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments, 0 on --help
        return int(exc.code or 0)

    if args.check:
        try:
            existing = json.loads(Path(args.check).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            print(f"Cannot read {args.check}: {exc}", file=sys.stderr)
            return 2
        found = differences(existing)
        if found:
            for line in found:
                _emit(f"MISMATCH {line}")
            _emit(f"{args.check} does not match the registry (schema {FEATURE_SCHEMA_VERSION})")
            return 1
        _emit(f"{args.check} matches the registry (schema {FEATURE_SCHEMA_VERSION})")
        return 0

    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(schema_text())
    except OSError as exc:
        print(f"Cannot write {args.out}: {exc}", file=sys.stderr)
        return 2
    _emit(
        f"Wrote {args.out}: schema {FEATURE_SCHEMA_VERSION}, {len(FEATURE_NAMES)} features, "
        f"hash {feature_schema_hash()[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
