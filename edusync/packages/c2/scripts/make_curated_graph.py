"""Draft a curated C1-v1 graph JSON for one lecture deck, for people to correct.

The draft takes concepts from the slide titles (generic and empty titles skipped) and adds
mention-based ``inferred`` prerequisite edges (confidence 0.5 to 0.8): A precedes B when A's
title appears on slides where B is discussed and A's first slide is earlier (at most 5 per
concept; a concept with none is a root). ``--prereq-map`` adds or
replaces edges from a JSON object ``{"label": ["prerequisite label", ...]}`` with confidence 0.9
and source_type ``curriculum_defined``. The draft is written with ``"curated": false``: after
correcting it, set ``"curated": true`` and load it with ``EDUSYNC_C2_GRAPH_PROVIDER=file``.

Exit codes: 0 written, 1 the draft has contract errors (nothing written), 2 unreadable input,
bad option value, or the output exists (use --force).
"""

import argparse
import json
import sys
from pathlib import Path

from edusync_c2.difficulty.concept_filter import is_generic_label
from edusync_c2.difficulty.concepts import locate_slides
from edusync_c2.difficulty.contract_check import check_c1_graph
from edusync_c2.difficulty.ids import slug_id
from edusync_c2.difficulty.materials import MaterialsError, parse_materials

_DIRECTION = "source_is_prerequisite_of_target"
_MAX_PREREQS = 5
_REVIEW = [
    "Check every node: delete slide titles that are not concepts, add missing concepts.",
    "Edges are inferred from mentions: A is a candidate prerequisite of B when A's title appears "
    "on slides where B is discussed and A's slide comes first. Mentions are not prerequisites, "
    "so correct them by hand: remove wrong edges, add the real ones "
    "(source must be understood before target). A concept with no prerequisite is a valid root.",
    "Raise confidence and set source_type to curriculum_defined for edges you confirmed.",
    "Replace each node description with a one-line definition.",
    "When done, set top-level 'curated' to true and run validate_c1_graph.py.",
]


def _usable(title: str | None) -> bool:
    t = (title or "").strip()
    return bool(slug_id(t)) and not t.endswith("?") and not is_generic_label(t, extras=True)


def _edge(source: str, target: str, confidence: float, source_type: str, page: int, text: str):
    return {
        "source": source,
        "target": target,
        "relation_type": "prerequisite",
        "direction": _DIRECTION,
        "confidence": confidence,
        "source_type": source_type,
        "evidence_count": 1,
        "evidence": [{"page": page, "sentence": text}],
    }


def build_draft(
    data: bytes,
    filename: str,
    *,
    min_frequency: int = 1,
    prereq_map: dict[str, list[str]] | None = None,
) -> dict:
    """The draft graph payload. Raises ``MaterialsError`` or ``ValueError`` (unknown map label)."""
    materials = parse_materials(data, filename)
    first: dict[str, tuple[str, int]] = {}
    for s in materials.slides:
        if _usable(s.title):
            label = (s.title or "").strip()
            first.setdefault(slug_id(label), (label, s.index))
    freq = {i: len(locate_slides(lab, i, materials.slides)) for i, (lab, _) in first.items()}
    ids = [i for i in first if freq[i] >= min_frequency]

    located = {i: set(locate_slides(first[i][0], i, materials.slides)) for i in ids}
    edges: dict[tuple[str, str], dict] = {}
    for b in ids:
        found: list[tuple[float, int, str, int]] = []
        for a in ids:
            if first[a][1] >= first[b][1]:
                continue  # edges only run forward in first-slide order: no cycles
            shared = located[a] & located[b]
            if shared:
                conf = min(0.8, 0.5 + 0.1 * min(3, len(shared) - 1))
                found.append((-conf, first[a][1], a, min(shared)))
        for neg_conf, _, a, page in sorted(found)[:_MAX_PREREQS]:
            edges[(a, b)] = _edge(a, b, -neg_conf, "inferred", page, "")
    for target_label, prereqs in sorted((prereq_map or {}).items()):
        target = slug_id(target_label)
        for prereq_label in prereqs:
            source = slug_id(prereq_label)
            for name, node_id in ((target_label, target), (prereq_label, source)):
                if node_id not in ids:
                    raise ValueError(f"prereq-map label {name!r} is not a concept in the draft")
            edges[(source, target)] = _edge(
                source,
                target,
                0.9,
                "curriculum_defined",
                first[target][1],
                f"Curriculum: '{prereq_label}' is a prerequisite of '{target_label}'.",
            )

    nodes = [
        {
            "id": i,
            "label": first[i][0],
            "description": f"Draft from the title of slide {first[i][1]}: replace this text.",
            "frequency": freq[i],
        }
        for i in ids
    ]
    return {
        "lecture_id": slug_id(Path(filename).stem) or None,
        "filename": filename,
        "page_count": materials.slide_count,
        "concept_count": len(nodes),
        "concepts": [{"name": n["label"], "frequency": n["frequency"]} for n in nodes],
        "graph": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": list(edges.values()),
        },
        "curated": False,
        "review": list(_REVIEW),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", required=True, help="Lecture deck (.pptx or .pdf)")
    parser.add_argument("--out", required=True, help="Where to write the draft graph JSON")
    parser.add_argument("--min-frequency", type=int, default=1, help="Drop titles on fewer slides")
    parser.add_argument("--prereq-map", help="JSON object: label -> list of prerequisite labels")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists; use --force to overwrite", file=sys.stderr)
        return 2
    prereq_map = None
    try:
        deck = Path(args.deck)
        data = deck.read_bytes()
        if args.prereq_map:
            prereq_map = json.loads(Path(args.prereq_map).read_text(encoding="utf-8-sig"))
        draft = build_draft(
            data, deck.name, min_frequency=args.min_frequency, prereq_map=prereq_map
        )
    except (OSError, ValueError, AttributeError, TypeError, MaterialsError) as exc:
        print(f"Cannot build the draft: {exc}", file=sys.stderr)
        return 2

    report = check_c1_graph(draft, "v1")
    for f in report.findings:
        print(f"  {f.severity.upper()} {f.rule} [{f.location}]: {f.message}", file=sys.stderr)
    if not report.ok:
        print("Draft has contract errors; nothing written", file=sys.stderr)
        return 1
    out.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}: {report.stats['nodes']} nodes, {report.stats['edges']} edges (draft)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
