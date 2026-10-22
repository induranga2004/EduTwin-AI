"""Required-key validator for the demo analysis JSON contract (shared by UI and backend tests)."""

from __future__ import annotations

from typing import Any

LAYER_GROUPS = ("A", "B", "C", "D")
LABELS = {"Low", "Medium", "High"}


def _need(obj: Any, keys: tuple[str, ...], where: str, errors: list[str]) -> bool:
    if not isinstance(obj, dict):
        errors.append(f"{where}: expected object")
        return False
    for k in keys:
        if k not in obj:
            errors.append(f"{where}: missing '{k}'")
    return True


def contract_errors(doc: Any) -> list[str]:
    """Return a list of contract violations (empty means the document is valid)."""
    errs: list[str] = []
    if not _need(doc, ("meta", "layers", "final", "warnings", "disclaimer"), "root", errs):
        return errs
    meta = doc["meta"]
    _need(
        meta,
        (
            "filename",
            "file_type",
            "slide_count",
            "concept_count",
            "pipeline_version",
            "score_version",
            "reference",
            "graph",
        ),
        "meta",
        errs,
    )
    if isinstance(meta, dict):
        _need(meta.get("reference"), ("n_decks", "provisional"), "meta.reference", errs)
        graph = meta.get("graph")
        if _need(graph, ("kind", "nodes", "edges"), "meta.graph", errs):
            for n in graph.get("nodes", []):
                _need(n, ("id", "label", "frequency"), "meta.graph.nodes[]", errs)
            for e in graph.get("edges", []):
                _need(
                    e, ("source", "target", "confidence", "source_type"), "meta.graph.edges[]", errs
                )
    layers = doc["layers"]
    if _need(layers, ("parse", "A", "B", "C", "D", "text", "combine"), "layers", errs):
        parse = layers.get("parse", {})
        if _need(parse, ("slides", "concepts"), "layers.parse", errs):
            for s in parse["slides"]:
                _need(s, ("index", "title", "words"), "parse.slides[]", errs)
            for c in parse["concepts"]:
                _need(c, ("id", "label", "slide_indexes"), "parse.concepts[]", errs)
        for g in LAYER_GROUPS:
            lg = layers.get(g, {})
            if _need(lg, ("group", "direction", "per_concept", "summary"), f"layers.{g}", errs):
                _need(lg["summary"], ("available_share",), f"layers.{g}.summary", errs)
                for pc in lg["per_concept"]:
                    _need(
                        pc,
                        ("concept_id", "features", "group_score", "available"),
                        f"layers.{g}.per_concept[]",
                        errs,
                    )
                    for f in pc.get("features", []) if isinstance(pc, dict) else []:
                        _need(
                            f,
                            ("name", "value", "status", "percentile", "note"),
                            f"layers.{g}.features[]",
                            errs,
                        )
        text = layers.get("text", {})
        if _need(text, ("models", "per_concept", "per_slide"), "layers.text", errs):
            _need(text["models"], ("t1", "t2"), "layers.text.models", errs)
            for pc in text["per_concept"]:
                _need(
                    pc, ("concept_id", "t1", "t2", "agreement"), "layers.text.per_concept[]", errs
                )
            for ps in text["per_slide"]:
                _need(ps, ("index", "t1", "t2"), "layers.text.per_slide[]", errs)
        comb = layers.get("combine", {})
        if _need(comb, ("weights", "cutoffs", "per_concept"), "layers.combine", errs):
            _need(comb.get("weights"), ("A", "B", "C", "D", "text"), "combine.weights", errs)
            _need(comb.get("cutoffs"), ("low", "high"), "combine.cutoffs", errs)
            for pc in comb["per_concept"]:
                _need(
                    pc,
                    (
                        "concept_id",
                        "contributions",
                        "score",
                        "label",
                        "confidence",
                        "confidence_parts",
                    ),
                    "combine.per_concept[]",
                    errs,
                )
                if isinstance(pc, dict) and pc.get("label") not in LABELS:
                    errs.append(f"combine.per_concept[]: bad label {pc.get('label')!r}")
    final = doc["final"]
    if _need(final, ("hotspots", "slide_scores", "concepts"), "final", errs):
        for h in final["hotspots"]:
            _need(
                h,
                ("rank", "kind", "slides", "score", "label", "confidence", "reasons"),
                "final.hotspots[]",
                errs,
            )
            if isinstance(h, dict) and "concept_id" not in h and "slide_index" not in h:
                errs.append("final.hotspots[]: needs concept_id or slide_index")
        for s in final["slide_scores"]:
            _need(
                s,
                ("index", "score", "label", "confidence", "concept_ids", "reasons"),
                "final.slide_scores[]",
                errs,
            )
    return errs
