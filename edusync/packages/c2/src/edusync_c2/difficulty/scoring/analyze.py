"""Layered analysis: run the frozen pipeline once and assemble every layer of the score.

The JSON is deterministic (no timings, no random ids) and carries no slide text beyond the slide
titles and concept labels the lecturer's own deck already contains. Logs hold counts only.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from edusync_c2.difficulty.concepts import MockConceptFilter
from edusync_c2.difficulty.models import RuleBaselineModel
from edusync_c2.difficulty.pipeline import DifficultyPipeline
from edusync_c2.difficulty.providers import GraphProvider, MockC1GraphProvider, PayloadGraphProvider
from edusync_c2.difficulty.schemas import (
    C1GraphResponse,
    FusionInput,
    GraphContext,
    MaterialsDoc,
    ModelOutput,
)
from edusync_c2.difficulty.scoring.artifact import ScoreArtifact
from edusync_c2.difficulty.scoring.combiner import SIGNALS
from edusync_c2.difficulty.scoring.confidence import evidence_confidence
from edusync_c2.difficulty.scoring.hotspots import (
    build_hotspots,
    build_slide_scores,
    concept_reasons,
)
from edusync_c2.difficulty.scoring.signals import (
    DIRECTIONS,
    FEATURE_GROUP,
    GROUP_TITLES,
    GROUPS,
    FeatureScore,
    group_scores,
    score_features,
)
from edusync_c2.difficulty.scoring.text_models import TEXT_CAP
from edusync_c2.logger import get_logger

logger = get_logger(__name__)

SCORE_VERSION = "1.0"
STATUS_NAMES = {"computed": "ok", "mocked": "mocked", "unavailable": "unavailable"}
GROUP_DIRECTION = {
    "A": "higher = harder",
    "B": "higher = harder (more and deeper prerequisites)",
    "C": "inverted: more definitions, examples, steps and diagrams = easier",
    "D": "higher = harder (denser slides)",
}
MODEL_TEXT = {
    "t1": "T1 fine-tuned MiniLM (ONNX INT8), general-English difficulty",
    "t2": "T2 frozen MiniLM embedding + simple features, ridge",
}
AGREE_BELOW = 0.15
PARTIAL_BELOW = 0.35
DISCLAIMER = (
    "Evidence-based estimate, not a probability. Scores are positions among a reference of "
    "{n} lecture decks ({status}); the text models were trained on general English, not lecture "
    "slides; weights and cut-offs are unvalidated defaults."
)


HIGH_SHARE = 0.25  # top quarter of the lecture's parts are labelled High
LOW_SHARE = 0.40  # bottom 40% are labelled Low
LABELLING_NOTE = (
    "Labels are relative to this lecture: the hardest quarter of its concepts (or slides) is "
    "High, the easiest 40% is Low. Scores still compare with the reference decks."
)


@dataclass(frozen=True)
class RelativeLabels:
    labels: dict
    low: float | None
    high: float | None


def relative_labels(
    items: Sequence[tuple[object, float | None]],
    high_share: float = HIGH_SHARE,
    low_share: float = LOW_SHARE,
) -> RelativeLabels:
    """Label parts by rank inside the lecture; ties share a label; no score -> Medium.

    Items are ``(key, score)``. The effective score cut-offs are returned for display.
    """
    scored = sorted(((s, i) for i, (_, s) in enumerate(items) if s is not None), reverse=True)
    n = len(scored)
    labels: dict = {key: "Medium" for key, _ in items}
    if n == 0:
        return RelativeLabels(labels, None, None)
    n_high = max(1, round(n * high_share))
    n_low = round(n * low_share)
    high_cut = scored[n_high - 1][0]
    low_cut = scored[n - n_low][0] if n_low else None
    for score, i in scored:
        key = items[i][0]
        if score >= high_cut:
            labels[key] = "High"
        elif low_cut is not None and score <= low_cut:
            labels[key] = "Low"
    return RelativeLabels(labels, low_cut, high_cut)


class RecordingModel:
    """Captures the pipeline's ``FusionInput`` list and delegates to the rule baseline."""

    def __init__(self) -> None:
        self._inner = RuleBaselineModel()
        self.info = self._inner.info
        self.captured: list[FusionInput] = []

    def predict(self, inputs: list[FusionInput]) -> list[ModelOutput | None]:
        self.captured = list(inputs)
        return self._inner.predict(inputs)


class _SpyProvider:
    """Wraps a graph provider and remembers what it returned."""

    def __init__(self, inner: GraphProvider, seen: dict) -> None:
        self.inner = inner
        self.seen = seen
        self.name = getattr(inner, "name", "spy")

    def build(self, materials: MaterialsDoc) -> GraphContext:
        ctx = self.inner.build(materials)
        self.seen["ctx"] = ctx
        self.seen["materials"] = materials
        return ctx


def _agreement(t1: float | None, t2: float | None) -> str | None:
    if t1 is None or t2 is None:
        return None
    gap = abs(t1 - t2)
    return "agree" if gap < AGREE_BELOW else "partial" if gap < PARTIAL_BELOW else "disagree"


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _text_percentiles(scorer, texts: list[str], name: str, warnings: list[str]) -> dict[str, float]:
    """Percentile per distinct non-blank text from one model; {} when it is off or fails."""
    if scorer is None or not texts:
        return {}
    try:
        return dict(zip(texts, scorer.percentiles(texts), strict=True))
    except Exception as exc:
        logger.warning("text_model_failed name=%s reason=%s", name, type(exc).__name__)
        warnings.append(f"Text model {name.upper()} failed at run time and was skipped")
        return {}


def _feature_json(s: FeatureScore) -> dict:
    return {
        "name": s.name,
        "value": s.value,
        "status": STATUS_NAMES[s.status],
        "percentile": s.percentile,
        "direction": s.direction,
        "effect": s.effect,
        "note": s.note,
    }


def analyze(
    data: bytes,
    filename: str,
    *,
    artifact: ScoreArtifact,
    graph: C1GraphResponse | None = None,
    provider_factory: Callable[[bytes, str], GraphProvider] | None = None,
    concept_filter=None,
    semantic=None,
    semantic_status: str = "off",
    top_n: int = 5,
    max_mb: int = 100,
) -> dict:
    """Run the pipeline once and return the layered JSON (see the demo plan's contract)."""
    recorder = RecordingModel()
    seen: dict = {}

    def factory(raw: bytes, name: str) -> GraphProvider:
        if graph is not None:
            inner: GraphProvider = PayloadGraphProvider(graph)
        elif provider_factory is not None:
            inner = provider_factory(raw, name)
        else:
            inner = MockC1GraphProvider()
        return _SpyProvider(inner, seen)

    pipeline = DifficultyPipeline(
        model=recorder,
        concept_filter=concept_filter or MockConceptFilter(),
        graph_provider_factory=factory,
        semantic=semantic,
        semantic_status=semantic_status,
        max_mb=max_mb,
    )
    response = pipeline.run(data, filename, None)
    ctx: GraphContext = seen["ctx"]
    ref, combiner = artifact.reference, artifact.combiner
    graph_kind = ctx.source.kind
    fusion_text = {f.concept_id: f.text for f in recorder.captured}
    slides = seen["materials"].slides

    warnings = list(response.warnings) + list(artifact.warnings)
    predictions = response.predictions

    # ---- texts through the two models (one batch per model)
    concept_text = {p.concept_id: fusion_text[p.concept_id][:TEXT_CAP] for p in predictions}
    slide_text = {s.index: s.text[:TEXT_CAP] for s in slides}
    distinct = sorted({t for t in (*concept_text.values(), *slide_text.values()) if t.strip()})
    pct = {
        "t1": _text_percentiles(artifact.t1, distinct, "t1", warnings),
        "t2": _text_percentiles(artifact.t2, distinct, "t2", warnings),
    }

    def text_of(text: str | None, model: str) -> float | None:
        return pct[model].get(text)

    # ---- per concept
    layer_rows: dict[str, list[dict]] = {g: [] for g in GROUPS}
    text_rows: list[dict] = []
    combine_rows: list[dict] = []
    concept_rows: list[dict] = []
    by_id: dict[str, dict] = {}
    for p in predictions:
        feats = score_features(p.features, ref)
        groups = group_scores(feats)
        for g in GROUPS:
            layer_rows[g].append(
                {
                    "concept_id": p.concept_id,
                    "features": [_feature_json(f) for f in feats if f.group == g],
                    "group_score": groups[g],
                    "available": groups[g] is not None,
                }
            )
        t1, t2 = (
            text_of(concept_text[p.concept_id], "t1"),
            text_of(concept_text[p.concept_id], "t2"),
        )
        text_signal = _mean([t1, t2])
        signals = {**groups, "text": text_signal}
        score = combiner.score(signals)
        confidence, parts = evidence_confidence(
            signals, t1, t2, provisional=ref.provisional, graph_kind=graph_kind
        )
        label = combiner.label(score) or "Medium"
        reasons = concept_reasons(feats, text_signal)
        text_rows.append(
            {"concept_id": p.concept_id, "t1": t1, "t2": t2, "agreement": _agreement(t1, t2)}
        )
        combine_rows.append(
            {
                "concept_id": p.concept_id,
                "signals": signals,
                "contributions": combiner.contributions(signals),
                "raw": combiner.raw(signals),
                "score": score,
                "label": label,
                "confidence": confidence,
                "confidence_parts": parts,
            }
        )
        row = {
            "concept_id": p.concept_id,
            "concept_label": p.label,
            "slide_indexes": list(p.slide_indexes),
            "score": score,
            "label": label,
            "confidence": confidence,
            "reasons": reasons,
            "signals": signals,
        }
        concept_rows.append(row)
        by_id[p.concept_id] = row

    # ---- labels relative to this lecture (the hardest parts of THIS lecture)
    concept_cut = relative_labels([(c["concept_id"], c["score"]) for c in concept_rows])
    for rows in (concept_rows, combine_rows):
        for r in rows:
            r["label"] = concept_cut.labels.get(r["concept_id"], "Medium")

    # ---- per slide
    slide_rows_text = [
        {
            "index": s.index,
            "t1": text_of(slide_text[s.index], "t1"),
            "t2": text_of(slide_text[s.index], "t2"),
        }
        for s in slides
    ]
    on_slide: dict[int, list[str]] = {}
    for c in concept_rows:
        for i in c["slide_indexes"]:
            on_slide.setdefault(i, []).append(c["concept_id"])
    slide_scores = build_slide_scores(
        [s.index for s in slides],
        on_slide,
        by_id,
        {r["index"]: _mean([r["t1"], r["t2"]]) for r in slide_rows_text},
        combiner.low,
        combiner.high,
    )
    slide_cut = relative_labels([(s["index"], s["score"]) for s in slide_scores])
    for s in slide_scores:
        s["label"] = slide_cut.labels.get(s["index"], "Medium")
    hotspots = build_hotspots(concept_rows, slide_scores, top_n)
    for h in hotspots:
        cut = concept_cut if h["kind"] == "concept" else slide_cut
        key = h.get("concept_id") if h["kind"] == "concept" else h.get("slide_index")
        h["label"] = cut.labels.get(key, h["label"])

    if not predictions:
        warnings.append("No concepts were found in this lecture")
    status = "provisional" if ref.provisional else "established"
    g = ctx.response.graph
    prerequisite_edges = [e for e in g.edges if e.relation_type == "prerequisite"]
    logger.info(
        "analyzed concepts=%d slides=%d hotspots=%d", len(predictions), len(slides), len(hotspots)
    )
    return {
        "meta": {
            "filename": response.filename,
            "file_type": response.file_type,
            "slide_count": response.slide_count,
            "concept_count": len(predictions),
            "pipeline_version": response.pipeline_version,
            "score_version": SCORE_VERSION,
            "reference": {
                "n_decks": ref.n_decks,
                "n_concepts": ref.n_concepts,
                "provisional": ref.provisional,
            },
            "graph": {
                "kind": graph_kind,
                "nodes": [
                    {"id": n.id, "label": n.label, "frequency": n.frequency} for n in g.nodes
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "confidence": e.confidence,
                        "source_type": e.source_type,
                    }
                    for e in prerequisite_edges
                ],
            },
            "semantic": semantic_status,
            "text_models": {"t1": artifact.t1 is not None, "t2": artifact.t2 is not None},
        },
        "layers": {
            "parse": {
                "slides": [
                    {"index": s.index, "title": s.title or "", "words": len(s.text.split())}
                    for s in slides
                ],
                "concepts": [
                    {
                        "id": c["concept_id"],
                        "label": c["concept_label"],
                        "slide_indexes": c["slide_indexes"],
                    }
                    for c in concept_rows
                ],
            },
            **{
                g: {
                    "group": GROUP_TITLES[g],
                    "direction": GROUP_DIRECTION[g],
                    "feature_directions": {
                        n: d for n, d in DIRECTIONS.items() if FEATURE_GROUP[n] == g
                    },
                    "per_concept": layer_rows[g],
                    "summary": {
                        "available_share": (
                            sum(1 for r in layer_rows[g] if r["available"]) / len(layer_rows[g])
                            if layer_rows[g]
                            else 0.0
                        )
                    },
                }
                for g in GROUPS
            },
            "text": {
                "models": {
                    "t1": MODEL_TEXT["t1"] + ("" if artifact.t1 else " (unavailable)"),
                    "t2": MODEL_TEXT["t2"] + ("" if artifact.t2 else " (unavailable)"),
                },
                "per_concept": text_rows,
                "per_slide": slide_rows_text,
            },
            "combine": {
                "weights": dict(combiner.weights),
                "signals": list(SIGNALS),
                "cutoffs": {"low": concept_cut.low, "high": concept_cut.high},
                "labelling": LABELLING_NOTE,
                "reference_cutoffs": {"low": combiner.low, "high": combiner.high},
                "per_concept": combine_rows,
            },
        },
        "final": {"hotspots": hotspots, "slide_scores": slide_scores, "concepts": concept_rows},
        "warnings": warnings,
        "disclaimer": DISCLAIMER.format(n=ref.n_decks, status=status),
    }
