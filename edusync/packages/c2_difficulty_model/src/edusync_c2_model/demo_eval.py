"""Compare the ILLUSTRATIVE demo ratings with the model's slide scores (ids and aggregates only).

``evaluate_demo`` maps each rated snippet to its slide in the analyze output
(``final.slide_scores[{index, score, label}]``) and reports Spearman with a bootstrap-by-slide CI,
3-class QWK, exact and adjacent agreement and the confusion matrix. The demo ratings are not
ground truth, so these numbers show that the path works, not that the model is valid. No slide
text is read or returned.
"""

import csv
import json
from pathlib import Path

from edusync_c2.difficulty.models import LABEL_ORDER
from edusync_c2_model.metrics import confusion_matrix, metrics_from_confusion
from edusync_c2_model.regression_metrics import bootstrap_by_group, spearman

_LABEL_ID = {name.lower(): i for i, name in enumerate(LABEL_ORDER)}


class DemoEvalError(ValueError):
    """The ratings file or analysis JSON cannot be evaluated."""


def _read_ratings(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    if not rows or {"snippet_id", "slide_index", "rating"} - set(rows[0]):
        raise DemoEvalError("ratings need the columns snippet_id, slide_index and rating")
    for row in rows:
        if (row["rating"] or "").strip().lower() not in _LABEL_ID:
            raise DemoEvalError(f"{row['snippet_id']}: rating must be Low, Medium or High")
    return rows


def _slide_scores(analysis_json: Path) -> dict[int, dict]:
    try:
        payload = json.loads(Path(analysis_json).read_text(encoding="utf-8"))
        return {int(s["index"]): s for s in payload["final"]["slide_scores"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise DemoEvalError(
            "analysis JSON needs final.slide_scores[{index, score, label}]"
        ) from exc


def evaluate_demo(
    ratings_csv: Path, analysis_json: Path, *, seed: int = 0, n_boot: int = 1000
) -> dict:
    """Agreement of demo ratings with model slide labels and scores; unmatched snippets listed."""
    ratings = _read_ratings(Path(ratings_csv))
    scores = _slide_scores(Path(analysis_json))
    per_snippet, missing = [], []
    for row in ratings:
        slide = int(row["slide_index"]) if (row["slide_index"] or "").strip().isdigit() else None
        hit = scores.get(slide) if slide is not None else None
        if hit is None or str(hit["label"]).lower() not in _LABEL_ID:
            missing.append(row["snippet_id"])
            continue
        per_snippet.append(
            {
                "id": row["snippet_id"],
                "slide": slide,
                "model_label": hit["label"],
                "demo_label": row["rating"].strip().capitalize(),
                "model_score": float(hit["score"]),
            }
        )
    result: dict = {
        "n": len(per_snippet),
        "n_missing": len(missing),
        "missing_ids": missing,
        "n_boot": n_boot,
        "seed": seed,
        "spearman": None,
        "spearman_ci": None,
        "qwk": None,
        "agreement_rate": None,
        "adjacent_agreement": None,
        "confusion": None,
        "per_snippet": per_snippet,
        "illustrative": True,
    }
    if not per_snippet:
        return result
    truth = [_LABEL_ID[p["demo_label"].lower()] for p in per_snippet]
    pred = [_LABEL_ID[str(p["model_label"]).lower()] for p in per_snippet]
    cm = confusion_matrix(truth, pred)
    n = len(truth)
    result["confusion"] = cm.tolist()
    result["qwk"] = metrics_from_confusion(cm)["qwk"]
    result["agreement_rate"] = sum(t == p for t, p in zip(truth, pred, strict=True)) / n
    result["adjacent_agreement"] = (
        sum(abs(t - p) <= 1 for t, p in zip(truth, pred, strict=True)) / n
    )
    human = [float(t) for t in truth]
    model = [p["model_score"] for p in per_snippet]
    result["spearman"] = spearman(human, model)
    result["spearman_ci"] = bootstrap_by_group(
        human,
        model,
        [str(p["slide"]) for p in per_snippet],
        {"spearman": spearman},
        n_boot=n_boot,
        seed=seed,
    )["spearman"]
    return result
