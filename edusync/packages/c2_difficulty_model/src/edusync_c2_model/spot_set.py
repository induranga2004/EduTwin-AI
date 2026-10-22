"""The spot-check set: snippets from real decks rated Low/Medium/High, used ONLY to evaluate.

``sample_snippets`` draws a seeded sample from cached feature rows (round-robin over lectures, so
every lecture is represented) and ``write_sheet`` writes the rating CSV. The sheet holds
copyrighted lecture text and belongs in the git-ignored ``data/real/`` folder. ``evaluate_spot_set``
compares the ratings with model scores (Spearman always, QWK when score cut-offs are given) and
reports n and bootstrap-by-lecture CIs.
"""

import csv
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from edusync_c2.difficulty.models import LABEL_ORDER
from edusync_c2_model.examples import LABEL_ID, FeatureRow
from edusync_c2_model.metrics import bootstrap_ci, confusion_matrix, metrics_from_confusion
from edusync_c2_model.public_data.targets import to_label_ids
from edusync_c2_model.regression_metrics import bootstrap_by_group, spearman

SHEET_COLUMNS = (
    "snippet_id",
    "lecture_id",
    "slide_index",
    "concept",
    "text",
    "rater_id",
    "rating",
    "notes",
)
DEFAULT_N = 100
MAX_TEXT_CHARS = 800
_RATINGS = {name.lower(): LABEL_ID[name] for name in LABEL_ORDER}


class SpotSetError(ValueError):
    """The sheet is malformed or cannot be evaluated."""


def _shorten(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= MAX_TEXT_CHARS else flat[: MAX_TEXT_CHARS - 1].rstrip() + "…"


def sample_snippets(
    rows: Sequence[FeatureRow],
    *,
    n: int = DEFAULT_N,
    seed: int = 0,
    slide_index: Mapping[tuple[str, str], int] | None = None,
) -> list[dict[str, str]]:
    """Up to ``n`` sheet rows (rater/rating/notes blank), deterministic for rows and seed.

    Rows with no text are skipped. Lectures are visited round-robin in sorted order and a snippet
    is drawn at random from each in turn, so no lecture dominates. Ids follow the sorted
    (lecture, concept) order, so they do not depend on the draw order.
    """
    by_lecture: dict[str, list[FeatureRow]] = {}
    for row in sorted(rows, key=lambda r: r.key):
        if row.text.strip():
            by_lecture.setdefault(row.lecture_id, []).append(row)
    rng = random.Random(seed)
    for pool in by_lecture.values():
        rng.shuffle(pool)
    chosen: list[FeatureRow] = []
    while len(chosen) < n and any(by_lecture.values()):
        for lecture in sorted(by_lecture):
            if by_lecture[lecture] and len(chosen) < n:
                chosen.append(by_lecture[lecture].pop())
    slides = slide_index or {}
    out = []
    for i, row in enumerate(sorted(chosen, key=lambda r: r.key), start=1):
        index = slides.get(row.key)
        out.append(
            {
                "snippet_id": f"S{i:03d}",
                "lecture_id": row.lecture_id,
                "slide_index": "" if index is None else str(index),
                "concept": row.concept_label,
                "text": _shorten(row.text),
                "rater_id": "",
                "rating": "",
                "notes": "",
            }
        )
    return out


def write_sheet(path: Path, snippets: Sequence[Mapping[str, str]]) -> None:
    """Write the rating sheet as UTF-8 CSV; refuses to overwrite an existing sheet."""
    try:
        with open(path, "x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
            writer.writeheader()
            writer.writerows(snippets)
    except FileExistsError as exc:
        raise SpotSetError(f"{path} already exists; ratings are never overwritten") from exc


def read_ratings(sheet_csv: Path) -> dict[str, dict]:
    """``{snippet_id: {"lecture_id", "ratings": [class ids]}}`` for snippets with >= 1 rating."""
    with open(sheet_csv, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or {"snippet_id", "lecture_id", "rating"} - set(
            reader.fieldnames
        ):
            raise SpotSetError("sheet needs the columns snippet_id, lecture_id and rating")
        out: dict[str, dict] = {}
        for line, row in enumerate(reader, start=2):
            rating = (row["rating"] or "").strip().lower()
            if not rating:
                continue
            if rating not in _RATINGS:
                raise SpotSetError(f"line {line}: rating must be Low, Medium or High")
            entry = out.setdefault(
                row["snippet_id"], {"lecture_id": row["lecture_id"], "ratings": []}
            )
            entry["ratings"].append(_RATINGS[rating])
    return out


def evaluate_spot_set(
    sheet_csv: Path,
    scores: Mapping[str, float],
    cutoffs: dict | None = None,
    *,
    seed: int = 0,
    n_boot: int = 1000,
) -> dict:
    """Agreement of model ``scores`` (by snippet_id) with the human ratings.

    The human value of a snippet is the mean of its ratings (Low=0, Medium=1, High=2). Spearman
    uses that mean; QWK uses the rounded mean against the class that ``cutoffs``
    (``{"low_below", "high_from"}``) give the score, and is ``None`` without cutoffs.
    """
    ratings = read_ratings(sheet_csv)
    ids = sorted(i for i in ratings if i in scores)
    result: dict = {
        "n_rated": len(ratings),
        "n": len(ids),
        "n_unscored": len(ratings) - len(ids),
        "n_lectures": len({ratings[i]["lecture_id"] for i in ids}),
        "n_boot": n_boot,
        "seed": seed,
        "spearman": None,
        "qwk": None,
        "ci": {"spearman": None, "qwk": None},
    }
    if not ids:
        return result
    human = [sum(ratings[i]["ratings"]) / len(ratings[i]["ratings"]) for i in ids]
    model = [float(scores[i]) for i in ids]
    lectures = [ratings[i]["lecture_id"] for i in ids]
    result["spearman"] = spearman(human, model)
    result["ci"]["spearman"] = bootstrap_by_group(
        human, model, lectures, {"spearman": spearman}, n_boot=n_boot, seed=seed
    )["spearman"]
    if cutoffs is not None:
        truth = [min(2, int(h + 0.5)) for h in human]
        predicted = to_label_ids(model, cutoffs)
        result["qwk"] = metrics_from_confusion(confusion_matrix(truth, predicted))["qwk"]
        result["ci"]["qwk"] = bootstrap_ci(truth, predicted, lectures, n_boot=n_boot, seed=seed)[
            "qwk"
        ]
    return result
