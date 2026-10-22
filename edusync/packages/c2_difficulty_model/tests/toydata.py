"""Tiny deterministic toy examples and a hash-based fake embedder (no MiniLM needed)."""

import hashlib

import numpy as np

from edusync_c2.difficulty.registry import FEATURE_NAMES, features_of
from edusync_c2.difficulty.schemas import FeatureGroup
from edusync_c2_model.examples import Example
from edusync_c2_model.labels import LABEL_ORDER

N = len(FEATURE_NAMES)


def hash_embed(texts) -> np.ndarray:
    """8-d vectors from sha256 of the text: deterministic, no signal."""
    out = np.zeros((len(texts), 8))
    for i, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        out[i] = np.frombuffer(digest[:8], dtype=np.uint8) / 255.0 - 0.5
    return out


def make_toy(
    n_lectures: int = 12, per_lecture: int = 9, seed: int = 0, prefix: str = "L", gap: float = 3.0
) -> list[Example]:
    """Separable toy data: every feature is ``gap * class + noise``; classes cycle Low/Med/High."""
    rng = np.random.default_rng(seed)
    examples = []
    for lec in range(n_lectures):
        for k in range(per_lecture):
            label_id = (lec + k) % 3
            values = tuple(float(v) for v in gap * label_id + rng.normal(0.0, 0.5, N))
            examples.append(
                Example(
                    lecture_id=f"{prefix}{lec:02d}",
                    course_id=f"C{lec % 3}",
                    concept_id=f"c{k}",
                    concept_label=f"Concept {k}",
                    text=f"{prefix}{lec} concept {k} {LABEL_ORDER[label_id]}",
                    values=values,
                    missing=(0,) * N,
                    statuses=("computed",) * N,
                    label=LABEL_ORDER[label_id],
                    label_id=label_id,
                    n_raters=2,
                    synthetic=True,
                )
            )
    return examples


def mask_group(examples, group: FeatureGroup, *, drop_text: bool = False) -> list[Example]:
    """Copies with every feature of ``group`` missing (and optionally the text emptied)."""
    cols = {FEATURE_NAMES.index(n) for n in features_of(group)}
    out = []
    for e in examples:
        values = tuple(0.0 if i in cols else v for i, v in enumerate(e.values))
        missing = tuple(1 if i in cols else m for i, m in enumerate(e.missing))
        statuses = tuple("unavailable" if i in cols else s for i, s in enumerate(e.statuses))
        out.append(
            Example(
                **{
                    **e.__dict__,
                    "values": values,
                    "missing": missing,
                    "statuses": statuses,
                    "text": "" if drop_text else e.text,
                }
            )
        )
    return out
