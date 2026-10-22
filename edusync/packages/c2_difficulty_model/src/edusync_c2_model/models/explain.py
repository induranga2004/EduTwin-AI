"""Per-example explanation strings: each expert's top class and its strongest standardised features.

Only feature names, numbers and class names appear: never slide text or the concept label.
"""

from collections.abc import Sequence

import numpy as np

from edusync_c2_model.examples import Example
from edusync_c2_model.labels import LABEL_ORDER
from edusync_c2_model.models.ensemble import GroupEnsemble
from edusync_c2_model.models.experts import to_arrays


def explain(ensemble: GroupEnsemble, examples: Sequence[Example], top: int = 3) -> list[list[str]]:
    """One list of lines per example (same order): per-expert vote, then the combined result."""
    if not examples:
        return []
    detail = ensemble.predict_detail(examples)
    arrays = to_arrays(examples)
    lines: list[list[str]] = [[] for _ in examples]
    for j, expert in enumerate(ensemble.experts):
        picked = np.argmax(detail.expert_probs[:, j], axis=1)
        contrib = expert.contributions(arrays, picked)
        shown = ~arrays.missing[:, expert.columns]
        for i in range(len(examples)):
            name = expert.spec.name
            if not detail.votes[i, j]:
                lines[i].append(f"{name}: no vote (nothing to look at)")
                continue
            cls = int(picked[i])
            head = f"{name}: {LABEL_ORDER[cls]} (p={detail.expert_probs[i, j, cls]:.2f})"
            order = [k for k in np.argsort(-np.abs(contrib[i]), kind="stable") if shown[i, k]]
            parts = [f"{expert.feature_names[k]} {contrib[i, k]:+.2f}" for k in order[:top]]
            lines[i].append(head + ("; top: " + ", ".join(parts) if parts else "; text only"))
    for i in range(len(examples)):
        if detail.votes[i].any():
            cls = int(np.argmax(detail.probs[i]))
            lines[i].append(f"combined: {LABEL_ORDER[cls]} (p={detail.probs[i, cls]:.2f})")
        else:
            lines[i].append("combined: no expert could vote")
    return lines
