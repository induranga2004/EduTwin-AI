"""Render an ``evaluate`` result as markdown; synthetic data always carries a banner."""

from collections.abc import Iterable

from edusync_c2_model.examples import Example
from edusync_c2_model.labels import LABEL_ORDER
from edusync_c2_model.metrics import METRIC_NAMES

SYNTHETIC_BANNER = "SYNTHETIC DATA: results test the plumbing only and are not accuracy"


def any_synthetic(examples: Iterable[Example]) -> bool:
    return any(e.synthetic for e in examples)


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(metrics: dict, title: str = "C2 difficulty evaluation") -> str:
    """Markdown for one ``evaluate`` dict (banner first when ``metrics['synthetic']``)."""
    lines = [f"# {title}", ""]
    if metrics.get("synthetic"):
        lines += [f"**{SYNTHETIC_BANNER}**", ""]
    lines += [
        f"Concepts: {metrics['n']}   Lectures: {metrics['n_lectures']}   "
        f"Bootstrap resamples: {metrics['n_boot']} (by lecture)",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
    ]
    for name in METRIC_NAMES:
        ci = metrics["ci"].get(name)
        interval = "n/a" if ci is None else f"{_num(ci['low'])} to {_num(ci['high'])}"
        lines.append(f"| {name} | {_num(metrics[name])} | {interval} |")
    lines += ["", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c in metrics["per_class"]:
        lines.append(
            f"| {c['label']} | {_num(c['precision'])} | {_num(c['recall'])} | "
            f"{_num(c['f1'])} | {c['support']} |"
        )
    lines += ["", "Confusion matrix (rows = true, columns = predicted):", ""]
    lines += ["| true \\ pred | " + " | ".join(LABEL_ORDER) + " |", "|---|---|---|---|"]
    for name, row in zip(LABEL_ORDER, metrics["confusion_matrix"], strict=True):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"
