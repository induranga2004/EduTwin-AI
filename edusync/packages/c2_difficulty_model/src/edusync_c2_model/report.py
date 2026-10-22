"""Render an ``evaluate`` result as markdown; synthetic data always carries a banner."""

from collections.abc import Iterable

from edusync_c2.difficulty.models import LABEL_ORDER
from edusync_c2_model.examples import Example
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


# ------------------------------------------------------------------------------- T1 reports

T1_CAVEAT = (
    "General English with human readability judgements (CLEAR, CC BY-NC-SA; OneStopEnglish, "
    "CC BY-SA), not technical lectures: these numbers say nothing about accuracy on lecture slides."
)


def _ci(ci: dict | None) -> str:
    return "n/a" if ci is None else f"{_num(ci['low'])} to {_num(ci['high'])}"


def render_quality_table(metrics: dict[str, dict[str, dict]]) -> str:
    """Markdown table of ``{part: {source: evaluate_regressor result}}`` (z-scored difficulty)."""
    lines = [
        "| Split | Source | n | Spearman | 95% CI | RMSE | 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for part in sorted(metrics):
        for source in sorted(metrics[part]):
            m = metrics[part][source]
            lines.append(
                f"| {part} | {source} | {m['n']} | {_num(m['spearman'])} | "
                f"{_ci(m['ci']['spearman'])} | {_num(m['rmse'])} | {_ci(m['ci']['rmse'])} |"
            )
    return "\n".join(lines) + "\n"


def render_quantization_table(comparison: dict) -> str:
    """Markdown fp32 vs INT8 comparison: quality per eval set, size, latency, decision."""
    lines = [
        "| Eval set | n | Spearman fp32 | Spearman INT8 | RMSE fp32 | RMSE INT8 | Max abs diff |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in sorted(comparison["quality"]):
        q = comparison["quality"][name]
        lines.append(
            f"| {name} | {q['n']} | {_num(q['fp32']['spearman'])} | {_num(q['int8']['spearman'])}"
            f" | {_num(q['fp32']['rmse'])} | {_num(q['int8']['rmse'])} | "
            f"{q['max_abs_diff']:.4f} |"
        )
    size = comparison["size_mb"]
    ratio = size["fp32"] / size["int8"] if size["int8"] else 0.0
    lines += [
        "",
        "| Metric | fp32 | INT8 |",
        "|---|---|---|",
        f"| File size (MB) | {size['fp32']:.1f} | {size['int8']:.1f} ({ratio:.2f}x smaller) |",
    ]
    for batch in sorted(comparison["latency_ms_per_text"]):
        lat = comparison["latency_ms_per_text"][batch]
        lines.append(
            f"| Mean latency per text, batch {batch} (ms) | {lat['fp32']:.2f} | {lat['int8']:.2f} |"
        )
    d = comparison["decision"]
    lines += [
        "",
        f"Decision (threshold {comparison['threshold']:.2f}): ship **{d['ship']}**. {d['reason']}.",
    ]
    return "\n".join(lines) + "\n"


def render_t1_report(summary: dict) -> str:
    """The full T1 run report: config, per-epoch history, quality tables, quantization."""
    lines = [
        "# T1 fine-tune report",
        "",
        f"**{T1_CAVEAT}**",
        "",
        f"Best epoch: {summary['best_epoch']} (selected on VAL mean Spearman); "
        f"train rows: {summary['n_train']}.",
        "",
        "| Epoch | Train loss | Val Spearman | Seconds |",
        "|---|---|---|---|",
    ]
    for h in summary["history"]:
        lines.append(
            f"| {h['epoch']} | {h['train_loss']:.4f} | {_num(h['val_spearman'])} | "
            f"{h['seconds']:.0f} |"
        )
    lines += [
        "",
        render_quality_table(summary["metrics"]),
        render_quantization_table(summary["quantization"]),
    ]
    return "\n".join(lines)


def render_model_card(summary: dict, inference_config: dict) -> str:
    """Short model card stored next to the ONNX files."""
    test = summary["metrics"]["test"]
    results = ", ".join(f"{s} test Spearman {_num(m['spearman'])}" for s, m in sorted(test.items()))
    return (
        "# EduSync C2 T1 text difficulty model\n\n"
        "MiniLM-L6 (all-MiniLM-L6-v2) fine-tuned as a regressor: mean pooling, dropout, one linear "
        f"unit. Output `{inference_config['output']}` is z-scored difficulty (higher = harder).\n\n"
        f"- Trained on: {inference_config['trained_on']}\n"
        f"- Max length: {inference_config['max_length']} tokens\n"
        f"- Results: {results}\n"
        f"- Shipped file: {summary['quantization']['decision']['ship']}\n"
        f"- Licence: {inference_config['licence']}\n\n"
        f"Caveat: {T1_CAVEAT}\n"
    )
