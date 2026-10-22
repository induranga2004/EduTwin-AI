"""Train and evaluate the T2 numpy text model (frozen MiniLM embedding + simple features -> ridge).

Reads the sealed CLEAR / OneStopEnglish splits, embeds with the frozen numpy MiniLM (cached on disk
in ``cache/embeddings``), fits every variant on the TRAIN part only (CLEAR-only, OneStop-only and
joint; first-128-tokens vs chunked embedding), evaluates val and test of both corpora next to the
text baselines, writes the best variant (chosen on mean VAL Spearman, never on test) to
``runs/t2/`` and a markdown report to ``runs/t2_report.md``. General English, not technical
lectures: the numbers say nothing about accuracy on lecture slides.
"""

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

from edusync_c2_model.baselines_text import run_baselines
from edusync_c2_model.metrics import confusion_matrix, metrics_from_confusion
from edusync_c2_model.models.embeddings import CachedEmbedder, Embedder, encoder_embedder
from edusync_c2_model.public_data.clear import load_clear
from edusync_c2_model.public_data.licences import SOURCES
from edusync_c2_model.public_data.onestop import load_onestop
from edusync_c2_model.public_data.splits import (
    make_group_split,
    partition,
    read_split,
    write_split_once,
)
from edusync_c2_model.public_data.targets import fit_targets, to_difficulty, to_label_ids
from edusync_c2_model.regression_metrics import evaluate_regressor
from edusync_c2_model.t2 import DEFAULT_CHUNK_WORDS, LICENCE_NOTE, T2Model

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    PACKAGE_ROOT.parent / "blueprint_model/artifacts/base/sentence-transformers--all-MiniLM-L6-v2"
)
LOADERS = {"clear": (load_clear, True), "onestop": (load_onestop, False)}
CAVEAT = (
    "Caveat: general English with human readability judgements (CLEAR passages, OneStopEnglish "
    "news articles), not technical lectures. Nothing here measures accuracy on lecture slides."
)


def load_parts(key: str, data_dir: Path, seed: int) -> dict | None:
    """Train/val/test examples of one corpus from its sealed split (None if file absent)."""
    source = next(s for s in SOURCES if s.key == key)
    path = data_dir / source.filename
    if not path.is_file():
        print(f"{key}: {path} not found, skipped")
        return None
    examples = LOADERS[key][0](path)
    split_path = data_dir / "splits" / f"{key}_split.json"
    if split_path.exists():
        split = read_split(split_path)
    else:
        split = make_group_split({e.group_id for e in examples}, seed=seed, source=key)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        write_split_once(split_path, split)
    return partition(examples, split)


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _ci(ci: dict | None) -> str:
    return "n/a" if ci is None else f"[{_num(ci['low'])}, {_num(ci['high'])}]"


def _row(name: str, source: str, part: str, ev: dict) -> dict:
    return {
        "name": name,
        "source": source,
        "part": part,
        "n": ev["n"],
        "spearman": ev["spearman"],
        "spearman_ci": ev["ci"]["spearman"],
        "rmse": ev["rmse"],
        "rmse_ci": ev["ci"]["rmse"],
    }


def fit_variant(
    fit_set: str,
    chunk_words: int | None,
    parts_by_source: dict[str, dict],
    scalers: dict[str, dict],
    embed: Embedder,
    encoder_sha: str,
    seed: int,
) -> T2Model:
    texts: list[str] = []
    targets: list[float] = []
    groups: list[str] = []
    for key in (k for k in parts_by_source if fit_set in (k, "joint")):
        train = parts_by_source[key]["train"]
        texts += [e.text for e in train]
        targets += list(to_difficulty([e.target for e in train], scalers[key]))
        groups += [f"{key}:{e.group_id}" for e in train]
    return T2Model(embed, encoder_sha, chunk_words).fit(texts, targets, seed, groups)


def variant_name(fit_set: str, chunk_words: int | None) -> str:
    kind = "first128" if chunk_words is None else f"chunked{chunk_words}"
    return f"t2_{fit_set}_{kind}"


def tercile_check(model: T2Model, parts: dict, outcome_targets: dict) -> dict:
    """Low/Medium/High accuracy and QWK on test using TRAIN-only terciles of the same corpus."""
    test = parts["test"]
    truth = to_difficulty([e.target for e in test], outcome_targets["scaler"])
    predicted, _ = model.predict([e.text for e in test])
    cutoffs = outcome_targets["cutoffs"]
    cm = confusion_matrix(to_label_ids(truth, cutoffs), to_label_ids(predicted, cutoffs))
    return metrics_from_confusion(cm)


def run(
    data_dir: Path,
    runs_dir: Path,
    *,
    embedder: tuple[Embedder, str],
    cache_dir: Path | None,
    seed: int = 42,
    n_boot: int = 1000,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    log: Callable[[str], None] = print,
) -> dict | None:
    """Train every variant, write the artifact and the report; None when no corpus is present."""
    started = time.perf_counter()
    parts_by_source = {
        key: parts for key in LOADERS if (parts := load_parts(key, data_dir, seed)) is not None
    }
    if not parts_by_source:
        return None
    raw_embed, sha = embedder
    cached = CachedEmbedder(raw_embed, sha, cache_dir)
    scalers, rows, models, terciles = {}, [], {}, {}
    baseline_rows = []
    targets = {}
    for key, parts in parts_by_source.items():
        targets[key] = fit_targets(
            [e.target for e in parts["train"]], higher_is_easier=LOADERS[key][1]
        )
        scalers[key] = targets[key]["scaler"]
        outcome = run_baselines(parts, higher_is_easier=LOADERS[key][1], seed=seed, n_boot=n_boot)
        for name in ("fk_grade", "ridge_all"):
            for part, ev in outcome["results"][name].items():
                baseline_rows.append(_row(f"baseline_{name}", key, part, ev))
    fit_sets = [*parts_by_source, *(["joint"] if len(parts_by_source) > 1 else [])]
    for fit_set in fit_sets:
        for chunk in (None, chunk_words):
            name = variant_name(fit_set, chunk)
            model = fit_variant(fit_set, chunk, parts_by_source, scalers, cached, sha, seed)
            models[name] = model
            log(f"fitted {name}: alpha={model.alpha} cv_rmse={model.cv_rmse:.3f}")
            for key, parts in parts_by_source.items():
                for part in ("val", "test"):
                    examples = parts[part]
                    truth = to_difficulty([e.target for e in examples], scalers[key])
                    predicted, _ = model.predict([e.text for e in examples])
                    ev = evaluate_regressor(
                        truth, predicted, [e.group_id for e in examples], seed, n_boot=n_boot
                    )
                    rows.append(_row(name, key, part, ev))
            if "clear" in parts_by_source:
                terciles[name] = tercile_check(model, parts_by_source["clear"], targets["clear"])
    best = _choose_best(rows, models)
    model_dir = runs_dir / "t2"
    npz_path, config_path = models[best].save(model_dir)
    summary = {
        "rows": rows,
        "baseline_rows": baseline_rows,
        "terciles": terciles,
        "best": best,
        "sizes": {k: {p: len(v[p]) for p in v} for k, v in parts_by_source.items()},
        "seconds": time.perf_counter() - started,
        "embeddings_computed": cached.computed,
        "seed": seed,
        "n_boot": n_boot,
        "alpha": {n: m.alpha for n, m in models.items()},
    }
    (model_dir / "model_card.md").write_text(
        render_model_card(summary, models[best]), encoding="utf-8", newline="\n"
    )
    (runs_dir / "t2_report.md").write_text(render_report(summary), encoding="utf-8", newline="\n")
    summary["artifact"] = [str(npz_path), str(config_path), str(model_dir / "model_card.md")]
    return summary


def _choose_best(rows: list[dict], models: dict) -> str:
    """Variant with the highest mean VAL Spearman over the corpora it was evaluated on."""
    scores: dict[str, list[float]] = {}
    for r in rows:
        if r["part"] == "val" and r["spearman"] is not None:
            scores.setdefault(r["name"], []).append(r["spearman"])

    def mean_val(name: str) -> float:
        values = scores.get(name)
        return sum(values) / len(values) if values else float("-inf")

    return min(models, key=lambda n: (-mean_val(n), n))


def _table(rows: list[dict]) -> list[str]:
    lines = [
        "| Model | Corpus | Part | n | Spearman | 95% CI | RMSE | 95% CI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['source']} | {r['part']} | {r['n']} | {_num(r['spearman'])} | "
            f"{_ci(r['spearman_ci'])} | {_num(r['rmse'])} | {_ci(r['rmse_ci'])} |"
        )
    return lines


def render_report(summary: dict) -> str:
    lines = [
        "# T2 numpy text model: frozen MiniLM + simple features -> ridge",
        "",
        CAVEAT,
        "",
        "Target: z-scored difficulty (higher = harder), scaled on each corpus' TRAIN part. In the "
        "joint fit each corpus keeps its own z-scale. RMSE of OneStop-only / CLEAR-only models "
        "on the other corpus is not comparable (different scale); use Spearman there. "
        "Joint = CLEAR + OneStop train rows with per-corpus z-targets. The auxiliary "
        "level-to-z ordinal head was not implemented (not simple enough to be honest about).",
        "",
        f"Part sizes: {summary['sizes']}. Seed {summary['seed']}, bootstrap {summary['n_boot']} "
        f"(by passage/article). Ridge alpha per variant: {summary['alpha']}.",
        f"Chosen artifact (highest mean VAL Spearman, test never used): `{summary['best']}`.",
        f"Wall time {summary['seconds']:.0f} s; {summary['embeddings_computed']} new embeddings.",
        "",
        "## Results",
        "",
        *_table(summary["baseline_rows"] + summary["rows"]),
        "",
        "## Low/Medium/High mapping check (CLEAR test, TRAIN-only terciles)",
        "",
        "| Model | n | Accuracy | Adjacent accuracy | QWK |",
        "|---|---|---|---|---|",
    ]
    for name, m in summary["terciles"].items():
        lines.append(
            f"| {name} | {m['n']} | {_num(m['accuracy'])} | {_num(m['adjacent_accuracy'])} | "
            f"{_num(m['qwk'])} |"
        )
    lines += ["", LICENCE_NOTE, ""]
    return "\n".join(lines)


def render_model_card(summary: dict, model: T2Model) -> str:
    best = summary["best"]
    test = [r for r in summary["rows"] if r["name"] == best and r["part"] == "test"]
    metrics = "\n".join(
        f"- {r['source']} test (n={r['n']}): Spearman {_num(r['spearman'])} "
        f"{_ci(r['spearman_ci'])}, RMSE {_num(r['rmse'])}"
        for r in test
    )
    return "\n".join(
        [
            "# T2 text difficulty model (DRAFT)",
            "",
            f"Variant `{best}`: frozen all-MiniLM-L6-v2 embedding (sha {model.encoder_sha}, "
            f"chunk_words={model.chunk_words}) + 5 simple text features -> closed-form ridge "
            f"(alpha {model.alpha}). Output: z-scored difficulty and an uncertainty proxy "
            "(not a probability).",
            "",
            "## Intended use",
            "Cheap second opinion on how hard a passage reads. Research prototype.",
            "",
            "## Training data",
            LICENCE_NOTE,
            f"Fitted on the TRAIN parts only (n={model.n_train}), seed {model.seed}.",
            "",
            "## Held-out results",
            metrics,
            "",
            "## Limitations",
            "General English readability, not technical lectures: domain shift is expected and "
            "unmeasured here. Do not redistribute the weights commercially (CLEAR is NC-SA).",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE_ROOT / "data" / "public")
    parser.add_argument("--runs-dir", type=Path, default=PACKAGE_ROOT / "runs")
    parser.add_argument("--cache-dir", type=Path, default=PACKAGE_ROOT / "cache" / "embeddings")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    args = parser.parse_args(argv)
    try:
        summary = run(
            args.data_dir,
            args.runs_dir,
            embedder=encoder_embedder(args.model_dir),
            cache_dir=args.cache_dir,
            seed=args.seed,
            n_boot=args.n_boot,
            chunk_words=args.chunk_words,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if summary is None:
        print("no corpus found; nothing trained")
        return 0
    print(f"report: {args.runs_dir / 't2_report.md'}\nartifact: {summary['artifact']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
