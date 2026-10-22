"""End-to-end SYNTHETIC demo: one dataset per expert, a combiner dataset, baselines, one report.

SYNTHETIC DATA: labels are rule-plus-noise, so every score here tests the plumbing and is never
evidence of accuracy. Expert A trains on the 'language' dataset, B on 'prerequisites', C on
'explanation', D on 'density' and the combiner on the 'combiner' dataset; each dataset is split
80/10/10 by lecture. Lecture ids are disjoint across datasets (asserted), so the combiner sees
genuinely out-of-sample expert probabilities. The validation parts pick the L2 strength from a
small grid by combiner-validation QWK; each test part is read once, at the end.

Exit codes: 0 ok, 2 bad arguments, missing model weights or a dataset error.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # before numpy: fixed BLAS threading

from edusync_c2_model.baselines import (
    BaselineResult,
    compare,
    human_ceiling,
    run_logistic_baseline,
    run_majority_baseline,
    run_rule_baseline,
    score_predictions,
)
from edusync_c2_model.extract import ExtractError
from edusync_c2_model.metrics import evaluate
from edusync_c2_model.models.embeddings import (
    CachedEmbedder,
    Embedder,
    encoder_embedder,
)
from edusync_c2_model.models.ensemble import (
    EnsembleParams,
    GroupEnsemble,
    split_by_lecture,
)
from edusync_c2_model.models.experts import EXPERT_SPECS
from edusync_c2_model.report import SYNTHETIC_BANNER, render_markdown
from edusync_c2_model.synthetic import (
    DEFAULT_BASE_SEED,
    DEFAULT_DATA_DIR,
    DEFAULT_ROWS,
    SyntheticError,
    build_expert_datasets,
    load_dataset,
)

PACKAGE = DEFAULT_DATA_DIR.parents[1]
DEFAULT_MODEL_DIR = (
    PACKAGE.parent
    / "blueprint_model"
    / "artifacts"
    / "base"
    / "sentence-transformers--all-MiniLM-L6-v2"
)
DATASET_OF = {"A": "language", "B": "prerequisites", "C": "explanation", "D": "density"}
L2_GRID = (0.001, 0.01, 0.1)


def _qwk(ensemble: GroupEnsemble, examples: list) -> float:
    pred = [int(p) for p in ensemble.predict(examples)]
    stats = evaluate(
        [e.label_id for e in examples], pred, [e.lecture_id for e in examples], n_boot=0
    )
    return stats["qwk"] if stats["qwk"] is not None else float("-inf")


def _ensemble_result(
    name: str, ensemble: GroupEnsemble, examples: list, n_boot: int
) -> BaselineResult:
    return score_predictions(
        name, examples, [int(p) for p in ensemble.predict(examples)], n_boot=n_boot
    )


def _expert_result(
    ensemble: GroupEnsemble, letter: str, examples: list, n_boot: int
) -> BaselineResult:
    j = [s.letter for s in EXPERT_SPECS].index(letter)
    detail = ensemble.predict_detail(examples)
    preds = [
        int(detail.expert_probs[i, j].argmax()) if detail.votes[i, j] else None
        for i in range(len(examples))
    ]
    return score_predictions(f"expert {letter} alone", examples, preds, n_boot=n_boot)


def run(
    data_dir: Path,
    cache_dir: Path,
    embed: Embedder,
    encoder_id: str,
    *,
    rows: int,
    seed: int,
    n_boot: int,
) -> tuple[str, str]:
    """Train and evaluate everything; returns (console table, markdown report)."""
    started = time.perf_counter()
    datasets = build_expert_datasets(data_dir, rows, seed)
    parts, ceilings = {}, {}
    for letter in (*DATASET_OF, "M"):
        paths = datasets[letter]
        examples, _, _ = load_dataset(paths, cache_dir / paths.name)
        parts[letter] = split_by_lecture(examples, seed)
        ceilings[letter] = human_ceiling(paths.labels)
    expert_train = {letter: parts[letter][0] for letter in DATASET_OF}
    combiner_train, combiner_val, _ = parts["M"]

    best = None
    for l2 in L2_GRID:
        params = EnsembleParams(l2=l2, combiner_l2=l2)
        candidate = GroupEnsemble.fit_from_example_sets(
            expert_train, combiner_train, seed, embed=embed, encoder_id=encoder_id, params=params
        )
        score = _qwk(candidate, combiner_val)
        print(f"  l2={l2}: combiner validation QWK {score:.3f}", flush=True)
        if best is None or score > best[0]:
            best = (score, l2, candidate)
    assert best is not None
    _, chosen_l2, ensemble = best

    sections, console = [], []
    for letter in ("M", *DATASET_OF):
        train, _, test = parts[letter]
        name = "combiner" if letter == "M" else DATASET_OF[letter]
        results = [
            run_rule_baseline(test, n_boot=n_boot),
            run_majority_baseline(train, test, n_boot=n_boot),
            run_logistic_baseline(train, test, l2=chosen_l2, n_boot=n_boot),
        ]
        if letter != "M":
            results.append(_expert_result(ensemble, letter, test, n_boot))
        results.append(_ensemble_result("group ensemble", ensemble, test, n_boot))
        lectures = len({e.lecture_id for e in test})
        title = f"{name} dataset: test split ({len(test)} concepts, {lectures} lectures)"
        table = compare(results, ceilings[letter])
        sections.append(f"## {title}\n\n{table}")
        console.append(f"{title}\n{table}")
        if letter == "M":
            final = results[-1].metrics
    elapsed = time.perf_counter() - started
    header = (
        f"**{SYNTHETIC_BANNER}**\n\n"
        f"Runtime {elapsed:.0f}s. Seed {seed}, {rows} rows per dataset, split 80/10/10 by lecture, "
        f"L2 {chosen_l2} chosen on combiner validation QWK from {list(L2_GRID)}. Labels are "
        "rule-plus-noise; the human-human row is the synthetic rater agreement.\n"
    )
    body = "\n".join(sections)
    detail = render_markdown(final, "Group ensemble on the combiner test split")
    report = f"# C2 group ensemble: synthetic demo\n\n{header}\n{body}\n{detail}"
    return "\n".join(console) + f"\nruntime {elapsed:.0f}s, L2 {chosen_l2}", report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="MiniLM folder")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache", type=Path, default=PACKAGE / "cache")
    parser.add_argument("--out", type=Path, default=PACKAGE / "runs")
    parser.add_argument("--rows-per-expert", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments, 0 on --help
        return int(exc.code or 0)
    try:
        embed, sha = encoder_embedder(args.model_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    cached = CachedEmbedder(embed, sha, args.cache / "embeddings")
    try:
        console, report = run(
            args.data,
            args.cache / "synthetic",
            cached,
            sha,
            rows=args.rows_per_expert,
            seed=args.seed,
            n_boot=args.bootstrap,
        )
    except (SyntheticError, ExtractError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "train_synthetic_demo.md"
    target.write_text(report, encoding="utf-8", newline="\n")
    print(f"SYNTHETIC DATA: plumbing only, never accuracy\n{console}\nreport: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
