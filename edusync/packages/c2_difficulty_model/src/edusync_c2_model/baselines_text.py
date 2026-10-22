"""Cheap text-difficulty baselines for the public corpora: Flesch-Kincaid, vocabulary rarity, ridge.

Every model here is a deterministic function of the text and the TRAIN part only. Features:
Flesch-Kincaid grade (own implementation, heuristic syllable counter), average word length,
average sentence length, rare-word share and mean word-piece fragmentation (both from the
vendored MiniLM vocabulary through ``VocabularyAnalyzer``: the vocabulary has no frequency
ranks, so "rare" means a word that splits into two or more word pieces).
"""

import re
from collections.abc import Sequence

import numpy as np

from edusync_c2.difficulty.features.vocab import VocabularyAnalyzer, get_default_analyzer
from edusync_c2_model.public_data.schema import TextExample
from edusync_c2_model.public_data.targets import fit_targets, to_difficulty
from edusync_c2_model.regression_metrics import evaluate_regressor

FEATURE_NAMES = ("fk_grade", "avg_word_len", "avg_sentence_len", "rare_word_share", "fragmentation")
SINGLE_FEATURE_BASELINES = FEATURE_NAMES
RIDGE_NAME = "ridge_all"
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
SINGLE_ALPHA = 1e-6
CV_FOLDS = 5

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_END = re.compile(r"[.!?]+(?:\s+|$)")
_VOWELS = re.compile(r"[aeiouy]+")
_KEEP_E_ENDINGS = ("le", "ee", "ye", "ue")
_KEEP_ES_ED_ENDINGS = ("ses", "zes", "xes", "ches", "shes", "ges", "ces", "ted", "ded")


def count_syllables(word: str) -> int:
    """Vowel groups minus a silent final ``e``/``es``/``ed``; at least 1 for any word."""
    letters = re.sub(r"[^a-z]", "", word.lower())
    if not letters:
        return 0
    groups = len(_VOWELS.findall(letters))
    silent_e = letters.endswith("e") and not letters.endswith(_KEEP_E_ENDINGS)
    silent_es_ed = letters.endswith(("es", "ed")) and not letters.endswith(_KEEP_ES_ED_ENDINGS)
    if (silent_e or silent_es_ed) and groups > 1:
        groups -= 1
    return max(1, groups)


def split_sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_END.split(text) if part.strip()]


def flesch_kincaid_grade(text: str) -> float:
    """``0.39 words/sentence + 11.8 syllables/word - 15.59``; 0.0 for text with no words."""
    words = _WORD.findall(text)
    if not words:
        return 0.0
    sentences = max(1, len(split_sentences(text)))
    syllables = sum(count_syllables(w) for w in words)
    return 0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59


def text_features(text: str, analyzer: VocabularyAnalyzer) -> np.ndarray:
    """The five baseline features in ``FEATURE_NAMES`` order (zeros for text without words)."""
    words = _WORD.findall(text)
    if not words:
        return np.zeros(len(FEATURE_NAMES))
    stats = analyzer.analyze(text)
    rare, fragmentation = (
        (0.0, 1.0)
        if stats is None
        else (
            stats.unfamiliar_share,
            stats.fragmentation,
        )
    )
    sentences = max(1, len(split_sentences(text)))
    return np.array(
        [
            flesch_kincaid_grade(text),
            sum(len(w) for w in words) / len(words),
            len(words) / sentences,
            rare,
            fragmentation,
        ]
    )


def feature_matrix(texts: Sequence[str], analyzer: VocabularyAnalyzer | None = None) -> np.ndarray:
    analyzer = analyzer or get_default_analyzer()
    if analyzer is None:
        raise RuntimeError("the vocabulary resources could not be loaded")
    if not texts:
        return np.zeros((0, len(FEATURE_NAMES)))
    return np.vstack([text_features(t, analyzer) for t in texts])


class RidgeModel:
    """Standardised ridge regression (intercept unpenalised); fitted on train data only."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.mean = np.zeros(0)
        self.std = np.ones(0)
        self.coef = np.zeros(0)
        self.intercept = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeModel":
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.std = np.where(std < 1e-12, 1.0, std)
        z = (x - self.mean) / self.std
        self.intercept = float(y.mean())
        gram = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.coef = np.linalg.solve(gram, z.T @ (y - self.intercept))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std) @ self.coef + self.intercept


def fold_assignment(groups: Sequence[str], folds: int, seed: int) -> np.ndarray:
    """Fold id per row; whole groups share a fold (seeded, independent of row order)."""
    names = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(names))
    fold_of = {names[g]: i % folds for i, g in enumerate(order)}
    return np.array([fold_of[g] for g in groups])


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    *,
    alphas: Sequence[float] = ALPHA_GRID,
    folds: int = CV_FOLDS,
    seed: int = 0,
) -> float:
    """Alpha with the lowest grouped k-fold RMSE on the training data (ties: the smaller alpha)."""
    assigned = fold_assignment(groups, folds, seed)
    best_alpha, best_error = alphas[0], np.inf
    for alpha in alphas:
        errors = []
        for fold in range(folds):
            held = assigned == fold
            if held.all() or not held.any():
                continue
            model = RidgeModel(alpha).fit(x[~held], y[~held])
            errors.append(((model.predict(x[held]) - y[held]) ** 2).sum())
        error = sum(errors)
        if error < best_error:
            best_alpha, best_error = alpha, error
    return float(best_alpha)


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _interval(ci: dict | None) -> str:
    return "n/a" if ci is None else f"[{_num(ci['low'])}, {_num(ci['high'])}]"


def render_baseline_report(source: str, outcome: dict, caveat: str) -> str:
    """Markdown table (n, Spearman with CI, RMSE with CI) for one ``run_baselines`` result."""
    lines = [
        f"# Text baselines on {source}",
        "",
        caveat,
        "",
        "Target: z-scored difficulty (higher = harder), scaled on the train part only. "
        f"Ridge alpha (grouped CV on train): {outcome['alpha']}. "
        f"Part sizes: {outcome['n']}.",
        "",
        "| Baseline | Part | n | Spearman | 95% CI | RMSE | 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, by_part in outcome["results"].items():
        for part, ev in by_part.items():
            lines.append(
                f"| {name} | {part} | {ev['n']} | {_num(ev['spearman'])} | "
                f"{_interval(ev['ci']['spearman'])} | {_num(ev['rmse'])} | "
                f"{_interval(ev['ci']['rmse'])} |"
            )
    return "\n".join(lines) + "\n"


def run_baselines(
    parts: dict[str, list[TextExample]],
    *,
    higher_is_easier: bool,
    seed: int = 0,
    n_boot: int = 1000,
    eval_parts: Sequence[str] = ("val", "test"),
    analyzer: VocabularyAnalyzer | None = None,
) -> dict:
    """Fit every baseline on ``parts['train']`` and evaluate on ``eval_parts``.

    Returns ``{"targets": fit statistics, "alpha", "n", "results": {baseline: {part: metrics}}}``.
    """
    train = parts["train"]
    targets = fit_targets([e.target for e in train], higher_is_easier=higher_is_easier)
    y = {
        name: to_difficulty([e.target for e in parts[name]], targets["scaler"])
        for name in ("train", *eval_parts)
    }
    x = {
        name: feature_matrix([e.text for e in parts[name]], analyzer)
        for name in ("train", *eval_parts)
    }
    train_groups = [e.group_id for e in train]
    alpha = select_alpha(x["train"], y["train"], train_groups, seed=seed)
    models: dict[str, tuple[RidgeModel, list[int]]] = {
        name: (RidgeModel(SINGLE_ALPHA).fit(x["train"][:, [i]], y["train"]), [i])
        for i, name in enumerate(SINGLE_FEATURE_BASELINES)
    }
    all_columns = list(range(len(FEATURE_NAMES)))
    models[RIDGE_NAME] = (RidgeModel(alpha).fit(x["train"], y["train"]), all_columns)
    results: dict[str, dict] = {}
    for name, (model, columns) in models.items():
        results[name] = {
            part: evaluate_regressor(
                y[part],
                model.predict(x[part][:, columns]),
                [e.group_id for e in parts[part]],
                seed,
                n_boot=n_boot,
            )
            for part in eval_parts
        }
    return {
        "targets": targets,
        "alpha": alpha,
        "n": {name: len(parts[name]) for name in ("train", *eval_parts)},
        "results": results,
    }
