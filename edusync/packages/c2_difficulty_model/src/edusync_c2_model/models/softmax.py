"""Deterministic multinomial (or ordinal cumulative-logit) logistic regression in float64 numpy.

Full-batch accelerated gradient descent with backtracking and momentum restarts: no randomness
except the seeded initial weights, a fixed iteration cap and a gradient-norm convergence test.
L2 never penalises the intercept (or the ordinal thresholds). Class weights are normalised so the
loss is a weighted mean.
"""

from collections.abc import Callable, Sequence

import numpy as np

Objective = Callable[[np.ndarray], tuple[float, np.ndarray]]

_MIN_PROB = 1e-12
_MIN_STEP = 1e-12
_CLIP = 500.0


def minimize(
    fun: Objective, x0: np.ndarray, *, max_iter: int = 300, tol: float = 1e-6
) -> tuple[np.ndarray, bool, int]:
    """Minimise ``fun`` (value, gradient) from ``x0``; returns (x, converged, iterations)."""
    x = np.array(x0, dtype=np.float64)
    z = x.copy()
    t = 1.0
    step = 1.0
    fx, _ = fun(x)
    for iteration in range(1, max_iter + 1):
        fz, gz = fun(z)
        gnorm2 = float(gz @ gz)
        if gnorm2 <= tol * tol:
            return x, True, iteration - 1
        cand = z - step * gz
        fc, _ = fun(cand)
        while fc > fz - 0.5 * step * gnorm2 and step > _MIN_STEP:
            step *= 0.5
            cand = z - step * gz
            fc, _ = fun(cand)
        if fc > fx:  # momentum overshoot: restart from the best point
            z = x.copy()
            t = 1.0
        else:
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            z = cand + ((t - 1.0) / t_next) * (cand - x)
            x, fx, t = cand, fc, t_next
        step *= 1.5
    return x, False, max_iter


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -_CLIP, _CLIP)))


def class_weights(y: np.ndarray, n_classes: int, spec: str | Sequence[float] | None) -> np.ndarray:
    """Per-class weights: ``None`` (all 1), ``"balanced"`` (n / (K * count)) or explicit values."""
    if spec is None:
        return np.ones(n_classes)
    if isinstance(spec, str):
        if spec != "balanced":
            raise ValueError(f"class_weight must be None, 'balanced' or a sequence, got {spec!r}")
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        return np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1.0)), 0.0)
    weights = np.asarray(spec, dtype=np.float64)
    if weights.shape != (n_classes,) or (weights < 0).any():
        raise ValueError(f"class_weight needs {n_classes} non-negative values")
    return weights


class SoftmaxRegression:
    """Multinomial logistic regression, or the ordinal cumulative-logit variant."""

    def __init__(
        self,
        n_classes: int = 3,
        *,
        l2: float = 0.01,
        ordinal: bool = False,
        class_weight: str | Sequence[float] | None = "balanced",
        max_iter: int = 300,
        tol: float = 1e-6,
    ) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.n_classes = n_classes
        self.l2 = l2
        self.ordinal = ordinal
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None  # (d, K) multinomial, (d,) ordinal
        self.intercept_: np.ndarray | None = None  # (K,) multinomial, (K-1,) thresholds ordinal
        self.converged_ = False
        self.n_iter_ = 0

    # ------------------------------------------------------------------ fitting

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> "SoftmaxRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be 2-D with one label per row and at least one row")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError(f"labels must be 0..{self.n_classes - 1}")
        sample_weight = class_weights(y, self.n_classes, self.class_weight)[y]
        total = sample_weight.sum()
        sw = sample_weight / total if total > 0 else np.full(len(y), 1.0 / len(y))
        rng = np.random.default_rng(seed)
        d, k = X.shape[1], self.n_classes
        if self.ordinal:
            fun = self._ordinal_objective(X, y, sw)
            x0 = np.concatenate([rng.normal(0.0, 0.01, d), [-1.0], np.zeros(k - 2)])
        else:
            fun = self._softmax_objective(X, y, sw)
            share = np.bincount(y, weights=sw, minlength=k) + 1e-3
            x0 = np.concatenate([rng.normal(0.0, 0.01, d * k), np.log(share / share.sum())])
        theta, self.converged_, self.n_iter_ = minimize(
            fun, x0, max_iter=self.max_iter, tol=self.tol
        )
        if self.ordinal:
            self.coef_ = theta[:d]
            c = theta[d:]
            self.intercept_ = c[0] + np.concatenate([[0.0], np.cumsum(np.exp(c[1:]))])
        else:
            self.coef_ = theta[: d * k].reshape(d, k)
            self.intercept_ = theta[d * k :]
        return self

    def _softmax_objective(self, X: np.ndarray, y: np.ndarray, sw: np.ndarray) -> Objective:
        d, k, l2 = X.shape[1], self.n_classes, self.l2
        onehot = np.eye(k)[y]

        def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
            w = theta[: d * k].reshape(d, k)
            probs = _softmax(X @ w + theta[d * k :])
            loss = -float((sw * np.log(np.maximum(probs[np.arange(len(y)), y], _MIN_PROB))).sum())
            loss += 0.5 * l2 * float((w * w).sum())
            resid = (probs - onehot) * sw[:, None]
            return loss, np.concatenate([(X.T @ resid + l2 * w).ravel(), resid.sum(axis=0)])

        return fun

    def _ordinal_objective(self, X: np.ndarray, y: np.ndarray, sw: np.ndarray) -> Objective:
        d, k, l2 = X.shape[1], self.n_classes, self.l2
        rows = np.arange(len(y))
        hit_hi = np.eye(k + 1)[y + 1]
        hit_lo = np.eye(k + 1)[y]

        def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
            w, c = theta[:d], theta[d:]
            gaps = np.exp(c[1:])
            thr = c[0] + np.concatenate([[0.0], np.cumsum(gaps)])
            cdf = _sigmoid(thr[None, :] - (X @ w)[:, None])
            pad = np.hstack([np.zeros((len(y), 1)), cdf, np.ones((len(y), 1))])
            slope = pad * (1.0 - pad)
            py = np.maximum(pad[rows, y + 1] - pad[rows, y], _MIN_PROB)
            loss = -float((sw * np.log(py)).sum()) + 0.5 * l2 * float(w @ w)
            coef = -sw / py
            d_score = -(slope[rows, y + 1] - slope[rows, y])
            d_thr = (slope * (hit_hi - hit_lo))[:, 1:k]
            g_thr = (coef[:, None] * d_thr).sum(axis=0)
            g_c = np.concatenate([[g_thr.sum()], gaps * np.cumsum(g_thr[1:][::-1])[::-1]])
            return loss, np.concatenate([X.T @ (coef * d_score) + l2 * w, g_c])

        return fun

    def fit_uniform(self, n_features: int) -> "SoftmaxRegression":
        """A neutral head with zero weights (the same output for every input)."""
        k = self.n_classes
        if self.ordinal:
            self.coef_ = np.zeros(n_features)
            self.intercept_ = np.linspace(-1.0, 1.0, k - 1)
        else:
            self.coef_ = np.zeros((n_features, k))
            self.intercept_ = np.zeros(k)
        return self

    # ---------------------------------------------------------------- prediction

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """``(n, K)`` probabilities that sum to 1 in every row."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("the model is not fitted")
        X = np.asarray(X, dtype=np.float64)
        if not self.ordinal:
            return _softmax(X @ self.coef_ + self.intercept_)
        cdf = _sigmoid(self.intercept_[None, :] - (X @ self.coef_)[:, None])
        pad = np.hstack([np.zeros((len(X), 1)), cdf, np.ones((len(X), 1))])
        probs = np.maximum(np.diff(pad, axis=1), 0.0)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def class_coefficients(self) -> np.ndarray:
        """``(d, K)`` per-class input weights, centred over classes (for explanations).

        Multinomial: the weight columns minus their class mean. Ordinal: the single weight vector
        scaled by -1 .. +1 across the ordered classes.
        """
        if self.coef_ is None:
            raise RuntimeError("the model is not fitted")
        if self.ordinal:
            span = np.linspace(-1.0, 1.0, self.n_classes)
            return self.coef_[:, None] * span[None, :]
        return self.coef_ - self.coef_.mean(axis=1, keepdims=True)

    # ------------------------------------------------------------ serialisation

    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("the model is not fitted")
        return {f"{prefix}.coef": self.coef_, f"{prefix}.intercept": self.intercept_}

    @classmethod
    def from_arrays(
        cls, data: dict[str, np.ndarray], prefix: str, *, ordinal: bool
    ) -> "SoftmaxRegression":
        coef = np.asarray(data[f"{prefix}.coef"], dtype=np.float64)
        intercept = np.asarray(data[f"{prefix}.intercept"], dtype=np.float64)
        model = cls(len(intercept) + (1 if ordinal else 0), ordinal=ordinal)
        model.coef_, model.intercept_ = coef, intercept
        return model


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)
