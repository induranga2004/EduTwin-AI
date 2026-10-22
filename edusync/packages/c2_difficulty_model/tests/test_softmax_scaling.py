"""Softmax/ordinal regression, the optimiser, standardisation and the embedding cache."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edusync_c2_model.models import embeddings as emb_mod
from edusync_c2_model.models.embeddings import CachedEmbedder, encoder_embedder, text_key
from edusync_c2_model.models.scaling import Standardizer
from edusync_c2_model.models.softmax import SoftmaxRegression, class_weights, minimize

BLUEPRINT = (
    Path(__file__).resolve().parents[2]
    / "blueprint_model"
    / "artifacts"
    / "base"
    / "sentence-transformers--all-MiniLM-L6-v2"
)


def blobs(n_per=40, seed=0, gap=4.0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(gap * c, 0.5, (n_per, 2)) for c in range(3)])
    y = np.repeat(np.arange(3), n_per)
    return X, y


# ------------------------------------------------------------------ optimiser


def test_minimize_solves_an_ill_conditioned_quadratic_with_halving_and_restarts():
    scale = np.array([1.0, 500.0])

    def fun(x):
        return 0.5 * float((scale * x * x).sum()), scale * x

    x, converged, iterations = minimize(fun, np.array([3.0, 3.0]), max_iter=2000, tol=1e-8)
    assert converged
    assert iterations > 0
    assert np.abs(x).max() < 1e-6


def test_minimize_stops_at_the_iteration_cap():
    def fun(x):
        return float(x @ x), 2 * x

    x, converged, iterations = minimize(fun, np.array([1.0]), max_iter=1, tol=1e-30)
    assert (converged, iterations) == (False, 1)
    assert x.shape == (1,)


def test_minimize_returns_at_once_when_already_optimal():
    _, converged, iterations = minimize(lambda x: (float(x @ x), 2 * x), np.zeros(2))
    assert (converged, iterations) == (True, 0)


# ------------------------------------------------------------------ softmax


def test_separable_toy_data_is_recovered():
    X, y = blobs()
    model = SoftmaxRegression(l2=1e-3).fit(X, y, seed=1)
    assert (model.predict(X) == y).all()
    probs = model.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert model.converged_
    assert model.class_coefficients().shape == (2, 3)


def test_fit_is_deterministic_and_the_seed_only_sets_the_start():
    X, y = blobs()
    a = SoftmaxRegression().fit(X, y, seed=3)
    b = SoftmaxRegression().fit(X, y, seed=3)
    c = SoftmaxRegression().fit(X, y, seed=4)
    assert np.array_equal(a.coef_, b.coef_)
    assert not np.array_equal(a.coef_, c.coef_)
    assert np.allclose(a.predict_proba(X), c.predict_proba(X), atol=1e-3)


def test_ordinal_variant_orders_the_classes_and_the_thresholds():
    rng = np.random.default_rng(0)
    score = rng.normal(0.0, 1.0, 600)
    y = np.digitize(score + rng.normal(0.0, 0.3, 600), [-0.5, 0.5])
    X = score[:, None]
    model = SoftmaxRegression(ordinal=True, l2=1e-4, class_weight=None).fit(X, y)
    assert model.coef_.shape == (1,)
    assert model.coef_[0] > 0
    assert (np.diff(model.intercept_) > 0).all()
    low, mid, high = (model.predict_proba(np.array([[v]]))[0] for v in (-2.0, 0.0, 2.0))
    assert low.argmax() == 0
    assert mid.argmax() == 1
    assert high.argmax() == 2
    assert np.allclose(low.sum(), 1.0)
    # the expected class index grows with the score
    grid = model.predict_proba(np.linspace(-3, 3, 25)[:, None]) @ np.arange(3)
    assert (np.diff(grid) > 0).all()
    assert (model.class_coefficients()[0] == model.coef_[0] * np.array([-1.0, 0.0, 1.0])).all()


def test_two_class_ordinal_has_one_threshold():
    X, y = blobs()
    y2 = (y > 0).astype(int)
    model = SoftmaxRegression(2, ordinal=True, l2=1e-3).fit(X, y2)
    assert model.intercept_.shape == (1,)
    assert (model.predict(X) == y2).mean() > 0.95


@pytest.mark.parametrize("ordinal", [False, True])
def test_objective_gradients_match_finite_differences(ordinal):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = rng.integers(0, 3, 30)
    sw = np.full(30, 1 / 30)
    model = SoftmaxRegression(l2=0.3, ordinal=ordinal)
    fun = (model._ordinal_objective if ordinal else model._softmax_objective)(X, y, sw)
    theta = rng.normal(0.0, 0.5, 3 + 2 if ordinal else 3 * 3 + 3)
    _, grad = fun(theta)
    for i in range(len(theta)):
        step = np.zeros_like(theta)
        step[i] = 1e-6
        numeric = (fun(theta + step)[0] - fun(theta - step)[0]) / 2e-6
        assert grad[i] == pytest.approx(numeric, abs=1e-6)


def test_balanced_class_weights_lift_the_minority_class():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(0.0, 1.0, (300, 1)), rng.normal(1.0, 1.0, (30, 1))])
    y = np.array([0] * 300 + [1] * 30)
    plain = SoftmaxRegression(2, class_weight=None, l2=1e-3).fit(X, y)
    balanced = SoftmaxRegression(2, class_weight="balanced", l2=1e-3).fit(X, y)
    assert (balanced.predict(X) == 1).sum() > (plain.predict(X) == 1).sum()


def test_class_weight_specs():
    y = np.array([0, 0, 0, 1])
    assert np.array_equal(class_weights(y, 3, None), np.ones(3))
    assert np.allclose(class_weights(y, 3, "balanced"), [4 / 9, 4 / 3, 0.0])
    assert np.array_equal(class_weights(y, 3, [1, 2, 3]), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="class_weight must be"):
        class_weights(y, 3, "heavy")
    with pytest.raises(ValueError, match="non-negative"):
        class_weights(y, 3, [1, 2])


def test_all_zero_class_weights_fall_back_to_equal_weights():
    X, y = blobs(10)
    model = SoftmaxRegression(class_weight=[0.0, 0.0, 0.0]).fit(X, y)
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_invalid_use_is_rejected():
    X, y = blobs(5)
    with pytest.raises(ValueError, match="at least 2"):
        SoftmaxRegression(1)
    with pytest.raises(ValueError, match="non-negative"):
        SoftmaxRegression(l2=-1)
    with pytest.raises(ValueError, match="2-D"):
        SoftmaxRegression().fit(X[:3], y)
    with pytest.raises(ValueError, match=r"0\.\.2"):
        SoftmaxRegression().fit(X, y + 1)
    fresh = SoftmaxRegression()
    with pytest.raises(RuntimeError, match="not fitted"):
        fresh.predict_proba(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        fresh.class_coefficients()
    with pytest.raises(RuntimeError, match="not fitted"):
        fresh.to_arrays("m")


@pytest.mark.parametrize("ordinal", [False, True])
def test_arrays_round_trip_and_neutral_head(ordinal):
    X, y = blobs(10)
    model = SoftmaxRegression(ordinal=ordinal).fit(X, y)
    clone = SoftmaxRegression.from_arrays(model.to_arrays("m"), "m", ordinal=ordinal)
    assert np.array_equal(clone.predict_proba(X), model.predict_proba(X))
    neutral = SoftmaxRegression(ordinal=ordinal).fit_uniform(2)
    probs = neutral.predict_proba(X)
    assert np.allclose(probs, probs[0])
    assert np.allclose(probs.sum(axis=1), 1.0)


# ------------------------------------------------------------------ scaling


def test_standardizer_uses_observed_cells_only_and_zeroes_missing():
    values = np.array([[1.0, 5.0, 7.0], [3.0, 99.0, 7.0], [5.0, 9.0, 7.0]])
    missing = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=bool)
    scaler = Standardizer.fit(values, missing)
    assert scaler.mean.tolist() == [3.0, 7.0, 0.0]  # third column has no data: (0, 1)
    assert scaler.std[0] == pytest.approx(np.sqrt(8 / 3))
    assert scaler.std[1] == pytest.approx(2.0)
    assert scaler.std[2] == 1.0
    z = scaler.transform(values, missing)
    assert z[1, 1] == 0.0
    assert z[:, 2].tolist() == [0.0, 0.0, 0.0]
    assert z[0, 0] == pytest.approx(-2 / np.sqrt(8 / 3))


def test_constant_columns_get_unit_std_and_arrays_round_trip():
    scaler = Standardizer.fit(np.full((4, 1), 2.0), np.zeros((4, 1), dtype=bool))
    assert scaler.std.tolist() == [1.0]
    clone = Standardizer.from_arrays(scaler.to_arrays("s"), "s")
    assert clone.mean.tolist() == [2.0]
    assert clone.transform(np.array([[3.0]]), np.zeros((1, 1), dtype=bool))[0, 0] == 1.0


# ------------------------------------------------------------------ embeddings


def fake_embed(texts):
    return np.array([[len(t), sum(map(ord, t)) % 7] for t in texts], dtype=np.float32)


def test_cached_embedder_computes_each_text_once_in_any_order(tmp_path):
    calls = []

    def spy(texts):
        calls.append(list(texts))
        return fake_embed(texts)

    cache = CachedEmbedder(spy, "enc1", tmp_path)
    first = cache(["bb", "a", "bb"])
    assert first.shape == (3, 2)
    assert (first[0] == first[2]).all()
    assert cache.computed == 2
    assert sorted(calls[0], key=text_key) == calls[0]  # distinct texts, sorted by hash
    assert np.array_equal(cache(["a", "bb"]), first[[1, 0]])
    assert cache.computed == 2

    reloaded = CachedEmbedder(lambda texts: pytest.fail("must come from disk"), "enc1", tmp_path)
    assert np.array_equal(reloaded(["bb"]), first[:1])
    assert (tmp_path / "embeddings-enc1.npz").is_file()
    other = CachedEmbedder(fake_embed, "enc2", tmp_path)
    other(["a"])
    assert other.computed == 1  # a different encoder never reuses the cache


def test_cached_embedder_without_a_directory_and_for_no_texts():
    cache = CachedEmbedder(fake_embed, "x")
    assert cache([]).shape[0] == 0
    assert cache(["z"]).shape == (1, 2)


def test_encoder_embedder_wraps_the_default_encoder(monkeypatch):
    encoder = SimpleNamespace(encode=fake_embed, sha="abcd1234")
    monkeypatch.setattr(
        "edusync_c2.difficulty.features.semantic_loader.get_default_encoder",
        lambda model_dir: encoder,
    )
    embed, sha = encoder_embedder(Path("anywhere"))
    assert sha == "abcd1234"
    assert embed(["ab"]).shape == (1, 2)
    monkeypatch.setattr(
        "edusync_c2.difficulty.features.semantic_loader.get_default_encoder", lambda model_dir: None
    )
    with pytest.raises(RuntimeError, match="cannot be loaded"):
        emb_mod.encoder_embedder(Path("anywhere"))


@pytest.mark.ml
@pytest.mark.skipif(not BLUEPRINT.is_dir(), reason="MiniLM weights are not present")
def test_real_minilm_embeddings_are_unit_norm_and_cached(tmp_path):
    embed, sha = encoder_embedder(BLUEPRINT)
    cache = CachedEmbedder(embed, sha, tmp_path)
    vectors = cache(["a loop repeats a block", "a tree has a root"])
    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert cache.computed == 2
