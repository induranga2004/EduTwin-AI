"""Group experts (abstention, mask dropout) and the stacking combiner."""

import numpy as np
import pytest

from edusync_c2.difficulty.schemas import FeatureGroup
from edusync_c2_model.models.combiner import StackingCombiner, combiner_features
from edusync_c2_model.models.experts import (
    EXPERT_SPECS,
    GroupExpert,
    canonical_order,
    mask_dropout,
    to_arrays,
)

A, B, C, D = EXPERT_SPECS


def embeddings_for(toy, examples):
    return toy.hash_embed([e.text for e in examples])


def test_to_arrays_and_canonical_order(toy):
    examples = toy.make_toy(3, 3)
    arrays = to_arrays(examples)
    assert arrays.values.shape == (9, toy.N)
    assert arrays.missing.dtype == bool
    assert arrays.has_text.all()
    assert arrays.y.tolist() == [e.label_id for e in examples]
    assert len(arrays) == 9
    assert len(arrays.take(np.array([0, 4]))) == 2
    assert to_arrays([]).values.shape == (0, toy.N)
    shuffled = list(reversed(examples))
    assert canonical_order(shuffled) == canonical_order(examples)


def test_mask_dropout_is_seeded_and_only_adds_missing():
    base = np.zeros((50, 4), dtype=bool)
    base[0, 0] = True
    one = mask_dropout(base, 0.3, np.random.default_rng(1), whole_rows=False)
    two = mask_dropout(base, 0.3, np.random.default_rng(1), whole_rows=False)
    assert np.array_equal(one, two)
    assert one[0, 0]
    assert 0 < one.sum() < one.size
    rows = mask_dropout(base, 0.5, np.random.default_rng(2), whole_rows=True)
    assert rows.all(axis=1).any()  # some whole rows are masked


def test_votes_and_abstention_rules(toy):
    examples = toy.make_toy(2, 3)
    no_b = toy.mask_group(examples, FeatureGroup.PREREQUISITES)
    no_a_no_text = toy.mask_group(examples, FeatureGroup.LANGUAGE, drop_text=True)
    no_a = toy.mask_group(examples, FeatureGroup.LANGUAGE)
    fitted = {
        s.letter: GroupExpert(s).fit(to_arrays(examples), embeddings_for(toy, examples))
        for s in (A, B)
    }
    assert fitted["B"].votes(to_arrays(examples)).all()
    assert not fitted["B"].votes(to_arrays(no_b)).any()  # all of its group missing: no vote
    assert fitted["A"].votes(to_arrays(no_a)).all()  # text exists: a text expert never abstains
    assert not fitted["A"].votes(to_arrays(no_a_no_text)).any()  # nothing to look at


def test_expert_probabilities_are_valid_with_a_group_masked_and_ignore_masked_values(toy):
    examples = toy.make_toy(8, 6)
    expert = GroupExpert(A, mask_dropout_rate=0.3).fit(
        to_arrays(examples), embeddings_for(toy, examples), seed=1
    )
    masked = toy.mask_group(examples, FeatureGroup.LANGUAGE)
    # what is stored under a masked cell must not matter
    tampered = [
        type(e)(
            **{
                **e.__dict__,
                "values": tuple(99.0 if m else v for v, m in zip(e.values, e.missing, strict=True)),
            }
        )
        for e in masked
    ]
    probs, votes = expert.predict_proba(to_arrays(masked), embeddings_for(toy, masked))
    probs2, _ = expert.predict_proba(to_arrays(tampered), embeddings_for(toy, tampered))
    assert votes.all()
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()
    assert np.array_equal(probs, probs2)


def test_features_recover_toy_labels_and_contributions_point_the_right_way(toy):
    train, test = toy.make_toy(10, 9, seed=1), toy.make_toy(4, 9, seed=2, prefix="T")
    expert = GroupExpert(D, l2=1e-3).fit(to_arrays(train), None, seed=0)
    probs, votes = expert.predict_proba(to_arrays(test), None)
    assert votes.all()
    assert (probs.argmax(axis=1) == [e.label_id for e in test]).mean() > 0.95
    arrays = to_arrays(test)
    high = np.full(len(test), 2)
    contrib = expert.contributions(arrays, high)
    assert contrib.shape == (len(test), len(expert.feature_names))
    labels = arrays.y
    assert contrib[labels == 2].mean() > 0 > contrib[labels == 0].mean()
    z = expert.feature_z(arrays)
    assert z.shape == contrib.shape


def test_ordinal_expert_and_zero_dropout(toy):
    train = toy.make_toy(8, 9)
    expert = GroupExpert(B, ordinal=True, mask_dropout_rate=0.0, l2=1e-3)
    expert.fit(to_arrays(train), None)
    probs, _ = expert.predict_proba(to_arrays(train), None)
    assert (probs.argmax(axis=1) == to_arrays(train).y).mean() > 0.95


def test_full_dropout_leaves_a_feature_only_expert_with_only_the_clean_rows(toy):
    train = toy.make_toy(4, 6)
    expert = GroupExpert(B, mask_dropout_rate=1.0).fit(to_arrays(train), None)
    probs, votes = expert.predict_proba(to_arrays(train), None)
    assert votes.all()
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_an_expert_with_nothing_to_learn_from_always_abstains_and_round_trips(toy):
    train = toy.mask_group(toy.make_toy(4, 6), FeatureGroup.PREREQUISITES)
    expert = GroupExpert(B).fit(to_arrays(train), None)
    assert not expert.active
    probs, votes = expert.predict_proba(to_arrays(train), None)
    assert not votes.any()
    assert np.allclose(probs, 1 / 3)
    clone = GroupExpert.from_arrays(B, expert.to_arrays(), ordinal=False)
    assert not clone.active
    text_expert = GroupExpert(C, ordinal=True).fit(
        to_arrays(toy.mask_group(train, FeatureGroup.EXPLANATION, drop_text=True)),
        embeddings_for(toy, train),
    )
    assert not text_expert.active
    assert text_expert.n_embedding == 8


def test_text_experts_need_one_embedding_per_example(toy):
    examples = toy.make_toy(2, 3)
    arrays = to_arrays(examples)
    with pytest.raises(ValueError, match="one embedding per example"):
        GroupExpert(A).fit(arrays, None)
    with pytest.raises(ValueError, match="one embedding per example"):
        GroupExpert(A).fit(arrays, np.zeros((2, 8)))


def test_expert_arrays_round_trip_to_identical_probabilities(toy):
    examples = toy.make_toy(6, 6)
    emb = embeddings_for(toy, examples)
    for spec, ordinal in ((A, False), (C, True), (D, False)):
        expert = GroupExpert(spec, ordinal=ordinal).fit(
            to_arrays(examples), emb if spec.use_text else None
        )
        clone = GroupExpert.from_arrays(spec, expert.to_arrays(), ordinal=ordinal)
        use = emb if spec.use_text else None
        assert np.array_equal(
            expert.predict_proba(to_arrays(examples), use)[0],
            clone.predict_proba(to_arrays(examples), use)[0],
        )
        assert clone.n_embedding == expert.n_embedding


# ------------------------------------------------------------------ combiner


def informative_probs(y, strength=0.8, experts=2):
    probs = np.full((len(y), experts, 3), (1 - strength) / 2)
    probs[np.arange(len(y)), :, y] = strength
    return probs


def test_combiner_features_zero_out_abstainers():
    probs = np.full((1, 2, 3), 1 / 3)
    feats = combiner_features(probs, np.array([[True, False]]))
    assert feats.shape == (1, 8)
    assert feats[0, :3].tolist() == pytest.approx([1 / 3] * 3)
    assert feats[0, 3:6].tolist() == [0.0, 0.0, 0.0]
    assert feats[0, 6:].tolist() == [1.0, 0.0]


def test_combiner_learns_and_tolerates_abstaining_experts():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    probs = informative_probs(y)
    votes = np.ones((300, 2), dtype=bool)
    combiner = StackingCombiner(l2=1e-3).fit(probs, votes, y, seed=0)
    assert (combiner.predict_proba(probs, votes).argmax(axis=1) == y).mean() > 0.95
    one_off = votes.copy()
    one_off[:, 0] = False
    still = combiner.predict_proba(probs, one_off)
    assert (still.argmax(axis=1) == y).mean() > 0.9  # the remaining expert carries the vote
    nobody = combiner.predict_proba(probs, np.zeros_like(votes))
    assert np.allclose(nobody.sum(axis=1), 1.0)
    assert np.allclose(nobody, nobody[0])  # just the class prior
    assert np.array_equal(
        combiner.predict_proba(probs, votes),
        StackingCombiner.from_arrays(combiner.to_arrays()).predict_proba(probs, votes),
    )


def test_combiner_trains_on_rows_with_votes_only_and_rejects_none():
    y = np.array([0, 1, 2, 1])
    probs = informative_probs(y)
    votes = np.array([[True, True], [False, False], [True, False], [False, True]])
    combiner = StackingCombiner(vote_dropout=0.0).fit(probs, votes, y)
    assert np.allclose(combiner.predict_proba(probs, votes).sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="no row has a vote"):
        StackingCombiner().fit(probs, np.zeros((4, 2), dtype=bool), y)
    all_dropped = StackingCombiner(vote_dropout=1.0).fit(probs, votes, y)
    assert np.allclose(all_dropped.predict_proba(probs, votes).sum(axis=1), 1.0)
