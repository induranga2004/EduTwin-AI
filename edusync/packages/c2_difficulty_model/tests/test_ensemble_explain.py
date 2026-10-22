"""Group ensemble: leak-free stacking, determinism, artifact round trip, explanations."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from edusync_c2.difficulty.schemas import FeatureGroup
from edusync_c2_model.examples import write_examples
from edusync_c2_model.models import ensemble as ensemble_mod
from edusync_c2_model.models.ensemble import (
    EnsembleParams,
    GroupEnsemble,
    fit_experts_from_datasets,
    split_by_lecture,
)
from edusync_c2_model.models.explain import explain
from edusync_c2_model.split import LeakageError, SplitError

SRC = Path(__file__).resolve().parents[1] / "src"
TOYDATA = Path(__file__).resolve().parent / "toydata.py"
FAST = EnsembleParams(l2=1e-3, max_iter=150)


@pytest.fixture(scope="module")
def fitted(toy):
    return GroupEnsemble.fit(
        toy.make_toy(12, 9, seed=1), seed=3, folds=3, embed=toy.hash_embed, params=FAST
    )


def labels(examples):
    return np.array([e.label_id for e in examples])


def test_toy_data_is_recovered_with_valid_probabilities(toy, fitted):
    test = toy.make_toy(4, 9, seed=2, prefix="T")
    out = fitted.predict_detail(test)
    assert (out.probs.argmax(axis=1) == labels(test)).mean() > 0.95
    assert np.allclose(out.probs.sum(axis=1), 1.0)
    assert out.expert_probs.shape == (len(test), 4, 3)
    assert np.allclose(out.expert_probs.sum(axis=2), 1.0)
    assert out.votes.all()
    assert np.array_equal(fitted.predict(test), out.probs.argmax(axis=1))
    assert np.array_equal(fitted.predict_proba(test), out.probs)
    empty = fitted.predict_detail([])
    assert empty.probs.shape == (0, 3)
    assert empty.expert_probs.shape == (0, 4, 3)


def test_out_of_fold_scheme_never_scores_a_lecture_with_a_model_that_saw_it(toy, monkeypatch):
    calls = []
    real = ensemble_mod.check_disjoint

    def spy(fit_ids, predict_ids):
        calls.append((set(fit_ids), set(predict_ids)))
        real(fit_ids, predict_ids)

    monkeypatch.setattr(ensemble_mod, "check_disjoint", spy)
    train = toy.make_toy(10, 6, seed=5)
    model = GroupEnsemble.fit(train, seed=1, folds=5, embed=toy.hash_embed, params=FAST)
    assert len(calls) == 5
    assert all(not (a & b) for a, b in calls)
    valid_all = [lec for _, valid in model.fold_lectures for lec in valid]
    assert sorted(valid_all) == sorted({e.lecture_id for e in train})  # each lecture once
    for fit_lectures, valid in model.fold_lectures:
        assert not set(fit_lectures) & set(valid)
    assert model.oof_probs.shape == (len(train), 4, 3)
    assert model.oof_votes.all()


def test_fit_is_deterministic_and_ignores_input_order(toy):
    train = toy.make_toy(8, 6, seed=3)
    one = GroupEnsemble.fit(train, seed=2, folds=3, embed=toy.hash_embed, params=FAST)
    shuffled = list(np.random.default_rng(0).permutation(np.array(train, dtype=object)))
    two = GroupEnsemble.fit(shuffled, seed=2, folds=3, embed=toy.hash_embed, params=FAST)
    test = toy.make_toy(3, 6, seed=9, prefix="T")
    assert np.array_equal(one.predict_proba(test), two.predict_proba(test))
    other_seed = GroupEnsemble.fit(train, seed=3, folds=3, embed=toy.hash_embed, params=FAST)
    assert not np.array_equal(one.predict_proba(test), other_seed.predict_proba(test))


def test_probabilities_are_identical_across_python_hash_seeds():
    code = f"""
import hashlib, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location('toydata', r'{TOYDATA}')
toy = importlib.util.module_from_spec(spec); spec.loader.exec_module(toy)
from edusync_c2_model.models.ensemble import EnsembleParams, GroupEnsemble
model = GroupEnsemble.fit(toy.make_toy(6, 6), seed=1, folds=3, embed=toy.hash_embed,
                          params=EnsembleParams(l2=1e-3, max_iter=60))
probs = model.predict_proba(toy.make_toy(2, 6, seed=4, prefix='T'))
print(hashlib.sha256(probs.tobytes()).hexdigest())
"""
    outputs = set()
    for hash_seed in ("0", "1"):
        env = {
            **os.environ,
            "PYTHONHASHSEED": hash_seed,
            "PYTHONPATH": os.pathsep.join([str(SRC), *[p for p in sys.path if p]]),
        }
        done = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(done.stdout.strip())
    assert len(outputs) == 1


def test_abstaining_experts_and_missing_groups_still_give_valid_probabilities(toy, fitted):
    test = toy.make_toy(3, 6, seed=4, prefix="T")
    no_graph = toy.mask_group(test, FeatureGroup.PREREQUISITES)
    out = fitted.predict_detail(no_graph)
    assert not out.votes[:, 1].any()  # B abstains without a graph
    assert out.votes[:, [0, 2, 3]].all()
    assert np.allclose(out.probs.sum(axis=1), 1.0)
    assert (out.probs.argmax(axis=1) == labels(test)).mean() > 0.9
    nothing = toy.mask_group(test, FeatureGroup.LANGUAGE, drop_text=True)
    for group in (FeatureGroup.PREREQUISITES, FeatureGroup.EXPLANATION, FeatureGroup.DENSITY):
        nothing = toy.mask_group(nothing, group, drop_text=True)
    blank = fitted.predict_detail(nothing)
    assert not blank.votes.any()
    assert np.allclose(blank.probs.sum(axis=1), 1.0)


def test_training_without_a_whole_group_makes_that_expert_abstain_always(toy):
    train = toy.mask_group(toy.make_toy(8, 6, seed=2), FeatureGroup.PREREQUISITES)
    model = GroupEnsemble.fit(train, seed=0, folds=3, embed=toy.hash_embed, params=FAST)
    out = model.predict_detail(toy.make_toy(2, 6, seed=8, prefix="T"))
    assert not out.votes[:, 1].any()
    assert np.allclose(out.probs.sum(axis=1), 1.0)


@pytest.mark.parametrize("ordinal", [False, True])
def test_npz_round_trip_gives_identical_probabilities(toy, tmp_path, ordinal):
    params = EnsembleParams(l2=1e-3, max_iter=80, ordinal=ordinal)
    model = GroupEnsemble.fit(
        toy.make_toy(6, 6), seed=1, folds=3, embed=toy.hash_embed, params=params, encoder_id="fake"
    )
    path = tmp_path / "weights.npz"
    np.savez(path, **model.to_npz_dict())
    with np.load(path, allow_pickle=False) as stored:
        clone = GroupEnsemble.from_npz_dict(stored, embed=toy.hash_embed)
    test = toy.make_toy(3, 6, seed=6, prefix="T")
    assert np.array_equal(model.predict_proba(test), clone.predict_proba(test))
    assert clone.encoder_id == "fake"
    assert clone.params == params


def test_artifact_validation_rejects_bad_or_mismatched_data(fitted):
    good = fitted.to_npz_dict()
    config = json.loads(str(good["config"]))

    def with_config(**changes):
        return {**good, "config": np.array(json.dumps({**config, **changes}))}

    with pytest.raises(ValueError, match="not a valid ensemble"):
        GroupEnsemble.from_npz_dict({k: v for k, v in good.items() if k != "combiner.coef"})
    with pytest.raises(ValueError, match="not a valid ensemble"):
        GroupEnsemble.from_npz_dict({**good, "config": np.array("{broken")})
    with pytest.raises(ValueError, match="unsupported ensemble format"):
        GroupEnsemble.from_npz_dict(with_config(format=99))
    with pytest.raises(ValueError, match="unsupported ensemble format"):
        GroupEnsemble.from_npz_dict(with_config(labels=["High", "Medium", "Low"]))
    with pytest.raises(ValueError, match="different feature schema"):
        GroupEnsemble.from_npz_dict(with_config(feature_schema_hash="0" * 64))
    with pytest.raises(ValueError, match="different feature schema"):
        GroupEnsemble.from_npz_dict(with_config(feature_schema_version="0.1"))


def test_an_embedder_is_required_to_fit_or_predict(toy, fitted):
    with pytest.raises(ValueError, match="embed callable is required"):
        GroupEnsemble.fit(toy.make_toy(4, 3), embed=None)
    bare = GroupEnsemble.from_npz_dict(fitted.to_npz_dict())
    with pytest.raises(ValueError, match="embed callable is required"):
        bare.predict_proba(toy.make_toy(1, 3))


# ------------------------------------------------- one dataset per expert + combiner


def dataset_sets(toy):
    experts = {letter: toy.make_toy(8, 6, seed=i, prefix=letter) for i, letter in enumerate("ABCD")}
    return experts, toy.make_toy(8, 6, seed=9, prefix="M")


def test_dataset_specific_training_and_leakage_guard(toy, tmp_path):
    experts, combiner = dataset_sets(toy)
    model = GroupEnsemble.fit_from_example_sets(
        experts, combiner, seed=1, embed=toy.hash_embed, params=FAST
    )
    test = toy.make_toy(3, 6, seed=11, prefix="T")
    out = model.predict_detail(test)
    assert (out.probs.argmax(axis=1) == labels(test)).mean() > 0.9
    assert model.oof_probs.shape == (len(combiner), 4, 3)
    paths = {}
    for letter, rows in experts.items():
        paths[letter] = tmp_path / f"{letter}.jsonl"
        write_examples(paths[letter], rows)
    combiner_path = tmp_path / "M.jsonl"
    write_examples(combiner_path, combiner)
    from_files = fit_experts_from_datasets(
        paths, combiner_path, seed=1, embed=toy.hash_embed, params=FAST
    )
    assert np.array_equal(from_files.predict_proba(test), model.predict_proba(test))

    overlapping = dict(experts)
    overlapping["C"] = toy.make_toy(8, 6, seed=3, prefix="A")  # reuses expert A's lecture ids
    with pytest.raises(LeakageError):
        GroupEnsemble.fit_from_example_sets(overlapping, combiner, embed=toy.hash_embed)
    with pytest.raises(LeakageError):
        GroupEnsemble.fit_from_example_sets(experts, experts["B"][:6], embed=toy.hash_embed)
    with pytest.raises(ValueError, match="exactly the keys"):
        GroupEnsemble.fit_from_example_sets({"A": experts["A"]}, combiner, embed=toy.hash_embed)


def test_split_by_lecture_keeps_lectures_whole(toy):
    examples = toy.make_toy(20, 3)
    train, val, test = split_by_lecture(examples, seed=4, val_fraction=0.1, test_fraction=0.2)
    ids = [{e.lecture_id for e in part} for part in (train, val, test)]
    assert not (ids[0] & ids[1])
    assert not (ids[0] & ids[2])
    assert not (ids[1] & ids[2])
    assert len(train) + len(val) + len(test) == len(examples)
    assert (len(ids[1]), len(ids[2])) == (2, 4)
    assert split_by_lecture(examples, seed=4, val_fraction=0.1, test_fraction=0.2)[2] == test
    with pytest.raises(SplitError, match="at least 3"):
        split_by_lecture(toy.make_toy(2, 3), seed=0)
    with pytest.raises(SplitError, match="fractions"):
        split_by_lecture(examples, seed=0, val_fraction=0.6, test_fraction=0.5)
    with pytest.raises(SplitError, match="fractions"):
        split_by_lecture(examples, seed=0, val_fraction=0.0)


# ------------------------------------------------------------------ explain


def test_explanations_name_features_but_never_slide_text(toy, fitted):
    test = toy.make_toy(2, 3, seed=7, prefix="T")
    no_graph = toy.mask_group(test, FeatureGroup.PREREQUISITES)
    only_text = toy.mask_group(no_graph, FeatureGroup.LANGUAGE)
    nothing = toy.mask_group(only_text, FeatureGroup.EXPLANATION, drop_text=True)
    nothing = toy.mask_group(nothing, FeatureGroup.DENSITY)
    lines = explain(fitted, test)
    assert len(lines) == len(test)
    assert len(lines[0]) == 5
    assert lines[0][0].startswith("A_language: ")
    assert "top: lang." in lines[0][0]
    assert lines[0][0].count("+") + lines[0][0].count("-") >= 3
    assert lines[0][4].startswith("combined: ")
    assert "no vote" in explain(fitted, no_graph)[0][1]
    assert "text only" in explain(fitted, only_text)[0][0]
    assert explain(fitted, nothing)[0][-1] == "combined: no expert could vote"
    assert explain(fitted, test, top=1)[0][0].count(", ") == 0
    assert explain(fitted, []) == []
    blob = " ".join(line for group in lines for line in group)
    assert not any(e.text in blob or e.concept_label in blob for e in test)
