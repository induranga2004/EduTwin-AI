"""Group splits (leakage, determinism), metrics (hand-computed) and the markdown report."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from edusync_c2_model.examples import Example, FeatureRow
from edusync_c2_model.metrics import (
    bootstrap_ci,
    confusion_matrix,
    evaluate,
    metrics_from_confusion,
)
from edusync_c2_model.report import SYNTHETIC_BANNER, any_synthetic, render_markdown
from edusync_c2_model.split import (
    LeakageError,
    SplitError,
    check_disjoint,
    group_kfold,
    make_split,
    read_split,
    validate_split,
    write_split_once,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def lectures(n_courses=4, per_course=3):
    return {f"L{c}{k}": f"C{c}" for c in range(n_courses) for k in range(per_course)}


# ------------------------------------------------------------------------------------ split


def test_split_by_course_holds_out_whole_courses():
    mapping = lectures(4, 3)
    split = make_split(mapping, seed=1)
    assert split["unit"] == "course"
    test_courses = {mapping[lec] for lec in split["test"]}
    train_courses = {mapping[lec] for lec in split["train"]}
    assert test_courses
    assert train_courses
    assert not test_courses & train_courses
    assert sorted(split["train"] + split["test"]) == sorted(mapping)
    for course in test_courses:  # a held-out course is entirely in test
        assert {lec for lec, c in mapping.items() if c == course} <= set(split["test"])


def test_split_by_lecture_with_fewer_than_three_courses():
    mapping = lectures(2, 5)
    split = make_split(mapping, seed=1, test_fraction=0.3)
    assert split["unit"] == "lecture"
    assert len(split["test"]) == 3
    assert not set(split["train"]) & set(split["test"])


def test_split_is_seeded_and_independent_of_input_order():
    mapping = lectures(5, 4)
    reordered = dict(reversed(list(mapping.items())))
    assert make_split(mapping, seed=3) == make_split(reordered, seed=3)
    outcomes = {tuple(make_split(mapping, seed=s)["test"]) for s in range(12)}
    assert len(outcomes) > 1


def test_split_never_empties_either_side():
    split = make_split({"L1": "C1", "L2": "C1"}, seed=0, test_fraction=0.99)
    assert len(split["train"]) == len(split["test"]) == 1
    tiny = make_split({"L1": "C1", "L2": "C1", "L3": "C1"}, seed=0, test_fraction=0.01)
    assert len(tiny["test"]) == 1


@pytest.mark.parametrize("fraction", [0, 1, -0.5, 1.5])
def test_bad_test_fraction_raises(fraction):
    with pytest.raises(SplitError, match="test_fraction"):
        make_split(lectures(), seed=0, test_fraction=fraction)


def test_too_few_groups_raise():
    with pytest.raises(SplitError, match="at least 2"):
        make_split({"L1": "C1"}, seed=0)


def test_write_split_once_refuses_to_overwrite(tmp_path):
    path = tmp_path / "split.json"
    split = make_split(lectures(), seed=2)
    write_split_once(path, split)
    first = path.read_text(encoding="utf-8")
    with pytest.raises(SplitError, match="never overwritten"):
        write_split_once(path, make_split(lectures(), seed=99))
    assert path.read_text(encoding="utf-8") == first
    assert read_split(path) == split


def test_write_split_validates_before_writing(tmp_path):
    bad = make_split(lectures(), seed=2)
    bad["train"] = bad["train"] + [bad["test"][0]]
    with pytest.raises(LeakageError):
        write_split_once(tmp_path / "split.json", bad)
    assert not (tmp_path / "split.json").exists()


def test_read_split_rejects_bad_files(tmp_path):
    path = tmp_path / "s.json"
    with pytest.raises(SplitError, match="cannot use split"):
        read_split(path)
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(SplitError, match="cannot use split"):
        read_split(path)
    path.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(SplitError, match="cannot use split"):
        read_split(path)
    other = dict(make_split(lectures(), seed=1), version=2)
    path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(SplitError, match="unsupported split version"):
        read_split(path)


def test_validate_split_detects_leakage_and_inconsistency():
    mapping = lectures(4, 2)
    good = make_split(mapping, seed=5)
    validate_split(good, mapping)

    overlap = dict(good, train=good["train"] + good["test"][:1])
    with pytest.raises(LeakageError, match="both train and test"):
        validate_split(overlap)

    empty = dict(good, test=[])
    with pytest.raises(SplitError, match="non-empty"):
        validate_split(empty)

    uncovered = dict(good, courses={k: v for k, v in good["courses"].items() if k != "L00"})
    with pytest.raises(SplitError, match="cover exactly"):
        validate_split(uncovered)

    with pytest.raises(SplitError, match="does not match"):
        validate_split(good, {**mapping, "L00": "other"})

    # same course on both sides in a course-level split is leakage
    moved = dict(good)
    moved["courses"] = dict(good["courses"], **{good["train"][0]: good["courses"][good["test"][0]]})
    with pytest.raises(LeakageError, match="courses in both"):
        validate_split(moved)


def test_group_kfold_validates_every_lecture_once_without_overlap():
    ids = [f"L{i}" for i in range(10)]
    folds = group_kfold(ids, 4, seed=1)
    assert len(folds) == 4
    validated = [lec for _, valid in folds for lec in valid]
    assert sorted(validated) == sorted(ids)
    for train, valid in folds:
        assert not set(train) & set(valid)
        assert sorted(train + valid) == sorted(ids)


def test_group_kfold_is_deterministic_and_order_independent():
    ids = [f"L{i}" for i in range(9)]
    assert group_kfold(ids, 3, 7) == group_kfold(list(reversed(ids)), 3, 7)
    assert group_kfold(ids, 3, 7) == group_kfold(ids + ids[:2], 3, 7)  # duplicates collapse


@pytest.mark.parametrize("k", [1, 0, 5])
def test_group_kfold_bounds(k):
    with pytest.raises(SplitError, match="k must be"):
        group_kfold(["a", "b", "c", "d"], k, 0)


def test_check_disjoint():
    check_disjoint(["a"], ["b"])
    with pytest.raises(LeakageError, match="both sides"):
        check_disjoint(["a", "b"], ["b"])


def test_split_does_not_depend_on_the_hash_seed():
    code = (
        "import json;from edusync_c2_model.split import make_split, group_kfold;"
        "m={f'L{c}{k}':f'C{c}' for c in range(5) for k in range(3)};"
        "print(json.dumps([make_split(m,seed=11),group_kfold(m,4,11)],sort_keys=True))"
    )
    outputs = set()
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(SRC)}
        done = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(done.stdout)
    assert len(outputs) == 1


# ---------------------------------------------------------------------------------- metrics

Y_TRUE = [0, 0, 1, 1, 2, 2]
Y_PRED = [0, 1, 1, 1, 2, 1]


def test_hand_computed_metrics():
    # cm = [[1,1,0],[0,2,0],[0,1,1]]; weighted observed 1/12, weighted expected 1/4
    stats = metrics_from_confusion(confusion_matrix(Y_TRUE, Y_PRED))
    assert stats["n"] == 6
    assert stats["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 1, 1]]
    assert stats["qwk"] == pytest.approx(1 - (1 / 12) / (1 / 4))
    assert stats["mae"] == pytest.approx(2 / 6)
    assert stats["accuracy"] == pytest.approx(4 / 6)
    assert stats["adjacent_accuracy"] == pytest.approx(1.0)
    assert stats["balanced_accuracy"] == pytest.approx((0.5 + 1 + 0.5) / 3)
    low, medium, high = stats["per_class"]
    assert (low["precision"], low["recall"], low["support"]) == (1.0, 0.5, 2)
    assert (medium["precision"], medium["recall"]) == (0.5, 1.0)
    assert (high["precision"], high["recall"]) == (1.0, 0.5)
    assert low["f1"] == pytest.approx(2 / 3)
    assert medium["f1"] == pytest.approx(2 / 3)
    assert high["f1"] == pytest.approx(2 / 3)


def test_two_class_distance_is_penalised_quadratically():
    # one Low predicted High among 3 rows: adjacent accuracy 2/3, MAE 2/3
    stats = metrics_from_confusion(confusion_matrix([0, 1, 2], [0, 1, 0]))
    assert stats["mae"] == pytest.approx(2 / 3)
    assert stats["adjacent_accuracy"] == pytest.approx(2 / 3)
    # observed weight 1/3 (distance 2 -> w 1); expected weight (1/12 + 1/6 + 3/4) / 3 = 1/3
    assert stats["qwk"] == pytest.approx(0.0)


def test_perfect_prediction_has_qwk_one():
    stats = metrics_from_confusion(confusion_matrix([0, 1, 2, 2], [0, 1, 2, 2]))
    assert stats["qwk"] == pytest.approx(1.0)
    assert stats["mae"] == 0.0
    assert stats["balanced_accuracy"] == 1.0


def test_undefined_metrics_are_none():
    empty = metrics_from_confusion(confusion_matrix([], []))
    assert empty["n"] == 0
    for name in ("qwk", "mae", "accuracy", "adjacent_accuracy", "balanced_accuracy"):
        assert empty[name] is None
    assert all(c["recall"] is None and c["precision"] is None for c in empty["per_class"])

    one_class = metrics_from_confusion(confusion_matrix([0, 0], [0, 0]))
    assert one_class["qwk"] is None  # no chance disagreement
    assert one_class["accuracy"] == 1.0
    assert one_class["per_class"][1]["precision"] is None  # never predicted, never present
    assert one_class["per_class"][1]["recall"] is None

    swapped = metrics_from_confusion(confusion_matrix([0, 1], [1, 0]))
    assert swapped["per_class"][0]["precision"] == 0.0
    assert swapped["per_class"][0]["recall"] == 0.0
    assert swapped["per_class"][0]["f1"] is None  # 0 + 0


def test_confusion_matrix_validates_input():
    with pytest.raises(ValueError, match="same length"):
        confusion_matrix([0], [0, 1])
    with pytest.raises(ValueError, match="class ids"):
        confusion_matrix([3], [0])
    with pytest.raises(ValueError, match="class ids"):
        confusion_matrix([0], [-1])


def test_bootstrap_is_seeded_and_resamples_whole_lectures():
    truth = [0, 1, 2] * 8
    pred = [0, 1, 2, 0, 1, 1] * 4
    ids = [f"L{i // 3}" for i in range(24)]
    one = bootstrap_ci(truth, pred, ids, n_boot=200, seed=4)
    again = bootstrap_ci(truth, pred, ids, n_boot=200, seed=4)
    assert one == again
    other = bootstrap_ci(truth, pred, ids, n_boot=200, seed=5)
    assert one != other
    point = metrics_from_confusion(confusion_matrix(truth, pred))["accuracy"]
    assert one["accuracy"]["low"] <= point <= one["accuracy"]["high"]


def test_bootstrap_of_a_perfect_model_is_a_point_interval():
    truth = [0, 1, 2, 0, 1, 2]
    ci = bootstrap_ci(truth, truth, ["a", "a", "a", "b", "b", "b"], n_boot=50, seed=0)
    assert ci["mae"] == {"low": 0.0, "high": 0.0}


def test_bootstrap_with_undefined_metric_or_no_data_is_none():
    ci = bootstrap_ci([0, 0], [0, 0], ["a", "b"], n_boot=20, seed=0)
    assert ci["qwk"] is None
    assert ci["accuracy"] == {"low": 1.0, "high": 1.0}
    assert all(v is None for v in bootstrap_ci([], [], [], n_boot=20).values())
    assert all(v is None for v in bootstrap_ci([0], [0], ["a"], n_boot=0).values())
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ci([0], [0], [])


def test_evaluate_bundles_everything():
    result = evaluate(Y_TRUE, Y_PRED, ["a", "a", "b", "b", "c", "c"], n_boot=30, seed=1)
    assert result["n_lectures"] == 3
    assert result["n_boot"] == 30
    assert result["synthetic"] is False
    assert set(result["ci"]) == {"qwk", "mae", "accuracy", "adjacent_accuracy", "balanced_accuracy"}
    assert evaluate([0], [0], ["a"], n_boot=1, synthetic=True)["synthetic"] is True


# ----------------------------------------------------------------------------------- report


def example(synthetic):
    n = 35
    row = FeatureRow("L1", "C1", "a", "A", "t", (0.0,) * n, (0,) * n, ("computed",) * n)
    return Example.from_row(row, "Low", 1, synthetic)


def test_any_synthetic():
    assert any_synthetic([example(False), example(True)])
    assert not any_synthetic([example(False)])
    assert not any_synthetic([])


def test_report_has_the_banner_exactly_when_synthetic():
    stats = evaluate(Y_TRUE, Y_PRED, ["a", "a", "b", "b", "c", "c"], n_boot=20, synthetic=True)
    text = render_markdown(stats, "Run 1")
    assert text.startswith("# Run 1\n")
    assert SYNTHETIC_BANNER in text
    assert "| qwk |" in text
    assert "| Low | 1.000 | 0.500 | 0.667 | 2 |" in text
    assert "| Medium | 0 | 2 | 0 |" in text
    real = render_markdown(dict(stats, synthetic=False))
    assert SYNTHETIC_BANNER not in real
    missing_flag = dict(stats)
    del missing_flag["synthetic"]
    assert SYNTHETIC_BANNER not in render_markdown(missing_flag)


def test_report_renders_undefined_values_as_na():
    stats = evaluate([0, 0], [0, 0], ["a", "b"], n_boot=10)
    text = render_markdown(stats)
    assert "| qwk | n/a | n/a |" in text
    assert "| Medium | n/a | n/a | n/a | 0 |" in text
    assert SYNTHETIC_BANNER == "SYNTHETIC DATA: results test the plumbing only and are not accuracy"
