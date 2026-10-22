import numpy as np
import pytest

from edusync_blueprint_model import deps
from edusync_blueprint_model.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    macro_f1,
    summarize,
)
from edusync_blueprint_model.train import TrainResult, iter_batches, linear_warmup_decay

Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0])


def test_accuracy() -> None:
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)
    assert accuracy(np.array([]), np.array([])) == 0.0


def test_confusion_matrix() -> None:
    cm = confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_macro_f1_and_report() -> None:
    report = classification_report(Y_TRUE, Y_PRED, ["a", "b", "c"])
    assert report["a"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    assert report["b"]["precision"] == pytest.approx(2 / 3)
    assert report["b"]["recall"] == 1.0
    assert report["c"]["precision"] == 1.0
    assert report["c"]["recall"] == 0.5
    expected = (0.5 + 0.8 + 2 / 3) / 3
    assert macro_f1(Y_TRUE, Y_PRED, 3) == pytest.approx(expected)


def test_macro_f1_handles_absent_class() -> None:
    assert macro_f1(np.array([0, 0]), np.array([0, 0]), 2) == pytest.approx(0.5)


def test_summarize() -> None:
    s = summarize(Y_TRUE, Y_PRED, ["a", "b", "c"])
    assert s["num_examples"] == 6
    assert s["labels"] == ["a", "b", "c"]
    assert s["confusion_matrix"][1] == [0, 2, 0]
    assert s["accuracy"] == pytest.approx(4 / 6)


def test_require_missing_module() -> None:
    with pytest.raises(deps.MissingExtraError, match="--extra finetune"):
        deps.require("definitely_not_a_module_xyz", "finetune")
    assert issubclass(deps.MissingExtraError, ImportError)


def test_require_present_and_is_available() -> None:
    assert deps.require("json", "x").__name__ == "json"
    assert deps.is_available("json")
    assert not deps.is_available("definitely_not_a_module_xyz")
    assert "--package edusync-blueprint-model --extra onnx" in deps.install_command("onnx")


def test_linear_warmup_decay() -> None:
    assert linear_warmup_decay(0, 100, 10) == 0.0
    assert linear_warmup_decay(5, 100, 10) == 0.5
    assert linear_warmup_decay(10, 100, 10) == 1.0
    assert linear_warmup_decay(55, 100, 10) == pytest.approx(0.5)
    assert linear_warmup_decay(100, 100, 10) == 0.0
    assert linear_warmup_decay(200, 100, 10) == 0.0
    assert linear_warmup_decay(0, 100, 0) == 1.0


def test_iter_batches() -> None:
    batches = list(iter_batches(10, 4, seed=1))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(np.concatenate(batches).tolist()) == list(range(10))
    again = list(iter_batches(10, 4, seed=1))
    assert all((a == b).all() for a, b in zip(batches, again, strict=True))
    ordered = list(iter_batches(5, 2, shuffle=False))
    assert [b.tolist() for b in ordered] == [[0, 1], [2, 3], [4]]


def test_train_result_serializes(tmp_path) -> None:
    r = TrainResult(
        best_epoch=1,
        best_val_macro_f1=0.5,
        test={"accuracy": 1.0},
        history=[],
        finetuned_dir=tmp_path,
    )
    assert "best_epoch" in r.model_dump_json()


def test_no_decay_names_and_model_module_imports_without_torch() -> None:
    from edusync_blueprint_model import model, train

    assert train._no_decay("encoder.layer.0.attention.output.LayerNorm.weight")
    assert train._no_decay("head.bias")
    assert not train._no_decay("head.weight")
    assert callable(model.build_classifier)


def test_selection_score_prefers_positive_recall_then_macro_f1() -> None:
    from edusync_blueprint_model.metrics import selection_score, summarize

    labels = ["0", "1"]
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    high_recall = summarize(y_true, np.array([1, 1, 0, 0, 1, 1, 1, 1]), labels)  # all 1s, 2 FP
    high_f1 = summarize(y_true, np.array([0, 0, 0, 0, 1, 1, 1, 0]), labels)  # misses a 1, no FP
    assert selection_score(high_recall, "positive_recall", "1") > selection_score(
        high_f1, "positive_recall", "1"
    )
    assert selection_score(high_f1, "macro_f1", None) > selection_score(
        high_recall, "macro_f1", None
    )
    same = summarize(y_true, np.array([0, 0, 0, 0, 1, 1, 1, 1]), labels)
    assert selection_score(same, "positive_recall", "1") > selection_score(
        high_recall, "positive_recall", "1"
    )  # equal recall, better macro-F1 wins
    with pytest.raises(ValueError, match="not in"):
        selection_score(same, "positive_recall", "x")
    with pytest.raises(ValueError, match="Unknown"):
        selection_score(same, "nope", None)


def test_selection_score_recall_floor() -> None:
    from edusync_blueprint_model.metrics import selection_score, summarize

    labels = ["0", "1"]
    y = np.array([0] * 10 + [1] * 100)
    # A: recall 1.00 but 3 false positives (lower macro-F1); B: recall 0.99, 1 FP (higher macro-F1)
    a_pred = np.array([1, 1, 1] + [0] * 7 + [1] * 100)
    b_pred = np.array([1] + [0] * 9 + [1] * 99 + [0])
    c_pred = np.array([0] * 10 + [1] * 90 + [0] * 10)  # recall 0.90, perfect REMOVE
    a, b, c = (summarize(y, p, labels) for p in (a_pred, b_pred, c_pred))
    score = lambda s: selection_score(s, "macro_f1_with_recall_floor", "1", 0.99)  # noqa: E731
    assert b["macro_f1"] > a["macro_f1"]
    assert score(b) > score(a)  # both meet the floor, so the higher macro-F1 wins
    assert score(a) > score(c)  # below the floor always loses
    assert score(b) > score(c)
    # raw positive_recall would still prefer A
    assert selection_score(a, "positive_recall", "1") > selection_score(b, "positive_recall", "1")
