"""Baselines on tiny sets with hand-computed metrics."""

import numpy as np
import pytest

from edusync_c2.difficulty.registry import FEATURE_NAMES
from edusync_c2_model.baselines import (
    BaselineResult,
    compare,
    fusion_input_from_example,
    human_ceiling,
    majority_class,
    run_logistic_baseline,
    run_majority_baseline,
    run_rule_baseline,
    score_predictions,
)
from edusync_c2_model.examples import Example
from edusync_c2_model.labels import LABEL_ORDER

N = len(FEATURE_NAMES)


def example(lecture, concept, label_id, computed=None, mocked=()):
    """An example whose named features are computed (everything else unavailable)."""
    computed = computed or {}
    values, missing, statuses = [0.0] * N, [1] * N, ["unavailable"] * N
    for name, value in computed.items():
        i = FEATURE_NAMES.index(name)
        values[i], missing[i], statuses[i] = value, 0, "computed"
    for name in mocked:
        statuses[FEATURE_NAMES.index(name)] = "mocked"
    return Example(
        lecture, "C1", concept, concept, "text", tuple(values), tuple(missing), tuple(statuses),
        LABEL_ORDER[label_id], label_id, 2, True,
    )  # fmt: skip


HIGH = {"lang.lexical_score": 1.0, "dens.words_per_slide": 120.0, "dens.formula_presence": 1.0}
LOW = {"lang.lexical_score": 0.0, "dens.words_per_slide": 0.0, "dens.formula_presence": 0.0}


def test_fusion_input_is_rebuilt_from_the_stored_vectors():
    e = example("L1", "k", 2, {"lang.lexical_score": 0.5}, mocked=("prereq.direct_count",))
    fusion = fusion_input_from_example(e)
    assert [f.name for f in fusion.features] == list(FEATURE_NAMES)
    lex = fusion.features[FEATURE_NAMES.index("lang.lexical_score")]
    assert (lex.value, lex.missing, lex.status.value) == (0.5, False, "computed")
    gone = fusion.features[FEATURE_NAMES.index("prereq.direct_count")]
    assert (gone.value, gone.missing, gone.status.value) == (None, True, "mocked")
    assert fusion.concept_id == "k"


def test_rule_baseline_counts_none_outputs_as_coverage_misses():
    examples = [
        example("L1", "hi", 2, HIGH),  # score 1.0 -> High
        example("L1", "lo", 1, LOW),  # score 0.0 -> Low (true Medium, one class off)
        example("L2", "none", 0, {"lang.lexical_score": 0.3}),  # too few components -> None
    ]
    result = run_rule_baseline(examples, n_boot=10)
    assert result.predictions == [2, 0, None]
    m = result.metrics
    assert m["coverage"] == pytest.approx(2 / 3)
    assert (m["n"], m["n_total"]) == (2, 3)
    assert m["accuracy"] == 0.5
    assert m["mae"] == 0.5
    assert m["adjacent_accuracy"] == 1.0
    assert m["synthetic"] is True


def test_majority_baseline_hand_computed():
    train = [example("L1", str(i), c) for i, c in enumerate([1, 1, 0])]
    test = [example("L2", str(i), c) for i, c in enumerate([0, 1, 2, 1])]
    assert majority_class(train) == 1
    result = run_majority_baseline(train, test, n_boot=10)
    assert result.predictions == [1, 1, 1, 1]
    m = result.metrics
    assert (m["accuracy"], m["mae"], m["adjacent_accuracy"]) == (0.5, 0.5, 1.0)
    assert m["balanced_accuracy"] == pytest.approx(1 / 3)
    assert m["qwk"] == 0.0
    assert m["coverage"] == 1.0


def test_majority_ties_go_to_the_lowest_class_and_empty_train_is_an_error():
    assert majority_class([example("L", "a", 2), example("L", "b", 1)]) == 1
    with pytest.raises(ValueError, match="empty"):
        majority_class([])


def test_feature_logistic_recovers_separable_features_and_handles_missing():
    def rows(prefix):
        out = []
        for i in range(30):
            c = i % 3
            feats = {"lang.lexical_score": float(c), "dens.words_per_slide": c * 40.0 + i % 2}
            if i % 5 == 0:
                feats = {"dens.words_per_slide": c * 40.0}  # lexical score missing
            out.append(example(f"{prefix}{i // 5}", f"c{i}", c, feats))
        return out

    result = run_logistic_baseline(rows("A"), rows("B"), l2=1e-3, n_boot=10)
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["qwk"] == 1.0
    assert result.metrics["coverage"] == 1.0
    with pytest.raises(ValueError, match="empty"):
        run_logistic_baseline([], rows("B"))


def test_score_predictions_with_no_examples_reports_no_coverage():
    result = score_predictions("none", [], [], n_boot=5)
    assert result.metrics["coverage"] is None
    assert result.metrics["qwk"] is None
    assert result.metrics["synthetic"] is False


def test_human_ceiling_is_the_qwk_between_the_first_two_raters(tmp_path):
    header = "lecture_id,concept_id,difficulty_label,rater_id,synthetic\n"
    rows = [
        "L1,a,Low,R1,true", "L1,a,Low,R2,true",
        "L1,b,Medium,R1,true", "L1,b,Medium,R2,true",
        "L1,c,High,R1,true", "L1,c,Medium,R2,true", "L1,c,Low,R3,true",
        "L1,d,High,ADJ,true", "L1,e,CannotJudge,R1,true", "L1,e,Low,R2,true",
        "L1,f,Low,R1,true", "L1,g,,R1,true", "L1,g,,R2,true",
    ]  # fmt: skip
    path = tmp_path / "labels.csv"
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    ceiling = human_ceiling(path)
    # pairs (a: 0,0) (b: 1,1) (c: 2,1): cm = [[1,0,0],[0,1,0],[0,1,0]]
    assert ceiling["n_pairs"] == 3
    assert ceiling["qwk"] == pytest.approx(2 / 3)
    assert ceiling["mae"] == pytest.approx(1 / 3)
    solo = tmp_path / "solo.csv"
    solo.write_text(header + "L1,a,Low,R1,true\n", encoding="utf-8")
    assert human_ceiling(solo) is None


def test_compare_renders_every_result_and_the_ceiling():
    examples = [example("L1", "hi", 2, HIGH), example("L1", "lo", 0, LOW)]
    rule = run_rule_baseline(examples, n_boot=5)
    table = compare(
        [rule, BaselineResult("empty", [], score_predictions("x", [], [], n_boot=2).metrics)],
        {"qwk": 0.5, "mae": 0.25, "n_pairs": 7},
    )
    lines = table.strip().splitlines()
    assert lines[0].startswith("| model | n | coverage | QWK |")
    assert lines[2].startswith("| rule baseline | 2 | 1.000 | 1.000 | 0.000 |")
    assert "| empty | 0 | n/a | n/a |" in lines[3]
    assert lines[4] == "| human-human (raters 1 and 2) | 7 | n/a | 0.500 | 0.250 | n/a | n/a |"
    assert len(compare([rule]).strip().splitlines()) == 3
    assert np.isclose(rule.metrics["qwk"], 1.0)
