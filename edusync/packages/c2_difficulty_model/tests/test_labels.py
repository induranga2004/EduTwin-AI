"""Label resolution: adjudicator, ordinal median, ties, CannotJudge, file reading."""

from pathlib import Path

import pytest

from edusync_c2.difficulty.models import LABEL_ORDER as C2_LABEL_ORDER
from edusync_c2_model.labels import (
    LABEL_ORDER,
    LabelError,
    read_labels,
    read_rows,
    resolve_labels,
)


def row(lecture="L1", concept="a", label="Low", rater="R1", synthetic="false"):
    return {
        "lecture_id": lecture,
        "concept_id": concept,
        "difficulty_label": label,
        "rater_id": rater,
        "synthetic": synthetic,
    }


def test_label_order_equals_c2_order():
    assert LABEL_ORDER == C2_LABEL_ORDER == ("Low", "Medium", "High")


def test_single_rater_label_is_kept():
    result = resolve_labels([row(label="Medium")])
    label = result.labels[("L1", "a")]
    assert (label.label, label.label_id, label.n_raters, label.source) == (
        "Medium",
        1,
        1,
        "median",
    )
    assert label.raw_labels == ("Medium",)
    assert len(result) == 1


def test_adjudicator_wins_over_raters():
    rows = [
        row(label="Low", rater="R1"),
        row(label="High", rater="R2"),
        row(label="Medium", rater="ADJ"),
    ]
    label = resolve_labels(rows).labels[("L1", "a")]
    assert (label.label, label.source, label.n_raters) == ("Medium", "adjudicated", 2)
    assert label.raw_labels == ("Low", "High")


def test_adjudicator_alone_has_zero_raters():
    label = resolve_labels([row(label="High", rater="ADJ")]).labels[("L1", "a")]
    assert (label.label, label.n_raters, label.raw_labels) == ("High", 0, ())


def test_odd_count_uses_the_median():
    rows = [row(label=lab, rater=f"R{i}") for i, lab in enumerate(["High", "Low", "Medium"])]
    assert resolve_labels(rows).labels[("L1", "a")].label == "Medium"


def test_even_count_with_equal_middle_labels_resolves():
    labels = ["Low", "Medium", "Medium", "High"]
    rows = [row(label=lab, rater=f"R{i}") for i, lab in enumerate(labels)]
    result = resolve_labels(rows)
    assert result.labels[("L1", "a")].label == "Medium"
    assert result.ties_excluded == 0


def test_unresolvable_tie_is_excluded_and_counted():
    rows = [row(label="Low", rater="R1"), row(label="High", rater="R2")]
    result = resolve_labels(rows)
    assert result.labels == {}
    assert result.ties_excluded == 1


def test_adjacent_disagreement_of_two_raters_is_a_tie():
    rows = [row(label="Low", rater="R1"), row(label="Medium", rater="R2")]
    assert resolve_labels(rows).ties_excluded == 1


def test_cannot_judge_rows_are_dropped_from_the_median():
    rows = [
        row(label="Low", rater="R1"),
        row(label="CannotJudge", rater="R2"),
        row(label="Low", rater="R3"),
    ]
    label = resolve_labels(rows).labels[("L1", "a")]
    assert (label.label, label.n_raters, label.raw_labels) == ("Low", 2, ("Low", "Low"))


def test_concept_with_only_cannot_judge_is_dropped():
    result = resolve_labels([row(label="CannotJudge")])
    assert result.labels == {}
    assert result.cannot_judge_dropped == 1


def test_adjudicated_cannot_judge_drops_the_concept():
    rows = [row(label="Low", rater="R1"), row(label="CannotJudge", rater="ADJ")]
    result = resolve_labels(rows)
    assert result.labels == {}
    assert result.cannot_judge_dropped == 1


def test_unlabelled_rows_are_counted_and_skipped():
    result = resolve_labels([row(label=""), row(concept="b", label="High")])
    assert result.unlabelled_rows == 1
    assert list(result.labels) == [("L1", "b")]


def test_synthetic_flag_is_true_if_any_row_is_synthetic():
    rows = [row(rater="R1", synthetic="false"), row(rater="R2", synthetic="true")]
    assert resolve_labels(rows).labels[("L1", "a")].synthetic is True
    assert resolve_labels([row()]).labels[("L1", "a")].synthetic is False


def test_row_order_does_not_change_the_result():
    rows = [
        row(concept=c, label=lab, rater=r)
        for c in "abc"
        for r, lab in (("R1", "Low"), ("R2", "Medium"), ("R3", "High"))
    ]
    forward = resolve_labels(rows)
    backward = resolve_labels(list(reversed(rows)))
    assert forward.labels == backward.labels


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (row(label="Easy"), "unknown difficulty_label"),
        (row(rater=""), "must not be empty"),
        (row(lecture=""), "must not be empty"),
        (row(concept=""), "must not be empty"),
        (row(synthetic="yes"), "synthetic must be"),
    ],
)
def test_invalid_rows_raise(bad, message):
    with pytest.raises(LabelError, match=message):
        resolve_labels([bad])


def test_duplicate_rater_row_raises():
    with pytest.raises(LabelError, match="duplicate"):
        resolve_labels([row(), row(label="High")])


HEADER = "lecture_id,concept_id,difficulty_label,rater_id,synthetic,notes\n"


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_text(text, encoding=encoding, newline="")
    return path


def test_read_rows_strips_skips_blank_rows_and_handles_bom(tmp_path):
    path = write(
        tmp_path / "l.csv",
        HEADER + " L1 , a ,Low, R1 ,false,hi\n,,,,,\nL1,b,High,R1,false,extra,spill\n",
        "utf-8-sig",
    )
    rows = read_rows(path)
    assert [r["concept_id"] for r in rows] == ["a", "b"]
    assert rows[0]["rater_id"] == "R1"


def test_read_rows_requires_columns(tmp_path):
    path = write(tmp_path / "l.csv", "lecture_id,concept_id\nL1,a\n")
    with pytest.raises(LabelError, match="missing column"):
        read_rows(path)


def test_read_rows_reports_unreadable_files(tmp_path):
    with pytest.raises(LabelError, match="cannot read"):
        read_rows(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(LabelError, match="cannot read"):
        read_rows(bad)


def test_read_labels_combines_files(tmp_path):
    one = write(tmp_path / "1.csv", HEADER + "L1,a,Low,R1,false,\n")
    two = write(tmp_path / "2.csv", HEADER + "L2,a,High,R1,false,\n")
    result = read_labels([one, two])
    assert sorted(result.labels) == [("L1", "a"), ("L2", "a")]
