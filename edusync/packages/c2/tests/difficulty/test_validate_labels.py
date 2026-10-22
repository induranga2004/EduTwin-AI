"""Tests for the validate_labels CLI."""

import csv
import importlib.util
import json
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_labels.py"
TEMPLATE = Path(__file__).resolve().parents[2] / "data" / "labels" / "labels_template.csv"

HEADER = [
    "lecture_id",
    "concept_id",
    "concept_slug",
    "concept_label",
    "slide_indexes",
    "difficulty_label",
    "rater_id",
    "student_level",
    "confidence",
    "notes",
    "labelled_at",
    "rubric_version",
    "synthetic",
]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("validate_labels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(**changes: str) -> dict[str, str]:
    base = {
        "lecture_id": "L001",
        "concept_id": "c1",
        "concept_slug": "c1",
        "concept_label": "C1",
        "slide_indexes": "1",
        "difficulty_label": "Medium",
        "rater_id": "R01",
        "student_level": "second_year",
        "confidence": "4",
        "notes": "",
        "labelled_at": "2026-02-03",
        "rubric_version": "r1",
        "synthetic": "false",
    }
    base.update(changes)
    return base


def write_csv(path: Path, rows: list[dict[str, str]], header: list[str] | None = None) -> Path:
    header = header or HEADER
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=header, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def run(script, path: Path, capsys, *extra: str) -> tuple[int, dict]:
    code = script.main(["--file", str(path), "--json", *extra])
    return code, json.loads(capsys.readouterr().out)


def rules(report: dict, severity: str | None = None) -> list[str]:
    return [f["rule"] for f in report["findings"] if severity is None or f["severity"] == severity]


def test_valid_file_passes(script, tmp_path: Path, capsys) -> None:
    path = write_csv(tmp_path / "l.csv", [row(), row(concept_id="c2", difficulty_label="High")])
    code, report = run(script, path, capsys)
    assert code == 0
    assert report["ok"] is True
    assert report["findings"] == []
    assert report["summary"]["rows"] == 2
    assert report["summary"]["lectures"] == 1
    assert report["summary"]["raters"] == 1
    assert report["summary"]["label_distribution"] == {
        "Low": 0,
        "Medium": 1,
        "High": 1,
        "CannotJudge": 0,
    }


def test_template_is_valid(script, capsys) -> None:
    code, report = run(script, TEMPLATE, capsys)
    assert code == 0
    assert report["summary"]["rows"] == 2
    assert rules(report, "warning") == []


def test_l001_missing_columns(script, tmp_path: Path, capsys) -> None:
    header = [h for h in HEADER if h not in ("rater_id", "confidence")]
    code, report = run(script, write_csv(tmp_path / "l.csv", [row()], header), capsys)
    assert code == 1
    assert rules(report) == ["L-001"]
    assert "rater_id" in report["findings"][0]["message"]
    assert "confidence" in report["findings"][0]["message"]


def test_l002_empty_ids(script, tmp_path: Path, capsys) -> None:
    path = write_csv(tmp_path / "l.csv", [row(concept_id=""), row(lecture_id="", concept_id="x")])
    code, report = run(script, path, capsys)
    assert code == 1
    assert rules(report, "error").count("L-002") == 2


@pytest.mark.parametrize("bad", ["low", "LOW", "Hard", "cannotjudge", "Cannot Judge"])
def test_l003_labels_are_case_sensitive(script, tmp_path: Path, capsys, bad: str) -> None:
    code, report = run(script, write_csv(tmp_path / "l.csv", [row(difficulty_label=bad)]), capsys)
    assert code == 1
    assert rules(report, "error") == ["L-003"]


@pytest.mark.parametrize("good", ["Low", "Medium", "High", "CannotJudge"])
def test_l003_accepts_the_four_labels(script, tmp_path: Path, capsys, good: str) -> None:
    code, _ = run(script, write_csv(tmp_path / "l.csv", [row(difficulty_label=good)]), capsys)
    assert code == 0


def test_l004_rater_required_on_labelled_rows_only(script, tmp_path: Path, capsys) -> None:
    path = write_csv(
        tmp_path / "l.csv",
        [row(rater_id=""), row(concept_id="c2", difficulty_label="", rater_id="")],
    )
    code, report = run(script, path, capsys)
    assert code == 1
    assert rules(report, "error") == ["L-004"]


@pytest.mark.parametrize("bad", ["", "freshman", "First_Year", "year1"])
def test_l005_student_level(script, tmp_path: Path, capsys, bad: str) -> None:
    code, report = run(script, write_csv(tmp_path / "l.csv", [row(student_level=bad)]), capsys)
    assert code == 1
    assert rules(report, "error") == ["L-005"]


@pytest.mark.parametrize(
    "level", ["first_year", "second_year", "third_year", "postgraduate", "mixed"]
)
def test_l005_accepts_levels(script, tmp_path: Path, capsys, level: str) -> None:
    code, _ = run(script, write_csv(tmp_path / "l.csv", [row(student_level=level)]), capsys)
    assert code == 0


@pytest.mark.parametrize("bad", ["0", "6", "3.5", "-1", "high", "1e1", "٣"])
def test_l006_confidence_bad(script, tmp_path: Path, capsys, bad: str) -> None:
    code, report = run(script, write_csv(tmp_path / "l.csv", [row(confidence=bad)]), capsys)
    assert code == 1
    assert rules(report, "error") == ["L-006"]


@pytest.mark.parametrize("good", ["", "1", "5"])
def test_l006_confidence_good(script, tmp_path: Path, capsys, good: str) -> None:
    code, _ = run(script, write_csv(tmp_path / "l.csv", [row(confidence=good)]), capsys)
    assert code == 0


@pytest.mark.parametrize("bad", ["", "03/02/2026", "2026-13-01", "yesterday", "2026-02-30"])
def test_l007_labelled_at(script, tmp_path: Path, capsys, bad: str) -> None:
    code, report = run(script, write_csv(tmp_path / "l.csv", [row(labelled_at=bad)]), capsys)
    assert code == 1
    assert rules(report, "error") == ["L-007"]


def test_l007_accepts_datetime(script, tmp_path: Path, capsys) -> None:
    path = write_csv(tmp_path / "l.csv", [row(labelled_at="2026-02-03T10:15:00")])
    assert run(script, path, capsys)[0] == 0


def test_l008_mixed_rubric_versions(script, tmp_path: Path, capsys) -> None:
    path = write_csv(tmp_path / "l.csv", [row(), row(concept_id="c2", rubric_version="r2")])
    code, report = run(script, path, capsys)
    assert code == 1
    assert rules(report, "error") == ["L-008"]
    assert "r1" in report["findings"][0]["message"]


def test_l008_empty_rubric_version(script, tmp_path: Path, capsys) -> None:
    code, report = run(script, write_csv(tmp_path / "l.csv", [row(rubric_version="")]), capsys)
    assert code == 1
    assert rules(report, "error") == ["L-008"]


@pytest.mark.parametrize("bad", ["", "True", "yes", "1"])
def test_l009_synthetic(script, tmp_path: Path, capsys, bad: str) -> None:
    code, report = run(script, write_csv(tmp_path / "l.csv", [row(synthetic=bad)]), capsys)
    assert code == 1
    assert rules(report, "error") == ["L-009"]


def test_l010_duplicates_per_rater_only(script, tmp_path: Path, capsys) -> None:
    path = write_csv(
        tmp_path / "l.csv",
        [row(), row(rater_id="R02"), row(difficulty_label="High")],
    )
    code, report = run(script, path, capsys)
    assert code == 1
    assert rules(report, "error") == ["L-010"]
    assert report["findings"][0]["location"] == "row 4"


def test_l011_sheet_membership(script, tmp_path: Path, capsys) -> None:
    sheet = write_csv(tmp_path / "sheet.csv", [row(difficulty_label="", rater_id="")])
    path = write_csv(
        tmp_path / "l.csv",
        [row(), row(concept_id="ghost"), row(lecture_id="L009", concept_id="c1")],
    )
    code, report = run(script, path, capsys, "--sheet", str(sheet))
    assert code == 1
    assert rules(report, "error") == ["L-011", "L-011"]


def test_sheet_ok(script, tmp_path: Path, capsys) -> None:
    sheet = write_csv(tmp_path / "sheet.csv", [row(difficulty_label="", rater_id="")])
    code, _ = run(script, write_csv(tmp_path / "l.csv", [row()]), capsys, "--sheet", str(sheet))
    assert code == 0


def test_bad_sheet_exits_two(script, tmp_path: Path, capsys) -> None:
    path = write_csv(tmp_path / "l.csv", [row()])
    assert script.main(["--file", str(path), "--sheet", str(tmp_path / "none.csv")]) == 2
    other = write_csv(tmp_path / "other.csv", [row()], ["a", "b"])
    assert script.main(["--file", str(path), "--sheet", str(other)]) == 2
    assert "needs lecture_id and concept_id" in capsys.readouterr().err


def test_l012_unlabelled_rows_warn(script, tmp_path: Path, capsys) -> None:
    path = write_csv(tmp_path / "l.csv", [row(), row(concept_id="c2", difficulty_label="")])
    code, report = run(script, path, capsys)
    assert code == 0
    assert rules(report, "warning") == ["L-012"]
    assert report["summary"]["rows"] == 2
    assert report["summary"]["labelled_rows"] == 1


def test_l013_mixed_synthetic_warns(script, tmp_path: Path, capsys) -> None:
    path = write_csv(tmp_path / "l.csv", [row(), row(concept_id="c2", synthetic="true")])
    code, report = run(script, path, capsys)
    assert code == 0
    assert rules(report, "warning") == ["L-013"]


def test_l014_non_pseudonymous_ids_warn(script, tmp_path: Path, capsys) -> None:
    path = write_csv(tmp_path / "l.csv", [row(lecture_id="Dr Smith CS101", rater_id="Jane Doe")])
    code, report = run(script, path, capsys)
    assert code == 0
    assert rules(report, "warning") == ["L-014", "L-014"]


def test_summary_and_cannot_judge_share(script, tmp_path: Path, capsys) -> None:
    rows = [
        row(concept_id="c1", difficulty_label="Low"),
        row(concept_id="c2", difficulty_label="CannotJudge"),
        row(concept_id="c3", difficulty_label="CannotJudge", rater_id="R02"),
        row(lecture_id="L002", concept_id="c1", difficulty_label="High", rater_id="R03"),
    ]
    code, report = run(script, write_csv(tmp_path / "l.csv", rows), capsys)
    assert code == 0
    summary = report["summary"]
    assert (summary["rows"], summary["lectures"], summary["raters"]) == (4, 2, 3)
    assert summary["label_distribution"] == {"Low": 1, "Medium": 0, "High": 1, "CannotJudge": 2}
    assert summary["cannot_judge_share"] == 0.5


def test_summary_without_labels(script, tmp_path: Path, capsys) -> None:
    path = write_csv(tmp_path / "l.csv", [row(difficulty_label="", rater_id="")])
    code, report = run(script, path, capsys)
    assert code == 0
    assert report["summary"]["cannot_judge_share"] is None


def test_human_report(script, tmp_path: Path, capsys) -> None:
    path = write_csv(
        tmp_path / "l.csv",
        [row(difficulty_label="nope"), row(concept_id="c2", difficulty_label="")],
    )
    assert script.main(["--file", str(path)]) == 1
    out = capsys.readouterr().out
    assert "Result: FAILED" in out
    assert "ERROR L-003 [row 2]" in out
    assert "WARNING L-012" in out
    assert out.index("ERROR") < out.index("WARNING")
    assert "1 error(s), 1 warning(s)" in out


def test_human_report_ok_has_no_agreement_block(script, tmp_path: Path, capsys) -> None:
    assert script.main(["--file", str(write_csv(tmp_path / "l.csv", [row()]))]) == 0
    out = capsys.readouterr().out
    assert "Result: ok" in out
    assert "Agreement" not in out


def test_blank_rows_and_bom_and_extra_cells(script, tmp_path: Path, capsys) -> None:
    path = tmp_path / "l.csv"
    line = ",".join(row().values())
    path.write_text(
        "﻿" + ",".join(HEADER) + "\n\n" + line + ",extra\n,,,,,,,,,,,,\n", encoding="utf-8"
    )
    code, report = run(script, path, capsys)
    assert code == 0
    assert report["summary"]["rows"] == 1


def test_short_row_is_checked_not_crashed(script, tmp_path: Path, capsys) -> None:
    path = tmp_path / "l.csv"
    path.write_text(",".join(HEADER) + "\nL001,c1\n", encoding="utf-8")
    code, report = run(script, path, capsys)
    assert code == 1
    assert "L-009" in rules(report)


def test_missing_file_exits_two(script, tmp_path: Path, capsys) -> None:
    assert script.main(["--file", str(tmp_path / "nope.csv")]) == 2
    assert "Cannot read" in capsys.readouterr().err


def test_empty_file_exits_two(script, tmp_path: Path, capsys) -> None:
    path = tmp_path / "l.csv"
    path.write_text("", encoding="utf-8")
    assert script.main(["--file", str(path)]) == 2
    assert "no header" in capsys.readouterr().err


def test_undecodable_file_exits_two(script, tmp_path: Path, capsys) -> None:
    path = tmp_path / "l.csv"
    path.write_bytes(b"\xff\xfe\x00bad\xff")
    assert script.main(["--file", str(path)]) == 2


def test_file_is_required_and_help(script) -> None:
    assert script.main([]) == 2
    assert script.main(["--help"]) == 0


def test_non_ascii_output_is_utf8(script, tmp_path: Path, capsys) -> None:
    name = "Été über"
    sheet = write_csv(tmp_path / "sheet.csv", [row()])
    path = write_csv(tmp_path / "l.csv", [row(concept_id=name)])
    assert script.main(["--file", str(path), "--sheet", str(sheet)]) == 1
    assert name in capsys.readouterr().out


def test_runs_as_main(tmp_path: Path, monkeypatch) -> None:
    path = write_csv(tmp_path / "l.csv", [row()])
    monkeypatch.setattr("sys.argv", ["validate_labels.py", "--file", str(path)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 0


# --- agreement -------------------------------------------------------------------------------


def kappa(script, pairs):
    return script.quadratic_weighted_kappa(pairs)


def test_qwk_hand_computed_example(script) -> None:
    # observed (0,0) (1,1) (2,2) (0,1), each 0.25; disagreement 0.0625 of expected 0.3125 -> 0.8
    assert kappa(script, [(0, 0), (1, 1), (2, 2), (0, 1)]) == 0.8


def test_qwk_perfect_and_opposite(script) -> None:
    assert kappa(script, [(0, 0), (1, 1), (2, 2)]) == 1.0
    assert kappa(script, [(0, 2), (2, 0)]) == -1.0


def test_qwk_undefined_cases(script) -> None:
    assert kappa(script, []) is None
    assert kappa(script, [(1, 1), (1, 1)]) is None


def test_qwk_penalises_far_errors_more(script) -> None:
    near = kappa(script, [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)])
    far = kappa(script, [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0)])
    assert far < near


def test_agreement_cli_hand_computed(script, tmp_path: Path, capsys) -> None:
    r1 = ["Low", "Medium", "High", "Low"]
    r2 = ["Low", "Medium", "High", "Medium"]
    rows = []
    for i, (a, b) in enumerate(zip(r1, r2, strict=True)):
        rows.append(row(concept_id=f"c{i}", difficulty_label=a, rater_id="R01"))
        rows.append(row(concept_id=f"c{i}", difficulty_label=b, rater_id="R02"))
    code, report = run(script, write_csv(tmp_path / "l.csv", rows), capsys, "--agreement")
    assert code == 0
    agreement = report["agreement"]
    assert agreement["kappa_bar"] == 0.6
    assert agreement["lectures"] == [
        {
            "lecture_id": "L001",
            "raters": ["R01", "R02"],
            "pairs": 4,
            "exact_agreement": 0.75,
            "qwk": 0.8,
        }
    ]
    assert agreement["pooled"] == {"pairs": 4, "exact_agreement": 0.75, "qwk": 0.8}


def test_agreement_excludes_cannot_judge_pairwise(script, tmp_path: Path, capsys) -> None:
    rows = [
        row(concept_id="c0", difficulty_label="Low", rater_id="R01"),
        row(concept_id="c0", difficulty_label="Low", rater_id="R02"),
        row(concept_id="c1", difficulty_label="CannotJudge", rater_id="R01"),
        row(concept_id="c1", difficulty_label="High", rater_id="R02"),
        row(concept_id="c2", difficulty_label="High", rater_id="R01"),
        row(concept_id="c2", difficulty_label="CannotJudge", rater_id="R02"),
        row(concept_id="c3", difficulty_label="High", rater_id="R01"),
        row(concept_id="c3", difficulty_label="High", rater_id="R02"),
        row(concept_id="c4", difficulty_label="Low", rater_id="R01"),
    ]
    _, report = run(script, write_csv(tmp_path / "l.csv", rows), capsys, "--agreement")
    pooled = report["agreement"]["pooled"]
    assert pooled["pairs"] == 2
    assert pooled["exact_agreement"] == 1.0
    assert pooled["qwk"] == 1.0


def test_agreement_uses_first_two_raters_sorted(script, tmp_path: Path, capsys) -> None:
    rows = []
    for rater, label in (("R03", "High"), ("R01", "Low"), ("R02", "Low")):
        rows.append(row(concept_id="c0", difficulty_label=label, rater_id=rater))
        rows.append(row(concept_id="c1", difficulty_label="High", rater_id=rater))
    _, report = run(script, write_csv(tmp_path / "l.csv", rows), capsys, "--agreement")
    lecture = report["agreement"]["lectures"][0]
    assert lecture["raters"] == ["R01", "R02"]
    assert lecture["exact_agreement"] == 1.0


def test_agreement_ignores_the_adjudicator(script, tmp_path: Path, capsys) -> None:
    rows = [
        row(difficulty_label="Low", rater_id="R01"),
        row(difficulty_label="High", rater_id="R02"),
        row(difficulty_label="Medium", rater_id="ADJ", notes="discussion"),
    ]
    code, report = run(script, write_csv(tmp_path / "l.csv", rows), capsys, "--agreement")
    assert code == 0
    assert report["agreement"]["lectures"][0]["raters"] == ["R01", "R02"]


def test_agreement_skips_single_rater_lectures(script, tmp_path: Path, capsys) -> None:
    rows = [
        row(lecture_id="L001", concept_id="c0", rater_id="R01"),
        row(lecture_id="L002", concept_id="c0", rater_id="R01"),
        row(lecture_id="L002", concept_id="c0", rater_id="R02"),
    ]
    _, report = run(script, write_csv(tmp_path / "l.csv", rows), capsys, "--agreement")
    assert [x["lecture_id"] for x in report["agreement"]["lectures"]] == ["L002"]


def test_agreement_pools_across_lectures(script, tmp_path: Path, capsys) -> None:
    rows = []
    for lecture, (a, b) in (("L001", ("Low", "Low")), ("L002", ("High", "Low"))):
        rows.append(row(lecture_id=lecture, difficulty_label=a, rater_id="R01"))
        rows.append(row(lecture_id=lecture, difficulty_label=b, rater_id="R02"))
    _, report = run(script, write_csv(tmp_path / "l.csv", rows), capsys, "--agreement")
    assert report["agreement"]["pooled"]["pairs"] == 2
    assert report["agreement"]["pooled"]["exact_agreement"] == 0.5
    assert [x["pairs"] for x in report["agreement"]["lectures"]] == [1, 1]


def test_agreement_ignores_rows_without_rater(script, tmp_path: Path, capsys) -> None:
    rows = [row(rater_id=""), row(rater_id="R01")]
    _, report = run(script, write_csv(tmp_path / "l.csv", rows), capsys, "--agreement")
    assert report["agreement"]["lectures"] == []


def test_agreement_does_not_change_exit_code(script, tmp_path: Path, capsys) -> None:
    rows = []
    for rater, label in (
        ("R01", "Low"),
        ("R02", "High"),
    ):
        rows.append(row(concept_id="c0", difficulty_label=label, rater_id=rater))
        rows.append(row(concept_id="c1", difficulty_label="Medium", rater_id=rater))
    code, report = run(script, write_csv(tmp_path / "l.csv", rows), capsys, "--agreement")
    assert code == 0
    assert report["agreement"]["pooled"]["exact_agreement"] == 0.5


def test_agreement_human_report(script, tmp_path: Path, capsys) -> None:
    rows = [
        row(concept_id="c0", difficulty_label="Low", rater_id="R01"),
        row(concept_id="c0", difficulty_label="Low", rater_id="R02"),
        row(concept_id="c1", difficulty_label="High", rater_id="R01"),
        row(concept_id="c1", difficulty_label="Low", rater_id="R02"),
    ]
    assert script.main(["--file", str(write_csv(tmp_path / "l.csv", rows)), "--agreement"]) == 0
    out = capsys.readouterr().out
    assert "L001 R01/R02: pairs=2 exact=0.5000" in out
    assert "protocol bar: qwk >= 0.6" in out


def test_agreement_human_report_without_second_rater(script, tmp_path: Path, capsys) -> None:
    assert script.main(["--file", str(write_csv(tmp_path / "l.csv", [row()])), "--agreement"]) == 0
    assert "no lecture has two or more raters" in capsys.readouterr().out


def test_agreement_human_report_undefined_kappa(script, tmp_path: Path, capsys) -> None:
    rows = [row(rater_id="R01"), row(rater_id="R02")]
    assert script.main(["--file", str(write_csv(tmp_path / "l.csv", rows)), "--agreement"]) == 0
    assert "qwk=n/a" in capsys.readouterr().out
