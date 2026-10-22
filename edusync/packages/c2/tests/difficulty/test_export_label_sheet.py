"""Tests for the export_label_sheet CLI."""

import csv
import importlib.util
import json
import runpy
from pathlib import Path

import pytest

from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.materials import parse_materials
from edusync_c2.difficulty.schemas import C1GraphResponse
from edusync_c2.difficulty.synthetic import build_c1_graph, build_pdf, build_pptx, make_deck_spec

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_label_sheet.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("export_label_sheet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def deck_spec():
    return make_deck_spec(7, 6, lecture_id="synthetic-lecture-7")


@pytest.fixture
def pptx_path(tmp_path: Path, deck_spec) -> Path:
    path = tmp_path / "lecture.pptx"
    path.write_bytes(build_pptx(deck_spec))
    return path


def read_sheet(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_columns_and_empty_rater_fields(script, pptx_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "sheet.csv"
    assert script.main(["--file", str(pptx_path), "--lecture-id", "L001", "--out", str(out)]) == 0
    rows = read_sheet(out)
    assert rows
    assert tuple(rows[0]) == script.COLUMNS
    for row in rows:
        assert row["lecture_id"] == "L001"
        assert row["rubric_version"] == "r1"
        assert row["synthetic"] == "false"
        for column in (
            "difficulty_label",
            "rater_id",
            "student_level",
            "confidence",
            "notes",
            "labelled_at",
        ):
            assert row[column] == ""


def test_concept_ids_are_exactly_what_c2_emits(
    script, pptx_path: Path, tmp_path: Path, deck_spec
) -> None:
    out = tmp_path / "sheet.csv"
    script.main(["--file", str(pptx_path), "--lecture-id", "L001", "--out", str(out)])
    response = DifficultyPipeline().run(pptx_path.read_bytes(), pptx_path.name)
    emitted = {p.concept_id: p for p in response.predictions}
    rows = read_sheet(out)
    assert {r["concept_id"] for r in rows} == set(emitted)
    for row in rows:
        prediction = emitted[row["concept_id"]]
        assert row["concept_label"] == prediction.label
        assert row["slide_indexes"] == ";".join(str(i) for i in sorted(prediction.slide_indexes))


def test_graph_option_uses_c1_ids(script, pptx_path: Path, tmp_path: Path, deck_spec) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(build_c1_graph(deck_spec, "v1")), encoding="utf-8")
    out = tmp_path / "sheet.csv"
    code = script.main(
        [
            "--file",
            str(pptx_path),
            "--lecture-id",
            "L002",
            "--graph",
            str(graph_path),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    graph = C1GraphResponse.model_validate(build_c1_graph(deck_spec, "v1"))
    response = DifficultyPipeline().run(pptx_path.read_bytes(), pptx_path.name, graph)
    assert [r["concept_id"] for r in read_sheet(out)] == [
        p.concept_id
        for p in sorted(
            response.predictions,
            key=lambda p: (min(p.slide_indexes) if p.slide_indexes else 10**9, p.concept_id),
        )
    ]


def test_slug_column_is_the_hyphen_slug(script, pptx_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "sheet.csv"
    script.main(["--file", str(pptx_path), "--lecture-id", "L001", "--out", str(out)])
    for row in read_sheet(out):
        assert row["concept_slug"] == row["concept_label"].casefold().replace(" ", "-")


def test_deterministic_bytes_and_order(script, pptx_path: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    script.main(["--file", str(pptx_path), "--lecture-id", "L001", "--out", str(first)])
    script.main(["--file", str(pptx_path), "--lecture-id", "L001", "--out", str(second)])
    assert first.read_bytes() == second.read_bytes()
    assert b"\r" not in first.read_bytes()
    rows = read_sheet(first)
    firsts = [int(r["slide_indexes"].split(";")[0]) for r in rows if r["slide_indexes"]]
    assert firsts == sorted(firsts)


def test_rubric_version_option(script, pptx_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "sheet.csv"
    code = script.main(
        [
            "--file",
            str(pptx_path),
            "--lecture-id",
            "L001",
            "--rubric-version",
            "r2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert {r["rubric_version"] for r in read_sheet(out)} == {"r2"}


@pytest.mark.parametrize("bad", ["", "my lecture", "a/b", "a\\b", "../x", "-x", "L" * 40, "Ü1"])
def test_bad_lecture_ids_are_rejected(script, pptx_path: Path, tmp_path: Path, bad: str) -> None:
    out = tmp_path / "sheet.csv"
    assert script.main(["--file", str(pptx_path), "--lecture-id", bad, "--out", str(out)]) == 2
    assert not out.exists()


def test_lecture_id_is_required(script, pptx_path: Path) -> None:
    assert script.main(["--file", str(pptx_path)]) == 2


def test_help_exits_zero(script) -> None:
    assert script.main(["--help"]) == 0


@pytest.mark.parametrize("bad", ["", "  ", "r,1", "r\n1"])
def test_bad_rubric_version_is_rejected(script, pptx_path: Path, tmp_path: Path, bad: str) -> None:
    out = tmp_path / "sheet.csv"
    code = script.main(
        [
            "--file",
            str(pptx_path),
            "--lecture-id",
            "L001",
            "--rubric-version",
            bad,
            "--out",
            str(out),
        ]
    )
    assert code == 2


def test_never_writes_slide_text(script, pptx_path: Path, tmp_path: Path, deck_spec) -> None:
    out = tmp_path / "sheet.csv"
    script.main(["--file", str(pptx_path), "--lecture-id", "L001", "--out", str(out)])
    sheet = out.read_text(encoding="utf-8")
    materials = parse_materials(pptx_path.read_bytes(), pptx_path.name)
    body_lines = {
        p.text for slide in materials.slides for p in slide.paragraphs if len(p.text) > 20
    }
    assert body_lines
    for line in body_lines:
        assert line not in sheet


def test_refuses_to_overwrite_without_force(
    script, pptx_path: Path, tmp_path: Path, capsys
) -> None:
    out = tmp_path / "sheet.csv"
    out.write_text("precious", encoding="utf-8")
    args = ["--file", str(pptx_path), "--lecture-id", "L001", "--out", str(out)]
    assert script.main(args) == 2
    assert out.read_text(encoding="utf-8") == "precious"
    assert "already exists" in capsys.readouterr().err
    assert script.main([*args, "--force"]) == 0
    assert out.read_text(encoding="utf-8").startswith("lecture_id,")


def test_default_out_name(script, pptx_path: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert script.main(["--file", str(pptx_path), "--lecture-id", "L009"]) == 0
    assert (tmp_path / "L009_label_sheet.csv").is_file()


def test_pdf_input_works(script, tmp_path: Path, deck_spec) -> None:
    pdf = tmp_path / "lecture.pdf"
    pdf.write_bytes(build_pdf(deck_spec))
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(build_c1_graph(deck_spec, "v1")), encoding="utf-8")
    out = tmp_path / "sheet.csv"
    args = ["--file", str(pdf), "--lecture-id", "L001", "--out", str(out)]
    assert script.main([*args, "--graph", str(graph)]) == 0
    assert read_sheet(out)


def test_no_concepts_exits_two_without_writing(script, tmp_path: Path, deck_spec, capsys) -> None:
    pdf = tmp_path / "lecture.pdf"  # a PDF has no slide titles, so the mock graph finds nothing
    pdf.write_bytes(build_pdf(deck_spec))
    out = tmp_path / "sheet.csv"
    assert script.main(["--file", str(pdf), "--lecture-id", "L001", "--out", str(out)]) == 2
    assert not out.exists()
    assert "--graph" in capsys.readouterr().err


def test_missing_lecture_file_exits_two(script, tmp_path: Path, capsys) -> None:
    code = script.main(["--file", str(tmp_path / "nope.pptx"), "--lecture-id", "L001"])
    assert code == 2
    assert "Cannot read" in capsys.readouterr().err


def test_rejected_lecture_file_exits_two(script, tmp_path: Path, capsys) -> None:
    bad = tmp_path / "lecture.txt"
    bad.write_text("hello", encoding="utf-8")
    code = script.main(
        ["--file", str(bad), "--lecture-id", "L001", "--out", str(tmp_path / "s.csv")]
    )
    assert code == 2
    assert "rejected" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["{not json", '{"graph": 3}', "[]"])
def test_bad_graph_exits_two(script, pptx_path: Path, tmp_path: Path, capsys, content: str) -> None:
    graph = tmp_path / "graph.json"
    graph.write_text(content, encoding="utf-8")
    code = script.main(
        [
            "--file",
            str(pptx_path),
            "--lecture-id",
            "L001",
            "--graph",
            str(graph),
            "--out",
            str(tmp_path / "s.csv"),
        ]
    )
    assert code == 2
    assert "Cannot use graph" in capsys.readouterr().err


def test_missing_graph_exits_two(script, pptx_path: Path, tmp_path: Path) -> None:
    code = script.main(
        ["--file", str(pptx_path), "--lecture-id", "L001", "--graph", str(tmp_path / "no.json")]
    )
    assert code == 2


def test_unwritable_output_exits_two(script, pptx_path: Path, tmp_path: Path, capsys) -> None:
    code = script.main(
        [
            "--file",
            str(pptx_path),
            "--lecture-id",
            "L001",
            "--out",
            str(tmp_path / "missing" / "s.csv"),
        ]
    )
    assert code == 2
    assert "Cannot write" in capsys.readouterr().err


def test_prediction_without_slides_sorts_last(script) -> None:
    class P:
        def __init__(self, cid: str, slides: list[int]) -> None:
            self.concept_id, self.label, self.slide_indexes = cid, cid.title(), slides

    rows = script.build_rows("L001", "r1", [P("b", []), P("a", [3, 3, 2]), P("c", [1])])
    assert [r[1] for r in rows] == ["c", "a", "b"]
    assert rows[1][4] == "2;3"
    assert rows[2][4] == ""


def test_runs_as_main(pptx_path: Path, tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "sheet.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_label_sheet.py",
            "--file",
            str(pptx_path),
            "--lecture-id",
            "L001",
            "--out",
            str(out),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 0
    assert out.is_file()
