"""Tests for the opt-in layout variation of synthetic PPTX decks."""

import importlib.util
import random
from pathlib import Path

from edusync_c2.difficulty.materials import parse_materials
from edusync_c2.difficulty.synthetic import (
    _draw_layout,
    build_c1_graph,
    build_labels_csv,
    build_pptx,
    make_deck_spec,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
GEOMETRY_FIELDS = (
    "text_area_emu",
    "picture_area_emu",
    "min_font_pt",
    "diagram_element_count",
    "table_cell_count",
)


def _specs():
    return [make_deck_spec(42 + k, 8, lecture_id=f"synthetic-lecture-{k}") for k in range(3)]


def _slides(spec, **kwargs):
    return parse_materials(build_pptx(spec, **kwargs), "deck.pptx").slides


def _load(name):
    module_spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_default_is_deterministic_and_fixed_layout() -> None:
    for spec in _specs():
        first, second = _slides(spec), _slides(spec, vary_layout=False)
        assert first == second
        concept_slides = first[1:-1]
        # classic layout: one 6x5in textbox at 14pt, diagram elements only from the layout
        assert {s.min_font_pt for s in concept_slides if s.paragraphs} <= {14.0}
        assert {s.table_cell_count for s in concept_slides} <= {0, None}
        assert {s.diagram_element_count for s in concept_slides} <= {0, None}


def test_default_text_area_is_constant_across_concepts() -> None:
    areas = {
        s.text_area_emu for spec in _specs() for s in _slides(spec)[1:-1] if s.paragraphs and s.text
    }
    assert len(areas) == 1


def test_varied_is_deterministic_and_differs_from_default() -> None:
    spec = _specs()[0]
    assert _slides(spec, vary_layout=True) == _slides(spec, vary_layout=True)
    assert _slides(spec, vary_layout=True) != _slides(spec)


def test_varied_text_is_unchanged() -> None:
    spec = _specs()[0]
    default = [(s.title, s.text) for s in _slides(spec)]
    varied = [(s.title, s.text) for s in _slides(spec, vary_layout=True)]
    assert default == varied


def test_varied_makes_slide_fields_vary() -> None:
    slides = [s for spec in _specs() for s in _slides(spec, vary_layout=True)]
    for field in GEOMETRY_FIELDS:
        assert len({getattr(s, field) for s in slides}) > 1, field


def test_labels_and_graphs_ignore_the_flag() -> None:
    specs = _specs()
    labels = build_labels_csv(specs)
    graphs = [build_c1_graph(s, v) for s in specs for v in ("v0", "v1")]
    for spec in specs:
        build_pptx(spec, vary_layout=True)
    assert build_labels_csv(specs) == labels
    assert [build_c1_graph(s, v) for s in specs for v in ("v0", "v1")] == graphs


def test_script_refuses_default_dir(capsys) -> None:
    script = _load("make_synthetic_data")
    assert script.main(["--vary-layout"]) == 2
    assert "--out" in capsys.readouterr().err


def test_script_writes_varied_decks_to_explicit_dir(tmp_path: Path) -> None:
    script = _load("make_synthetic_data")
    plain, varied = tmp_path / "plain", tmp_path / "varied"
    base = ["--seed", "1", "--lectures", "1", "--concepts", "6"]
    assert script.main([*base, "--out", str(plain)]) == 0
    assert script.main([*base, "--vary-layout", "--out", str(varied)]) == 0
    for name in ("labels.csv", "lecture_1_graph_v1.json", "lecture_1.pdf"):
        assert (plain / name).read_bytes() == (varied / name).read_bytes()
    one = parse_materials((plain / "lecture_1.pptx").read_bytes(), "a.pptx").slides
    two = parse_materials((varied / "lecture_1.pptx").read_bytes(), "a.pptx").slides
    assert one != two


def test_feature_report_vary_layout_runs(capsys) -> None:
    report = _load("feature_report")
    rows, predictions = report.collect(42, 3, 8, vary_layout=True)
    assert predictions > 0
    assert rows
    assert report.main(["--vary-layout", "--lectures", "1", "--concepts", "6"]) == 0
    assert "Feature report" in capsys.readouterr().out


def test_rare_overflow_widens_the_text_box() -> None:
    class Overflowing(random.Random):
        def random(self) -> float:
            return 0.0

    layout = _draw_layout(Overflowing("x"))
    assert layout.text_box[2] == 6.0  # uniform low 3.0 plus the 3.0 overflow
    assert layout.table is not None
