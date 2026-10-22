"""Tests for the two concept-id conventions and the code that shares them."""

import json
from pathlib import Path

import pytest

from edusync_c2.difficulty import providers, synthetic
from edusync_c2.difficulty.concepts import normalize
from edusync_c2.difficulty.ids import casefold_id, slug_id

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_synthetic_data.py"
COMMITTED = Path(__file__).resolve().parents[2] / "data" / "synthetic"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  Gradient \t DESCENT\n", "gradient descent"),
        ("TCP/IP", "tcp/ip"),
        ("", ""),
        ("Stra\u00dfe", "strasse"),
    ],
)
def test_casefold_id(text: str, expected: str) -> None:
    assert casefold_id(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Gradient Descent", "gradient-descent"),
        ("TCP/IP", "tcp-ip"),
        ("  --Big  O (notation)!  ", "big-o-notation"),
        ("k-means", "k-means"),
        ("", ""),
        ("!!!", ""),
    ],
)
def test_slug_id(text: str, expected: str) -> None:
    assert slug_id(text) == expected


def test_slug_id_is_idempotent() -> None:
    assert slug_id(slug_id("A  B/C")) == slug_id("A  B/C")


def test_old_names_delegate() -> None:
    for text in ("Loss   Function", "TCP/IP", "  x "):
        assert normalize(text) == providers.slugify(text) == casefold_id(text)
        assert synthetic.slugify(text) == slug_id(text)


def test_committed_synthetic_data_is_reproducible(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_synthetic_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "out"
    args = ["--seed", "42", "--lectures", "3", "--concepts", "8", "--out", str(out)]
    assert module.main(args) == 0
    names = sorted(p.name for p in COMMITTED.iterdir() if p.suffix in {".json", ".csv", ".txt"})
    assert names
    for name in names:
        assert (out / name).read_bytes() == (COMMITTED / name).read_bytes(), name
    json.loads((out / "lecture_1_graph_v1.json").read_text(encoding="utf-8"))
