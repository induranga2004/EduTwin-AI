"""Static checks for the self-contained demo UI page and its sample payload."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from contract_keys import contract_errors

UI = Path(__file__).resolve().parents[1] / "src" / "edusync_c2" / "ui"
HTML = UI / "index.html"
SAMPLE = UI / "sample_analysis.json"


def _html() -> str:
    return HTML.read_text(encoding="utf-8")


def test_page_exists_and_small() -> None:
    assert HTML.is_file()
    assert HTML.stat().st_size < 300 * 1024


def test_page_is_offline_and_self_contained() -> None:
    text = _html()
    assert "http://" not in text.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in text
    assert "<script src" not in text.lower()
    assert not re.search(r"<link[^>]+rel=[\"']?stylesheet", text, re.I)
    assert "@import" not in text


def test_fetch_paths_are_relative() -> None:
    paths = re.findall(r"fetch\(\s*[\"']([^\"']+)[\"']", _html())
    assert paths, "page should call the backend"
    for p in paths:
        assert not p.startswith(("/", "http", "//")), p
    assert {"difficulty/analyze", "demo/sample", "demo/evaluation"} <= set(paths)


def test_embedded_sample_equals_file() -> None:
    m = re.search(r'<script type="application/json" id="sample">(.*?)</script>', _html(), re.S)
    assert m
    assert json.loads(m.group(1)) == json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_sample_satisfies_contract() -> None:
    doc = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert contract_errors(doc) == []
    assert 20 <= doc["meta"]["concept_count"] <= 30
    assert doc["meta"]["slide_count"] >= 24


def test_contract_validator_flags_problems() -> None:
    assert contract_errors([]) == ["root: expected object"]
    doc = json.loads(SAMPLE.read_text(encoding="utf-8"))
    del doc["layers"]["combine"]["weights"]
    doc["final"]["hotspots"][0].pop("concept_id")
    doc["layers"]["combine"]["per_concept"][0]["label"] = "Huge"
    errs = contract_errors(doc)
    assert any("weights" in e for e in errs)
    assert any("concept_id or slide_index" in e for e in errs)
    assert any("bad label" in e for e in errs)


def test_page_has_required_content() -> None:
    text = _html()
    for needle in (
        "EduSync AI - Lecture Difficulty Detector",
        "evidence quality, not a probability",
        "not independent human validation",
        "provisional",
        "@media print",
    ):
        assert needle in text
