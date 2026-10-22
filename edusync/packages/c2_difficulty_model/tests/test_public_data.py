"""Public data layer: xlsx reader, CLEAR/OneStop loaders, schema, splits, targets, licences."""

import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from public_fixtures import clear_rows, make_onestop_zip, make_xlsx

from edusync_c2_model.public_data import clear, onestop, xlsx
from edusync_c2_model.public_data.licences import (
    SOURCES,
    DownloadError,
    download,
    fetch_url,
    render_licences,
    sha256_file,
)
from edusync_c2_model.public_data.schema import TextExample
from edusync_c2_model.public_data.splits import (
    make_group_split,
    partition,
    read_split,
    validate_group_split,
    write_split_once,
)
from edusync_c2_model.public_data.targets import (
    TargetError,
    fit_cutoffs,
    fit_difficulty_scaler,
    fit_targets,
    to_difficulty,
    to_label_ids,
)
from edusync_c2_model.split import LeakageError, SplitError

PKG = Path(__file__).resolve().parents[1]
REAL = PKG / "data" / "public"

# ------------------------------------------------------------------------------------ xlsx


def test_read_rows_shared_inline_numbers_and_gaps(tmp_path):
    rows = [["a", "b", "c"], ["x", None, 3.5], [None, "y"]]
    rows[0] += [None] * 24 + ["AB"]  # a cell in column AB (index 27)
    path = make_xlsx(tmp_path / "t.xlsx", rows)
    got = xlsx.read_rows(path)
    assert got[0][:3] == ["a", "b", "c"]
    assert got[0][27] == "AB"
    assert len(got[0]) == 28
    assert got[1] == ["x", "", "3.5"]
    assert got[2] == ["", "y"]


def test_inline_strings_and_missing_shared_strings_part(tmp_path):
    path = make_xlsx(tmp_path / "t.xlsx", [["h"], ["v & w"]], shared=False)
    assert xlsx.read_rows(path) == [["h"], ["v & w"]]


def test_empty_cell_element_and_empty_row(tmp_path):
    path = tmp_path / "t.xlsx"
    sheet = (
        f'<worksheet xmlns="{xlsx.NS[1:-1]}"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c>'
        '<c r="B1"/></row><row r="2"/></sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
        archive.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="{xlsx.NS[1:-1]}"><si><t>only</t></si></sst>',
        )
    assert xlsx.read_rows(path) == [["only", ""], []]


def test_first_sheet_is_resolved_through_the_relationships(tmp_path):
    path = make_xlsx(
        tmp_path / "t.xlsx",
        [["h"], ["v"]],
        sheet_path="xl/worksheets/data9.xml",
        rels_target="/xl/worksheets/data9.xml",
    )
    assert xlsx.read_rows(path) == [["h"], ["v"]]
    rel = make_xlsx(
        tmp_path / "u.xlsx", [["h"]], sheet_path="xl/other.xml", rels_target="other.xml"
    )
    assert xlsx.read_rows(rel) == [["h"]]


def test_unresolvable_relationship_falls_back_to_sheet1(tmp_path):
    no_match = make_xlsx(tmp_path / "a.xlsx", [["h"]], rel_id="rId9")
    assert xlsx.read_rows(no_match) == [["h"]]
    no_workbook = make_xlsx(tmp_path / "b.xlsx", [["h"]], rels_target=None)
    assert xlsx.read_rows(no_workbook) == [["h"]]


def test_unreadable_workbooks_raise(tmp_path):
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(b"not a zip")
    with pytest.raises(xlsx.XlsxError):
        xlsx.read_rows(bad)
    with pytest.raises(xlsx.XlsxError):
        xlsx.read_rows(tmp_path / "missing.xlsx")
    no_sheet = tmp_path / "ns.xlsx"
    with zipfile.ZipFile(no_sheet, "w") as archive:
        archive.writestr("other.txt", "x")
    with pytest.raises(xlsx.XlsxError):
        xlsx.read_rows(no_sheet)


def test_bad_cell_reference_and_empty_table(tmp_path):
    path = tmp_path / "t.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{xlsx.NS[1:-1]}"><sheetData><row><c r="1"><v>1</v></c></row>'
            "</sheetData></worksheet>",
        )
    with pytest.raises(xlsx.XlsxError, match="bad cell"):
        xlsx.read_rows(path)
    empty = make_xlsx(tmp_path / "e.xlsx", [])
    with pytest.raises(xlsx.XlsxError, match="no rows"):
        xlsx.read_table(empty)


def test_read_table_pads_short_rows(tmp_path):
    path = make_xlsx(tmp_path / "t.xlsx", [["a", "b"], ["1"]])
    assert xlsx.read_table(path) == [{"a": "1", "b": ""}]


# ------------------------------------------------------------------------------------ CLEAR


def test_load_clear_reads_id_excerpt_and_ease(tmp_path):
    path = make_xlsx(tmp_path / "clear.xlsx", clear_rows(30))
    examples = clear.load_clear(path)
    assert len(examples) == 30
    first = examples[0]
    assert (first.source, first.level, first.group_id) == ("clear", None, "0")
    assert first.target == pytest.approx(-1.5)
    assert first.text.endswith("Passage number 0.")


def test_clear_skips_empty_excerpts_and_checks_columns():
    base = {"ID": "1", "Excerpt": "Some text.", "BT_easiness": "0.5"}
    blank = {"ID": "2", "Excerpt": "   ", "BT_easiness": "0.1"}
    assert [e.group_id for e in clear.parse_rows([base, blank])] == ["1"]
    assert clear.parse_rows([]) == []
    with pytest.raises(clear.ClearError, match="missing columns"):
        clear.parse_rows([{"ID": "1"}])


@pytest.mark.parametrize("ease", ["abc", "nan", "inf"])
def test_clear_rejects_bad_ease(ease):
    with pytest.raises(clear.ClearError, match=r"numeric|non-finite"):
        clear.parse_rows([{"ID": "1", "Excerpt": "t", "BT_easiness": ease}])


def test_clear_rejects_duplicate_ids():
    row = {"ID": "1", "Excerpt": "t", "BT_easiness": "0"}
    with pytest.raises(clear.ClearError, match="duplicate"):
        clear.parse_rows([row, dict(row)])


# --------------------------------------------------------------------------------- OneStop


def test_load_onestop_levels_groups_and_junk(tmp_path):
    path = make_onestop_zip(tmp_path / "o.zip", 12)
    examples = onestop.load_onestop(path)
    assert len(examples) == 36
    assert {e.group_id for e in examples} == {f"Article {i}" for i in range(12)}
    one = [e for e in examples if e.group_id == "Article 0"]
    assert [e.level for e in one] == [0, 1, 2]
    assert [e.target for e in one] == [0.0, 1.0, 2.0]
    assert not one[0].text.startswith("﻿")
    assert examples == sorted(examples, key=lambda e: (e.group_id, e.level))


def test_onestop_errors(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"nope")
    with pytest.raises(onestop.OneStopError):
        onestop.load_onestop(bad)
    with pytest.raises(onestop.OneStopError):
        onestop.load_onestop(tmp_path / "missing.zip")
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("readme.md", "x")
    with pytest.raises(onestop.OneStopError, match="no texts"):
        onestop.load_onestop(empty)
    latin = tmp_path / "latin.zip"
    with zipfile.ZipFile(latin, "w") as archive:
        archive.writestr(f"r/{onestop.TEXTS_DIR}/Ele-Txt/a -ele.txt", b"\xff\xfe\xfa")
    with pytest.raises(onestop.OneStopError):
        onestop.load_onestop(latin)
    dup = tmp_path / "dup.zip"
    with zipfile.ZipFile(dup, "w") as archive:
        archive.writestr(f"a/{onestop.TEXTS_DIR}/Ele-Txt/x -ele.txt", "one")
        archive.writestr(f"b/{onestop.TEXTS_DIR}/Ele-Txt/x -ele.txt", "two")
    with pytest.raises(onestop.OneStopError, match="twice"):
        onestop.load_onestop(dup)


# ------------------------------------------------------------------------------------ schema


def test_text_example_json_round_trip_and_validation():
    example = TextExample("clear", "Héllo.", 0.25, None, "7")
    assert TextExample.from_json(example.to_json()) == example
    assert "é" in example.to_json()
    levelled = TextExample("onestop", "t", 2.0, 2, "a")
    assert TextExample.from_json(levelled.to_json()) == levelled
    with pytest.raises(ValueError, match="text"):
        TextExample("clear", "  ", 0.0, None, "1")
    with pytest.raises(ValueError, match="group_id"):
        TextExample("clear", "t", 0.0, None, "")
    with pytest.raises(ValueError, match=r"NaN|not JSON"):
        TextExample("clear", "t", float("nan"), None, "1").to_json()


# ------------------------------------------------------------------------------------ splits


def ids(n=100):
    return [f"g{i:03d}" for i in range(n)]


def test_split_sizes_disjoint_and_deterministic():
    split = make_group_split(ids(100), seed=3, source="clear")
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (80, 10, 10)
    assert set(split["train"]) | set(split["val"]) | set(split["test"]) == set(ids(100))
    shuffled = list(reversed(ids(100))) + ids(10)  # order and duplicates do not matter
    assert make_group_split(shuffled, seed=3, source="clear") == split
    assert make_group_split(ids(100), seed=4, source="clear") != split


def test_split_validation_errors():
    with pytest.raises(SplitError, match="at least"):
        make_group_split(ids(2), seed=0, source="x")
    with pytest.raises(SplitError, match="fractions"):
        make_group_split(ids(20), seed=0, source="x", val_fraction=0.6, test_fraction=0.5)
    split = make_group_split(ids(20), seed=0, source="x")
    split["val"] = [*split["val"], split["test"][0]]
    with pytest.raises(LeakageError, match="val and test"):
        validate_group_split(split)
    split["val"] = []
    with pytest.raises(SplitError, match="non-empty"):
        validate_group_split(split)


def test_split_seal_write_once_and_read(tmp_path):
    split = make_group_split(ids(30), seed=1, source="clear")
    path = tmp_path / "split.json"
    write_split_once(path, split)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(SplitError, match="never overwritten"):
        write_split_once(path, make_group_split(ids(30), seed=2, source="clear"))
    assert path.read_text(encoding="utf-8") == before
    assert read_split(path) == split


def test_read_split_rejects_bad_files(tmp_path):
    with pytest.raises(SplitError, match="cannot use"):
        read_split(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(SplitError, match="cannot use"):
        read_split(bad)
    split = make_group_split(ids(30), seed=1, source="x") | {"version": 99}
    old = tmp_path / "old.json"
    old.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(SplitError, match="version"):
        read_split(old)


def test_onestop_levels_of_one_article_never_straddle(tmp_path):
    examples = onestop.load_onestop(make_onestop_zip(tmp_path / "o.zip", 30))
    split = make_group_split({e.group_id for e in examples}, seed=5, source="onestop")
    parts = partition(reversed(examples), split)
    for name, members in parts.items():
        assert len(members) == 3 * len(split[name])
        assert {e.group_id for e in members} == set(split[name])


def test_partition_rejects_unknown_group():
    split = make_group_split(ids(20), seed=0, source="x")
    with pytest.raises(SplitError, match="not in the split"):
        partition([TextExample("clear", "t", 0.0, None, "zzz")], split)


# ----------------------------------------------------------------------------------- targets


def test_difficulty_scaler_negates_ease_and_uses_train_only():
    train_ease = [-2.0, -1.0, 0.0, 1.0, 2.0]
    scaler = fit_difficulty_scaler(train_ease, higher_is_easier=True)
    assert scaler["mean"] == pytest.approx(0.0)
    assert scaler["n_train"] == 5
    z = to_difficulty([2.0, -2.0, 0.0], scaler)
    assert z[0] < 0 < z[1]
    assert z[2] == pytest.approx(0.0)
    assert np.std(to_difficulty(train_ease, scaler)) == pytest.approx(1.0)
    again = fit_difficulty_scaler(train_ease, higher_is_easier=True)
    assert again == scaler
    # a different held-out set leaves the fitted statistics unchanged
    to_difficulty([100.0], scaler)
    assert scaler == again


def test_scaler_keeps_direction_for_levels_and_handles_constant_targets():
    levels = fit_difficulty_scaler([0.0, 1.0, 2.0], higher_is_easier=False)
    assert to_difficulty([2.0], levels)[0] > to_difficulty([0.0], levels)[0]
    constant = fit_difficulty_scaler([3.0, 3.0], higher_is_easier=False)
    assert constant["std"] == 1.0


@pytest.mark.parametrize("bad", [[1.0], [], [1.0, float("nan")]])
def test_scaler_needs_finite_training_targets(bad):
    with pytest.raises(TargetError):
        fit_difficulty_scaler(bad, higher_is_easier=True)


def test_cutoffs_are_terciles_of_train_and_label_ids():
    train = list(np.linspace(-1, 1, 300))
    cutoffs = fit_cutoffs(train)
    labels = to_label_ids(train, cutoffs)
    assert [labels.count(i) for i in range(3)] == [100, 100, 100]
    assert to_label_ids([-9.0, 0.0, 9.0], cutoffs) == [0, 1, 2]
    skew = fit_cutoffs(train, low_share=0.4, high_share=0.25)
    assert to_label_ids(train, skew).count(2) == pytest.approx(75, abs=2)
    with pytest.raises(TargetError, match="low_share"):
        fit_cutoffs(train, low_share=0.6, high_share=0.5)
    with pytest.raises(TargetError, match="finite"):
        to_label_ids([float("nan")], cutoffs)


def test_fit_targets_bundles_scaler_and_cutoffs():
    fitted = fit_targets([float(i) for i in range(30)], higher_is_easier=True)
    assert set(fitted) == {"scaler", "cutoffs"}
    assert fitted["cutoffs"]["low_below"] < fitted["cutoffs"]["high_from"]
    json.dumps(fitted)  # JSON-serialisable


# ----------------------------------------------------------------------------------- licences


def test_download_refuses_overwrite_and_records_bytes(tmp_path):
    source = SOURCES[0]
    path = download(source, tmp_path / "pub", fetch=lambda url: b"data " + url.encode())
    assert path.read_bytes().startswith(b"data https://")
    with pytest.raises(DownloadError, match="--force"):
        download(source, tmp_path / "pub", fetch=lambda url: b"other")
    download(source, tmp_path / "pub", force=True, fetch=lambda url: b"other")
    assert path.read_bytes() == b"other"
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_download_failures(tmp_path):
    def boom(url):
        raise OSError("offline")

    with pytest.raises(DownloadError, match="OSError"):
        download(SOURCES[0], tmp_path, fetch=boom)
    with pytest.raises(DownloadError, match="no data"):
        download(SOURCES[0], tmp_path, fetch=lambda url: b"")
    assert not (tmp_path / SOURCES[0].filename).exists()


def test_fetch_url_uses_urlopen(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"payload"

    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["timeout"] = request.full_url, timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert fetch_url("https://example.org/x", timeout=5) == b"payload"
    assert seen == {"url": "https://example.org/x", "timeout": 5}


def test_render_licences_lists_hash_counts_and_restrictions(tmp_path):
    for source in SOURCES:
        (tmp_path / source.filename).write_bytes(source.key.encode())
    text = render_licences(SOURCES, tmp_path, {"clear": 10, "onestop": 6}, "2026-01-02")
    assert "CC BY-NC-SA 4.0" in text
    assert "CC BY-SA 4.0" in text
    assert sha256_file(tmp_path / "onestop.zip") in text
    assert "Rows used: 10" in text
    assert "Accessed: 2026-01-02" in text
    assert "never committed" in text


def load_script(name):
    path = PKG / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_fetch(tmp_path):
    clear_path = make_xlsx(tmp_path / "src_clear.xlsx", clear_rows(12))
    zip_path = make_onestop_zip(tmp_path / "src_one.zip", 5)

    def fetch(url):
        return (clear_path if "CLEAR" in url else zip_path).read_bytes()

    return fetch


def test_download_script_end_to_end(tmp_path, capsys):
    script = load_script("download_public_data")
    licences = tmp_path / "DATA_LICENCES.md"
    counts = script.run(tmp_path / "pub", licences, fetch=fake_fetch(tmp_path), today="2026-03-04")
    assert counts == {"clear": 12, "onestop": 15}
    assert "Accessed: 2026-03-04" in licences.read_text(encoding="utf-8")
    assert "CC BY-NC-SA" in capsys.readouterr().out
    with pytest.raises(DownloadError):
        script.run(tmp_path / "pub", licences, fetch=fake_fetch(tmp_path))


def test_download_script_main_exit_codes(tmp_path, monkeypatch, capsys):
    script = load_script("download_public_data")
    monkeypatch.setattr(script, "fetch_url", fake_fetch(tmp_path))
    args = ["--folder", str(tmp_path / "pub"), "--licences", str(tmp_path / "L.md")]
    assert script.main(args) == 0
    assert script.main(args) == 1
    assert "--force" in capsys.readouterr().err
    assert script.main([*args, "--force"]) == 0


# ------------------------------------------------------------------------------- real data


@pytest.mark.skipif(
    not (REAL / "CLEAR_corpus_final.xlsx").is_file() or not (REAL / "onestop.zip").is_file(),
    reason="public corpora not downloaded",
)
def test_real_corpora_counts():
    examples = clear.load_clear(REAL / "CLEAR_corpus_final.xlsx")
    assert 4500 <= len(examples) <= 4800
    assert all(e.text.strip() for e in examples)
    one = onestop.load_onestop(REAL / "onestop.zip")
    assert len(one) == 567
    assert len({e.group_id for e in one}) == 189
