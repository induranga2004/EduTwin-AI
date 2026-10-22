"""Regression metrics (hand-computed), text baselines, the baseline script and the spot set."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from public_fixtures import clear_rows, make_onestop_zip, make_xlsx

from edusync_c2_model import baselines_text as bt
from edusync_c2_model import regression_metrics as rm
from edusync_c2_model.examples import FeatureRow
from edusync_c2_model.public_data.schema import TextExample
from edusync_c2_model.spot_set import (
    SHEET_COLUMNS,
    SpotSetError,
    evaluate_spot_set,
    read_ratings,
    sample_snippets,
    write_sheet,
)

PKG = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------- metrics


def test_rank_average_ties():
    assert rm.rank_average([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert rm.rank_average([3, 1, 2]).tolist() == [3.0, 1.0, 2.0]
    assert rm.rank_average([5, 5, 5]).tolist() == [2.0, 2.0, 2.0]
    assert rm.rank_average([]).tolist() == []


def test_spearman_and_pearson_known_values():
    assert rm.spearman([1, 2, 3, 4], [10, 20, 30, 45]) == pytest.approx(1.0)
    assert rm.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # ties: ranks [1,2.5,2.5,4] vs [1,2,3,4] -> r = 0.9486832981
    assert rm.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832981)
    assert rm.pearson([1, 2, 3], [2, 4, 7]) == pytest.approx(0.9933992678)
    assert rm.spearman([1, 1, 1], [1, 2, 3]) is None
    assert rm.pearson([1], [1]) is None
    assert rm.spearman([1], [1]) is None


def test_rmse_mae_r2_known_values():
    y, p = [1.0, 2.0, 3.0], [1.0, 3.0, 5.0]
    assert rm.rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert rm.mae(y, p) == pytest.approx(1.0)
    assert rm.r2(y, p) == pytest.approx(1 - 5 / 2)
    assert rm.r2([2.0, 2.0], [1.0, 3.0]) is None
    assert rm.rmse([], []) is None
    assert rm.mae([], []) is None
    assert rm.r2([], []) is None


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        rm.rmse([1, 2], [1])
    with pytest.raises(ValueError, match="1-D"):
        rm.mae([[1, 2]], [[1, 2]])
    with pytest.raises(ValueError, match="groups"):
        rm.bootstrap_by_group([1, 2], [1, 2], ["a"], {"rmse": rm.rmse})


def data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    return y, y + rng.normal(scale=0.5, size=n), [f"g{i // 2}" for i in range(n)]


def test_bootstrap_is_seeded_and_brackets_the_estimate():
    y, p, g = data()
    a = rm.evaluate_regressor(y, p, g, 7, n_boot=200)
    assert a == rm.evaluate_regressor(y, p, g, 7, n_boot=200)
    assert a != rm.evaluate_regressor(y, p, g, 8, n_boot=200)
    low, high = a["ci"]["spearman"]["low"], a["ci"]["spearman"]["high"]
    assert low < a["spearman"] < high
    assert a["ci"]["rmse"]["low"] < a["rmse"] < a["ci"]["rmse"]["high"]
    assert (a["n"], a["n_groups"], a["n_boot"], a["seed"]) == (60, 30, 200, 7)
    assert set(a) >= {"pearson", "mae", "r2"}


def test_bootstrap_resamples_whole_groups():
    # every group has two rows, so a replicate always has an even number of rows
    seen = []

    def count(a, b):
        seen.append(len(a))
        return 0.0

    rm.bootstrap_by_group(
        [1.0] * 8, [1.0] * 8, ["a", "a", "b", "b", "c", "c", "d", "d"], {"c": count}, n_boot=30
    )
    assert seen
    assert all(n == 8 for n in seen)


def test_bootstrap_undefined_and_empty():
    out = rm.bootstrap_by_group([1, 1, 1], [1, 2, 3], ["a", "b", "c"], {"s": rm.spearman}, n_boot=5)
    assert out == {"s": None}
    assert rm.bootstrap_by_group([], [], [], {"s": rm.spearman}) == {"s": None}
    empty = rm.evaluate_regressor([], [], [], 0, n_boot=3)
    assert empty["n"] == 0
    assert empty["spearman"] is None
    assert empty["ci"]["rmse"] is None


# ----------------------------------------------------------------------------- baselines


@pytest.mark.parametrize(
    ("word", "count"),
    [
        ("cat", 1),
        ("the", 1),
        ("late", 1),
        ("table", 2),
        ("banana", 3),
        ("jumped", 1),
        ("wanted", 2),
        ("boxes", 2),
        ("houses", 2),
        ("readily", 3),
        ("strengths", 1),
        ("rhythm", 1),
        ("I", 1),
    ],
)
def test_syllable_counter(word, count):
    assert bt.count_syllables(word) == count


def test_syllables_of_non_words():
    assert bt.count_syllables("123") == 0
    assert bt.count_syllables("") == 0


def test_flesch_kincaid_known_value_and_order():
    # 6 words, 1 sentence, 6 syllables: 0.39*6 + 11.8*1 - 15.59 = -1.45
    assert bt.flesch_kincaid_grade("The cat sat on the mat.") == pytest.approx(-1.45)
    hard = "Epistemological considerations notwithstanding, institutional heterogeneity persists."
    assert bt.flesch_kincaid_grade(hard) > bt.flesch_kincaid_grade("The cat sat on the mat.")
    assert bt.flesch_kincaid_grade("12345 !!!") == 0.0
    assert bt.split_sentences("One. Two! Three? Four") == ["One", "Two", "Three", "Four"]
    assert bt.split_sentences("...") == []


def test_text_features_shape_and_edge_cases():
    analyzer = bt.get_default_analyzer()
    assert analyzer is not None
    easy = bt.text_features("The cat sat on the mat.", analyzer)
    hard = bt.text_features(
        "Cryptographic authentication notwithstanding, vulnerabilities proliferate.", analyzer
    )
    assert easy.shape == (len(bt.FEATURE_NAMES),)
    assert hard[bt.FEATURE_NAMES.index("fragmentation")] > easy[3 + 1]
    assert hard[bt.FEATURE_NAMES.index("avg_word_len")] > easy[1]
    assert bt.text_features("1 2 3", analyzer).tolist() == [0.0] * 5
    single = bt.text_features("I a.", analyzer)  # words too short for the analyzer
    assert single[3] == 0.0
    assert single[4] == 1.0


def test_feature_matrix_and_missing_resources(monkeypatch):
    assert bt.feature_matrix([]).shape == (0, 5)
    assert bt.feature_matrix(["a cat", "the dog"]).shape == (2, 5)
    monkeypatch.setattr(bt, "get_default_analyzer", lambda: None)
    with pytest.raises(RuntimeError, match="vocabulary"):
        bt.feature_matrix(["x"])


def test_ridge_recovers_a_linear_relation_and_shrinks():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    x[:, 2] = 5.0  # constant column must not break standardisation
    y = 2.0 * x[:, 0] - 1.0 * x[:, 1] + 0.5
    model = bt.RidgeModel(1e-6).fit(x, y)
    assert np.allclose(model.predict(x), y, atol=1e-3)
    strong = bt.RidgeModel(1e6).fit(x, y)
    assert np.abs(strong.coef).max() < np.abs(model.coef).max()


def test_select_alpha_prefers_light_penalty_for_clean_data_and_handles_tiny_input():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(120, 2))
    y = 3 * x[:, 0] + x[:, 1]
    groups = [f"g{i}" for i in range(120)]
    assert bt.select_alpha(x, y, groups, seed=1) == bt.ALPHA_GRID[0]
    assert bt.select_alpha(x, y, groups, seed=1) == bt.select_alpha(x, y, groups, seed=1)
    # one group: the only fold holds everything; two groups: some folds are empty
    assert bt.select_alpha(x, y, ["a"] * 120) == bt.ALPHA_GRID[0]
    assert bt.select_alpha(x, y, ["a", "b"] * 60) in bt.ALPHA_GRID


def passages(n=60):
    easy = "The cat sat on the mat. It was a big red hat."
    hard = (
        "Cryptographic authentication notwithstanding, institutional vulnerabilities proliferate."
    )
    out = []
    for i in range(n):
        text = (easy if i % 2 else hard) + f" Item {i}."
        out.append(TextExample("clear", text, 1.0 if i % 2 else -1.0, None, str(i)))
    return out


def test_run_baselines_on_separable_text():
    examples = passages(60)
    parts = {"train": examples[:40], "val": examples[40:50], "test": examples[50:]}
    outcome = bt.run_baselines(parts, higher_is_easier=True, seed=3, n_boot=50)
    assert outcome["n"] == {"train": 40, "val": 10, "test": 10}
    assert set(outcome["results"]) == {*bt.SINGLE_FEATURE_BASELINES, bt.RIDGE_NAME}
    ridge = outcome["results"][bt.RIDGE_NAME]
    assert set(ridge) == {"val", "test"}
    assert ridge["test"]["n"] == 10
    assert ridge["test"]["spearman"] > 0.8
    assert outcome["results"]["fk_grade"]["test"]["spearman"] > 0.8
    assert outcome["targets"]["scaler"]["n_train"] == 40
    again = bt.run_baselines(parts, higher_is_easier=True, seed=3, n_boot=50)
    assert again == outcome


def test_render_baseline_report_table():
    examples = passages(30)
    parts = {"train": examples[:20], "val": examples[20:25], "test": examples[25:]}
    outcome = bt.run_baselines(parts, higher_is_easier=True, n_boot=20)
    outcome["results"]["fk_grade"]["val"]["ci"]["rmse"] = None
    outcome["results"]["fk_grade"]["val"]["rmse"] = None
    text = bt.render_baseline_report("clear", outcome, "Caveat: general English.")
    assert "| Baseline | Part | n | Spearman |" in text
    assert "| fk_grade | val | 5 |" in text
    assert "n/a" in text
    assert "Caveat: general English." in text
    assert text.endswith("\n")


# ------------------------------------------------------------------------------- script


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PKG / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_script_runs_on_fixtures_and_seals_split(tmp_path, capsys):
    script = load_script("eval_baselines_text")
    data_dir, runs = tmp_path / "pub", tmp_path / "runs"
    data_dir.mkdir()
    make_xlsx(data_dir / "CLEAR_corpus_final.xlsx", clear_rows(60))
    make_onestop_zip(data_dir / "onestop.zip", 30)
    assert (
        script.main(["--data-dir", str(data_dir), "--runs-dir", str(runs), "--n-boot", "20"]) == 0
    )
    out = capsys.readouterr().out
    assert "Text baselines on clear" in out
    assert "Text baselines on onestop" in out
    assert (runs / "baselines_text_clear.md").is_file()
    sealed = (data_dir / "splits" / "clear_split.json").read_text(encoding="utf-8")
    assert (
        script.main(["--data-dir", str(data_dir), "--runs-dir", str(runs), "--n-boot", "20"]) == 0
    )
    assert (data_dir / "splits" / "clear_split.json").read_text(encoding="utf-8") == sealed


def test_eval_script_skips_missing_data_and_reports_errors(tmp_path, capsys):
    script = load_script("eval_baselines_text")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert script.main(["--data-dir", str(empty), "--runs-dir", str(tmp_path / "r")]) == 0
    assert "skipped" in capsys.readouterr().out
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "onestop.zip").write_bytes(b"nope")
    assert script.main(["--data-dir", str(bad), "--runs-dir", str(tmp_path / "r")]) == 1
    assert "error" in capsys.readouterr().err


# ------------------------------------------------------------------------------ spot set


def feature_rows():
    rows = []
    for lecture in ("L1", "L2", "L3"):
        for k in range(10):
            rows.append(
                FeatureRow(
                    lecture,
                    "C1",
                    f"c{k}",
                    f"Concept {k}",
                    f"Text   of\n{lecture} concept {k}. " * 3,
                    (0.0,) * 35,
                    (0,) * 35,
                    ("computed",) * 35,
                )
            )
    rows.append(
        FeatureRow("L1", "C1", "empty", "Empty", "   ", (0.0,) * 35, (0,) * 35, ("computed",) * 35)
    )
    return rows


def test_sample_snippets_seeded_balanced_and_order_independent():
    rows = feature_rows()
    a = sample_snippets(rows, n=12, seed=4)
    assert a == sample_snippets(list(reversed(rows)), n=12, seed=4)
    assert a != sample_snippets(rows, n=12, seed=5)
    assert [r["snippet_id"] for r in a] == [f"S{i:03d}" for i in range(1, 13)]
    assert {r["lecture_id"] for r in a} == {"L1", "L2", "L3"}
    assert [r["lecture_id"] for r in a].count("L1") == 4
    assert all("empty" not in r["concept"].lower() for r in a)
    assert all(r["rater_id"] == r["rating"] == r["notes"] == "" for r in a)
    assert "\n" not in a[0]["text"]
    assert "  " not in a[0]["text"]


def test_sample_snippets_caps_at_available_and_slide_index_and_long_text():
    rows = feature_rows()
    every = sample_snippets(rows, n=1000)
    assert len(every) == 30
    first = rows[0]
    out = sample_snippets([first], n=5, slide_index={first.key: 7})
    assert out[0]["slide_index"] == "7"
    long_row = FeatureRow(
        "L9", "C", "c", "Long", "word " * 400, (0.0,) * 35, (0,) * 35, ("x",) * 35
    )
    text = sample_snippets([long_row])[0]["text"]
    assert len(text) == 800
    assert text.endswith("…")


def test_write_sheet_columns_and_no_overwrite(tmp_path):
    snippets = sample_snippets(feature_rows(), n=5)
    path = tmp_path / "sheet.csv"
    write_sheet(path, snippets)
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == SHEET_COLUMNS
        assert len(list(reader)) == 5
    with pytest.raises(SpotSetError, match="never overwritten"):
        write_sheet(path, snippets)


def rate(path, ratings):
    """Fill the rating column: ``ratings`` maps snippet_id -> list of (rater, rating)."""
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        given = ratings.get(row["snippet_id"], [("", "")])
        for rater, rating in given:
            out.append({**row, "rater_id": rater, "rating": rating})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(out)


def make_rated_sheet(tmp_path, n=30):
    path = tmp_path / "sheet.csv"
    write_sheet(path, sample_snippets(feature_rows(), n=n))
    labels = ["Low", "Medium", "High"]
    ratings = {f"S{i:03d}": [("r1", labels[(i - 1) % 3])] for i in range(1, n + 1)}
    rate(path, ratings)
    return path, ratings


def test_evaluate_spot_set_perfect_and_reversed(tmp_path):
    path, _ = make_rated_sheet(tmp_path)
    scores = {f"S{i:03d}": ((i - 1) % 3) / 2 for i in range(1, 31)}
    cutoffs = {"low_below": 0.25, "high_from": 0.75}
    result = evaluate_spot_set(path, scores, cutoffs, n_boot=100)
    assert result["n"] == 30
    assert result["n_lectures"] == 3
    assert result["n_unscored"] == 0
    assert result["spearman"] == pytest.approx(1.0)
    assert result["qwk"] == pytest.approx(1.0)
    assert set(result["ci"]) == {"spearman", "qwk"}
    flipped = {k: 1 - v for k, v in scores.items()}
    assert evaluate_spot_set(path, flipped, cutoffs, n_boot=20)["spearman"] == pytest.approx(-1.0)
    no_cut = evaluate_spot_set(path, scores, n_boot=20)
    assert no_cut["qwk"] is None
    assert no_cut["ci"]["qwk"] is None


def test_evaluate_spot_set_multiple_raters_unrated_and_unscored(tmp_path):
    path = tmp_path / "sheet.csv"
    write_sheet(path, sample_snippets(feature_rows(), n=6))
    rate(
        path,
        {
            "S001": [("a", "low"), ("b", "HIGH")],  # mean 1.0 -> Medium
            "S002": [("a", "High")],
            "S003": [("a", "Low")],
            "S004": [("a", "Medium")],
            "S005": [("a", "High")],
        },  # S006 unrated
    )
    ratings = read_ratings(path)
    assert sorted(ratings) == ["S001", "S002", "S003", "S004", "S005"]
    assert ratings["S001"]["ratings"] == [0, 2]
    scores = {"S001": 0.5, "S002": 0.9, "S003": 0.1, "S004": 0.5, "S006": 0.4}  # S005 unscored
    result = evaluate_spot_set(path, scores, {"low_below": 0.3, "high_from": 0.7}, n_boot=30)
    assert (result["n_rated"], result["n"], result["n_unscored"]) == (5, 4, 1)
    assert result["qwk"] == pytest.approx(1.0)


def test_evaluate_spot_set_nothing_to_compare_and_bad_sheets(tmp_path):
    path = tmp_path / "sheet.csv"
    write_sheet(path, sample_snippets(feature_rows(), n=4))
    result = evaluate_spot_set(path, {"S001": 0.1})
    assert result["n"] == 0
    assert result["spearman"] is None
    rate(path, {"S001": [("a", "Hard")]})
    with pytest.raises(SpotSetError, match="Low, Medium or High"):
        read_ratings(path)
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(SpotSetError, match="columns"):
        read_ratings(wrong)
    blank = tmp_path / "blank.csv"
    blank.write_text("", encoding="utf-8")
    with pytest.raises(SpotSetError, match="columns"):
        read_ratings(blank)


def test_export_spot_set_script(tmp_path, make_entry, capsys):
    from edusync_c2_model.manifest import write_manifest

    entry = make_entry(7, graph=False)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [entry.__class__(entry.lecture_id, "C1", entry.file, None)])
    out = tmp_path / "real" / "spot.csv"
    script = load_script("export_spot_set")
    argv = [
        "--manifest",
        str(manifest),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out",
        str(out),
        "--n",
        "3",
    ]
    assert script.main(argv) == 0
    assert "wrote 3 snippets" in capsys.readouterr().out
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4
    assert script.main(argv) == 1  # never overwrites a rated sheet
    assert "already exists" in capsys.readouterr().err
    assert script.main([*argv[:1], str(tmp_path / "nope.csv")]) == 1
