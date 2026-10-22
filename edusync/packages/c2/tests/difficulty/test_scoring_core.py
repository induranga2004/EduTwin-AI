import json

import numpy as np
import pytest

from edusync_c2.difficulty.registry import FEATURE_NAMES, FEATURE_SPECS
from edusync_c2.difficulty.schemas import FeatureStatus, FeatureValue
from edusync_c2.difficulty.scoring import combiner as cb
from edusync_c2.difficulty.scoring import hotspots as hs
from edusync_c2.difficulty.scoring.confidence import evidence_confidence
from edusync_c2.difficulty.scoring.reference import (
    Reference,
    ScoringError,
    build_reference,
    percentile_in,
    quantile_table,
)
from edusync_c2.difficulty.scoring.signals import (
    DIRECTIONS,
    FEATURE_GROUP,
    group_scores,
    score_features,
    top_features,
)

# ------------------------------------------------------------------------------ reference


def test_quantile_table_and_percentile():
    assert quantile_table([]) == []
    table = quantile_table([1, 2, 3, 4, 5], n=5)
    assert table == [1, 2, 3, 4, 5]
    assert percentile_in([], 1) is None
    assert percentile_in([3.0], 9) == 0.5
    assert percentile_in(table, 0) == 0.0
    assert percentile_in(table, 9) == 1.0
    assert percentile_in(table, 3) == 0.5
    assert percentile_in(table, 2.5) == pytest.approx(0.375)
    assert percentile_in([2, 2, 2, 2], 2) == 0.5  # constant feature: mid-rank


class Row:
    def __init__(self, lecture, values, statuses=None, missing=None):
        n = len(FEATURE_NAMES)
        self.lecture_id = lecture
        self.values = values
        self.statuses = statuses or ("computed",) * n
        self.missing = missing or (0,) * n


def test_build_reference_and_roundtrip(tmp_path):
    n = len(FEATURE_NAMES)
    rows = [
        Row("a", tuple(float(i) for i in range(n))),
        Row("b", tuple(float(i) + 1 for i in range(n)), missing=(1,) + (0,) * (n - 1)),
        Row("b", (None,) * n, statuses=("unavailable",) * n),
    ]
    ref = build_reference(rows, schema_version="1.0", schema_hash="h")
    assert (ref.n_decks, ref.n_concepts, ref.provisional) == (2, 3, True)
    assert ref.counts[FEATURE_NAMES[0]] == 1
    assert ref.counts[FEATURE_NAMES[1]] == 2
    assert ref.percentile("nope", 1.0) is None
    path = tmp_path / "ref.json"
    ref.save(path)
    assert Reference.load(path) == ref
    big = Reference(ref.tables, ref.counts, 30, 3)
    assert not big.provisional


def test_reference_errors(tmp_path):
    with pytest.raises(ScoringError):
        Reference.from_dict({"format": 9})
    with pytest.raises(ScoringError):
        Reference.from_dict({})
    with pytest.raises(ScoringError):
        Reference.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ScoringError):
        Reference.load(bad)


# --------------------------------------------------------------------------------- signals


def test_directions_cover_registry():
    assert set(DIRECTIONS) == set(FEATURE_NAMES) == set(FEATURE_GROUP)
    assert all(v in (-1, 0, 1) for v in DIRECTIONS.values())


def fv(name, value, status=FeatureStatus.COMPUTED):
    spec = next(s for s in FEATURE_SPECS if s.name == name)
    return FeatureValue(
        name=name,
        group=spec.group,
        value=value if status is FeatureStatus.COMPUTED else None,
        status=status,
        source="t",
    )


def test_score_features_and_groups():
    ref = Reference(
        tables={
            "lang.long_word_ratio": quantile_table([0, 1, 2, 3, 4]),
            "expl.example_count": quantile_table([0, 1, 2, 3, 4]),
        },
        counts={},
        n_decks=1,
        n_concepts=5,
    )
    feats = [
        fv("lang.long_word_ratio", 4.0),
        fv("expl.example_count", 4.0),  # inverted: many examples = easier
        fv("prereq.mean_confidence", 1.0),  # direction 0, no table
        fv("lang.avg_word_length", None, FeatureStatus.UNAVAILABLE),
    ]
    scores = score_features(feats, ref)
    by = {s.name: s for s in scores}
    assert by["lang.long_word_ratio"].effect == 1.0
    assert by["expl.example_count"].percentile == 1.0
    assert by["expl.example_count"].effect == 0.0
    assert by["prereq.mean_confidence"].effect is None
    assert by["lang.avg_word_length"].percentile is None
    groups = group_scores(scores)
    assert groups["A"] == 1.0
    assert groups["C"] == 0.0
    assert groups["B"] is None
    assert groups["D"] is None
    assert [s.name for s in top_features(scores, 1)] == ["lang.long_word_ratio"]


# -------------------------------------------------------------------------------- combiner


def test_label_for():
    assert cb.label_for(None, 0.4, 0.75) is None
    assert [cb.label_for(v, 0.4, 0.75) for v in (0.1, 0.4, 0.74, 0.75)] == [
        "Low",
        "Medium",
        "Medium",
        "High",
    ]


def corr_matrix(n=40, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    cols = [base + rng.normal(scale=0.3, size=n) for _ in range(5)]
    return np.column_stack(cols)


def test_fit_weights_pc_sign_both_ways(monkeypatch):
    m = corr_matrix()
    real = np.linalg.eigh

    def flipped(sign):
        def eigh(c):
            values, vectors = real(c)
            return values, vectors * sign

        return eigh

    for sign in (1, -1):
        monkeypatch.setattr(cb.np.linalg, "eigh", flipped(sign))
        w = cb.fit_weights(m)
        assert sum(w.values()) == pytest.approx(1.0)
        assert all(v > 0 for v in w.values())


def test_fit_weights_degenerate_cases():
    m = corr_matrix()
    m[:, 2] = np.nan  # a signal that never exists gets weight 0
    w = cb.fit_weights(m)
    assert w["C"] == 0
    assert sum(w.values()) == pytest.approx(1.0)
    one = np.full((10, 5), np.nan)
    one[:, 0] = np.arange(10)
    assert cb.fit_weights(one)["A"] == 1.0  # single usable signal
    const = np.full((10, 5), 0.5)
    assert sum(cb.fit_weights(const).values()) == pytest.approx(1.0)  # fallback columns
    assert sum(cb.fit_weights(np.full((10, 5), np.nan)).values()) == 0
    assert cb.fit_weights(m[:2])["A"] == pytest.approx(0.25)  # fewer than 3 rows: equal


def test_combiner_scoring_and_persistence(tmp_path):
    rows = [dict(zip(cb.SIGNALS, row, strict=True)) for row in corr_matrix(30).tolist()]
    rows.append({"A": 0.5})
    comb = cb.fit_combiner(rows)
    full = dict.fromkeys(cb.SIGNALS, 0.9)
    assert comb.score(full) > 0.8
    part = {"A": 0.2, "B": None}
    assert comb.raw(part) == 0.2
    assert comb.raw({"A": None}) is None
    assert comb.score({}) is None
    contrib = comb.contributions(full)
    assert sum(contrib.values()) == pytest.approx(comb.raw(full))
    assert comb.contributions({"A": 0.3})["B"] is None
    assert comb.label(comb.score(full)) == "High"
    path = tmp_path / "c.json"
    comb.save(path)
    assert cb.Combiner.load(path) == comb
    with pytest.raises(ScoringError):
        cb.fit_combiner([{"A": None}])
    zero = cb.Combiner(dict.fromkeys(cb.SIGNALS, 0.0), 0.4, 0.75, [0.0, 1.0])
    assert zero.raw({"A": 0.4}) is None


def test_combiner_from_dict_errors(tmp_path):
    good = cb.fit_combiner(
        [dict(zip(cb.SIGNALS, row, strict=True)) for row in corr_matrix(10).tolist()]
    ).to_dict()
    for bad in (
        {**good, "format": 9},
        {k: v for k, v in good.items() if k != "weights"},
        {**good, "combined_table": []},
        {**good, "cutoffs": {"low": 0.9, "high": 0.1}},
        {**good, "weights": {**good["weights"], "A": -1}},
    ):
        with pytest.raises(ScoringError):
            cb.Combiner.from_dict(bad)
    with pytest.raises(ScoringError):
        cb.Combiner.load(tmp_path / "none.json")
    p = tmp_path / "x.json"
    p.write_text(json.dumps("{"), encoding="utf-8")
    with pytest.raises(ScoringError):
        cb.Combiner.load(p)


# ------------------------------------------------------------------------------ confidence


def test_confidence():
    full = dict.fromkeys(cb.SIGNALS, 0.5)
    value, parts = evidence_confidence(full, 0.5, 0.5, provisional=False, graph_kind="real")
    assert value == pytest.approx(1.0)
    assert parts["graph_factor"] == 1.0
    low, parts = evidence_confidence(
        {"A": 0.0, "B": 1.0}, None, 0.9, provisional=True, graph_kind="mock"
    )
    assert 0 < low < 0.5
    assert parts["t1_t2_gap"] is None
    assert parts["spread"] == 1.0
    cur, _ = evidence_confidence(full, 0.0, 1.0, provisional=False, graph_kind="curated")
    assert cur < value
    none, _ = evidence_confidence({}, None, None, provisional=True, graph_kind="none")
    assert none > 0


# ------------------------------------------------------------------------------- hotspots


def test_reasons_and_slide_score():
    ref = Reference(
        {
            "lang.long_word_ratio": quantile_table([0, 1, 2, 3, 4]),
            "expl.example_count": quantile_table([0, 1, 2, 3, 4]),
        },
        {},
        1,
        5,
    )
    feats = score_features(
        [
            fv("lang.long_word_ratio", 4.0),
            fv("expl.example_count", 0.0),
            fv("lang.avg_word_length", 2.0),
        ],
        ref,
    )
    reasons = hs.concept_reasons(feats, 0.9)
    assert len(reasons) == 3
    assert any("is high" in r for r in reasons)
    assert any("is low" in r for r in reasons)
    assert hs.concept_reasons([], 0.1) == [hs.NO_REASON]
    assert hs.phrase("lang.avg_word_length") == "avg word length"
    assert hs.phrase("dens.words_per_slide") == "words per slide"
    assert hs.slide_score([0.5, 1.0], 0.5) == pytest.approx(0.8)
    assert hs.slide_score([0.5], None) == 0.5
    assert hs.slide_score([], 0.3) == 0.3
    assert hs.slide_score([], None) is None


def test_slide_scores_and_hotspots():
    concepts = {
        "a": {"concept_label": "A", "score": 0.9, "confidence": 0.8},
        "b": {"concept_label": "B", "score": 0.5, "confidence": 0.6},
        "c": {"concept_label": "C", "score": None, "confidence": 0.1},
    }
    rows = hs.build_slide_scores(
        [1, 2, 3, 4, 5],
        {1: ["b", "a"], 2: ["c"], 3: ["a"]},
        concepts,
        {1: 0.7, 3: None, 4: 0.9, 5: None},
        0.4,
        0.75,
    )
    by = {r["index"]: r for r in rows}
    assert by[1]["concept_ids"] == ["a", "b"]
    assert by[1]["label"] == "High"
    assert by[1]["confidence"] == pytest.approx(0.7)
    assert by[2]["score"] is None
    assert by[2]["confidence"] == 0.0
    assert by[3]["reasons"][0].startswith("Contains the concept 'A'")
    assert by[4]["confidence"] == hs.TEXT_ONLY_CONFIDENCE
    assert "Wording" in by[4]["reasons"][0]
    assert by[5]["score"] is None
    assert by[5]["reasons"] == [hs.NO_REASON]
    crow = [
        {
            "concept_id": "a",
            "concept_label": "A",
            "slide_indexes": [3, 1],
            "score": 0.9,
            "label": "High",
            "confidence": 0.8,
            "reasons": ["r"],
        },
        {
            "concept_id": "z",
            "concept_label": "Z",
            "slide_indexes": [],
            "score": 0.9,
            "label": "High",
            "confidence": 0.8,
            "reasons": ["r"],
        },
        {
            "concept_id": "n",
            "concept_label": "N",
            "slide_indexes": [2],
            "score": None,
            "label": "Medium",
            "confidence": 0.0,
            "reasons": [],
        },
    ]
    out = hs.build_hotspots(crow, rows, 2)
    assert [(h["rank"], h["kind"]) for h in out][:2] == [(1, "concept"), (2, "concept")]
    assert out[0]["concept_id"] == "a"
    assert out[1]["concept_id"] == "z"
    assert {h["kind"] for h in out} == {"concept", "slide"}
    assert all("slide_index" in h for h in out if h["kind"] == "slide")
