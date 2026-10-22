import json

import pytest

from edusync_c2_model.demo_eval import DemoEvalError, evaluate_demo

HEADER = "snippet_id,lecture_id,slide_index,rater_id,rating,illustrative\n"


def _ratings(tmp_path, rows, header=HEADER):
    path = tmp_path / "r.csv"
    body = "".join(f"{r}\n" for r in rows)
    path.write_text("# comment line\n" + header + body, encoding="utf-8")
    return path


def _analysis(tmp_path, slides):
    path = tmp_path / "a.json"
    scores = [{"index": i, "score": s, "label": lab} for i, s, lab in slides]
    path.write_text(json.dumps({"final": {"slide_scores": scores}}), encoding="utf-8")
    return path


def test_evaluate_demo_aggregates(tmp_path):
    ratings = _ratings(
        tmp_path,
        [
            "D1,L,1,x,Low,true",
            "D2,L,2,x,Medium,true",
            "D3,L,3,x,high,true",
            "D4,L,4,x,Low,true",
            "D5,L,9,x,Low,true",  # slide absent from the analysis
            "D6,L,,x,Low,true",  # no slide number
            "D7,L,5,x,Low,true",  # model label outside the three classes
        ],
    )
    analysis = _analysis(
        tmp_path,
        [(1, 0.1, "Low"), (2, 0.5, "Medium"), (3, 0.6, "Medium"), (4, 0.9, "High"), (5, 0.2, "?")],
    )
    out = evaluate_demo(ratings, analysis, n_boot=50)
    assert out["n"] == 4
    assert out["n_missing"] == 3
    assert out["missing_ids"] == ["D5", "D6", "D7"]
    assert out["agreement_rate"] == 0.5
    assert out["adjacent_agreement"] == 0.75
    assert sum(map(sum, out["confusion"])) == 4
    assert out["spearman"] is not None
    assert out["qwk"] is not None
    assert [p["id"] for p in out["per_snippet"]] == ["D1", "D2", "D3", "D4"]
    assert out["per_snippet"][2]["demo_label"] == "High"
    assert out["illustrative"] is True
    assert "text" not in out["per_snippet"][0]


def test_evaluate_demo_nothing_matches(tmp_path):
    ratings = _ratings(tmp_path, ["D1,L,7,x,Low,true"])
    out = evaluate_demo(ratings, _analysis(tmp_path, [(1, 0.1, "Low")]))
    assert out["n"] == 0
    assert out["spearman"] is None
    assert out["confusion"] is None
    assert out["n_missing"] == 1


def test_evaluate_demo_rejects_bad_inputs(tmp_path):
    analysis = _analysis(tmp_path, [(1, 0.1, "Low")])
    with pytest.raises(DemoEvalError):
        evaluate_demo(_ratings(tmp_path, [], header=HEADER), analysis)
    with pytest.raises(DemoEvalError):
        evaluate_demo(_ratings(tmp_path, ["D1,L,1,x,Hard,true"]), analysis)
    good = _ratings(tmp_path, ["D1,L,1,x,Low,true"])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"final": {}}), encoding="utf-8")
    with pytest.raises(DemoEvalError):
        evaluate_demo(good, bad)
