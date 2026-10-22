"""Prototype scorer: splitting, dedupe, budget, cache, thresholds, class assignment, rounding."""

import math
import re
import threading
from pathlib import Path

import numpy as np
import pytest

from edusync_c2.difficulty.features import semantic
from edusync_c2.difficulty.features.minilm import MiniLMEncoder
from edusync_c2.difficulty.features.semantic import (
    SemanticScorer,
    SentenceScore,
    load_prototypes,
    parse_prototypes,
    split_sentences,
)

VEC_RE = re.compile(r"v:([0-9.]+),([0-9.]+),([0-9.]+)")
REAL_DIR = (
    Path(__file__).resolve().parents[3]
    / "blueprint_model"
    / "artifacts"
    / "base"
    / "sentence-transformers--all-MiniLM-L6-v2"
)


class FakeEncoder:
    """Prototypes map to the axes e0/e1/e2; a test sentence carries ``v:d,e,s`` as its cosines."""

    dims = 4

    def __init__(self) -> None:
        protos = load_prototypes()
        self.axis = {
            text: i
            for i, name in enumerate(("definition", "example", "step"))
            for text in protos[name]
        }
        self.batches: list[list[str]] = []

    def encode(self, texts):
        self.batches.append(list(texts))
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for row, text in enumerate(texts):
            if text in self.axis:
                out[row, self.axis[text]] = 1.0
                continue
            match = VEC_RE.search(text)
            if match:
                d, e, s = (float(g) for g in match.groups())
                out[row, :3] = (d, e, s)
                out[row, 3] = math.sqrt(max(0.0, 1.0 - d * d - e * e - s * s))
            else:
                out[row, 3] = 1.0
        return out


def sentence(d: float, e: float, s: float = 0.0, tag: str = "x") -> str:
    return f"Sentence {tag} with some words v:{d},{e},{s}"


# --- splitting -------------------------------------------------------------------------------


def test_split_basic_punctuation_and_newlines():
    text = "First sentence here now. Second one is here! Is this third one? Last line has words"
    assert split_sentences(text) == [
        "First sentence here now.",
        "Second one is here!",
        "Is this third one?",
        "Last line has words",
    ]


def test_split_strips_bullets_and_numbering():
    text = (
        "- bullet text with words\n* star text with words\n1. numbered text with words\n"
        "2) next numbered item here\n(3) paren numbered item here\na) lettered item with words"
    )
    assert split_sentences(text) == [
        "bullet text with words",
        "star text with words",
        "numbered text with words",
        "next numbered item here",
        "paren numbered item here",
        "lettered item with words",
    ]


def test_split_strips_unicode_bullets():
    marks = [chr(0x2022), chr(0x25CF), chr(0x2013), chr(0xB7)]
    text = "\n".join(f"{m} item number {i} here" for i, m in enumerate(marks))
    assert split_sentences(text) == [f"item number {i} here" for i in range(4)]


def test_split_word_count_bounds_and_whitespace():
    long = " ".join(["word"] * 81)
    ok = " ".join(["word"] * 80)
    assert split_sentences(f"two words\n  three   spaced   words  \n{long}\n{ok}") == [
        "three spaced words",
        ok,
    ]
    assert split_sentences("") == []
    assert split_sentences("   \n\n") == []


def test_split_keeps_decimals_and_dedupes_case_insensitively_in_order():
    text = (
        "Pi is about 3.14 in value.\nPI IS ABOUT 3.14 IN VALUE.\n"
        "Another fine sentence here.\nPi is about 3.14 in value."
    )
    assert split_sentences(text) == ["Pi is about 3.14 in value.", "Another fine sentence here."]


def test_split_is_deterministic():
    text = "Alpha beta gamma. Delta epsilon zeta.\nEta theta iota."
    assert split_sentences(text) == split_sentences(text)


# --- prototypes ------------------------------------------------------------------------------


def test_packaged_prototypes_are_complete_ascii_and_distinct():
    protos = load_prototypes()
    assert set(protos) == {"definition", "example", "step"}
    everything = [s for name in protos for s in protos[name]]
    assert all(len(protos[name]) == 12 for name in protos)
    assert len(set(everything)) == 36
    assert all(s.isascii() for s in everything)
    raw = Path(semantic.__file__).parents[1] / "resources" / "semantic_prototypes.txt"
    assert b"\r" not in raw.read_bytes()


def test_prototypes_do_not_repeat_the_reference_sentences():
    import json

    gold = json.loads(
        (Path(__file__).parent / "golden" / "minilm_reference.json").read_text(encoding="utf-8")
    )
    refs = {c["text"] for c in gold["cases"]}
    assert not refs & {s for v in load_prototypes().values() for s in v}


def test_parse_prototypes_rules():
    text = "# comment\n\n[Definition]\n a \n[example]\nb\n[step]\nc\n"
    assert parse_prototypes(text) == {"definition": ["a"], "example": ["b"], "step": ["c"]}
    with pytest.raises(ValueError, match="before the first"):
        parse_prototypes("loose line\n[definition]\na")
    with pytest.raises(ValueError, match="missing or empty: step"):
        parse_prototypes("[definition]\na\n[example]\nb\n[step]\n")


# --- scoring ---------------------------------------------------------------------------------


def test_scores_are_cosines_rounded_to_four_decimals():
    scorer = SemanticScorer(FakeEncoder())
    (score,) = scorer.score_sentences([sentence(0.123456, 0.5, 0.25)])
    assert score.definition == 0.1235
    assert score.example == 0.5
    assert score.step == 0.25
    assert isinstance(score, SentenceScore)


def test_centroid_is_normalised_mean_of_prototypes():
    # Prototypes sit on one axis each, so a vector on that axis has cosine 1 to its centroid.
    scorer = SemanticScorer(FakeEncoder())
    proto = load_prototypes()["example"][0]
    (score,) = scorer.score_sentences([proto])
    assert (score.definition, score.example, score.step) == (0.0, 1.0, 0.0)


def test_thresholds_and_class_assignment():
    scorer = SemanticScorer(FakeEncoder(), def_threshold=0.3, ex_threshold=0.4)
    scores = scorer.score_sentences(
        [
            sentence(0.30, 0.10, tag="a"),  # definition, exactly at its threshold
            sentence(0.29, 0.10, tag="b"),  # below the definition threshold
            sentence(0.10, 0.40, tag="c"),  # example, exactly at its threshold
            sentence(0.10, 0.39, tag="d"),  # below the example threshold
            sentence(0.50, 0.50, tag="e"),  # tie: counts for neither
            sentence(0.60, 0.45, tag="f"),  # definition wins although both pass
            sentence(0.45, 0.60, tag="g"),  # example wins although both pass
            sentence(0.20, 0.20, 0.9, tag="h"),  # step-like: neither
        ]
    )
    assert scorer.count_semantic(scores) == (2, 2)
    assert scorer.count_semantic([]) == (0, 0)


def test_default_thresholds():
    scorer = SemanticScorer(FakeEncoder())
    assert (scorer.def_threshold, scorer.ex_threshold, scorer.max_sentences) == (0.23, 0.23, 400)
    scores = scorer.score_sentences([sentence(0.23, 0.0, tag="a"), sentence(0.0, 0.2299, tag="b")])
    assert scorer.count_semantic(scores) == (1, 0)


def test_unit_scores_are_deterministic_across_scorers():
    texts = [sentence(0.3, 0.1, tag=str(i)) for i in range(5)]
    first = SemanticScorer(FakeEncoder()).score_sentences(texts)
    second = SemanticScorer(FakeEncoder()).score_sentences(texts)
    assert first == second


# --- cache and budget ------------------------------------------------------------------------


def test_repeated_sentences_are_encoded_once():
    encoder = FakeEncoder()
    scorer = SemanticScorer(encoder)
    encoder.batches.clear()  # drop the prototype batches
    texts = [sentence(0.3, 0.1, tag="a"), sentence(0.1, 0.3, tag="b")]
    scorer.score_sentences(texts)
    scorer.score_sentences([texts[1], texts[0], texts[0]])
    assert encoder.batches == [texts]


def test_views_share_the_cache_but_not_the_budget():
    encoder = FakeEncoder()
    scorer = SemanticScorer(encoder, max_sentences=2)
    encoder.batches.clear()
    texts = [sentence(0.3, 0.1, tag=t) for t in "abc"]
    one = scorer.for_request()
    report = one.score_report(texts)
    assert [s.text for s in report.scores] == texts[:2]
    assert report.truncated
    assert one.truncated
    assert not scorer.truncated
    two = scorer.for_request()
    assert two.truncated is False
    assert [s.text for s in two.score_report(texts[:1]).scores] == texts[:1]
    assert encoder.batches == [texts[:2]]  # the second view hit the shared cache


def test_budget_counts_distinct_sentences_and_repeats_are_free():
    scorer = SemanticScorer(FakeEncoder(), max_sentences=2).for_request()
    a, b, c = (sentence(0.3, 0.1, tag=t) for t in "abc")
    assert len(scorer.score_sentences([a, b])) == 2
    report = scorer.score_report([a, c, b, a])
    assert [s.text for s in report.scores] == [a, b, a]
    assert report.truncated


def test_no_truncation_when_within_budget():
    scorer = SemanticScorer(FakeEncoder(), max_sentences=2).for_request()
    report = scorer.score_report([sentence(0.3, 0.1, tag="a")])
    assert not report.truncated
    assert not scorer.truncated


def test_cache_is_bounded_and_evicts_oldest(monkeypatch):
    monkeypatch.setattr(semantic, "CACHE_LIMIT", 3)
    encoder = FakeEncoder()
    scorer = SemanticScorer(encoder, max_sentences=100)
    texts = [sentence(0.3, 0.1, tag=str(i)) for i in range(5)]
    scorer.score_sentences(texts)
    assert len(scorer._shared.cache) == 3
    encoder.batches.clear()
    scorer.score_sentences(texts[-1:])  # newest is still cached
    assert encoder.batches == []
    scorer.score_sentences(texts[:1])  # oldest was evicted
    assert encoder.batches == [texts[:1]]


def test_concurrent_requests_share_one_scorer():
    scorer = SemanticScorer(FakeEncoder(), max_sentences=1000)
    texts = [sentence(0.3, 0.1, tag=str(i)) for i in range(30)]
    results: list[list[SentenceScore]] = []

    def work() -> None:
        results.append(scorer.for_request().score_sentences(texts))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8
    assert all(r == results[0] for r in results)


# --- real model ------------------------------------------------------------------------------


def real_weights_available() -> bool:
    from edusync_c2.difficulty.features.minilm import WEIGHTS_FILE, is_lfs_pointer

    weights = REAL_DIR / WEIGHTS_FILE
    return weights.is_file() and not is_lfs_pointer(weights)


@pytest.mark.ml
def test_real_encoder_separates_definition_and_example_sentences():
    if not real_weights_available():
        pytest.skip("real MiniLM weights are missing or an LFS pointer")
    scorer = SemanticScorer(MiniLMEncoder.from_directory(REAL_DIR))
    definition, example = scorer.score_sentences(
        [
            "A hash function maps data of arbitrary size to a fixed-size value.",
            "For example, consider sorting the list [4, 2, 7] with bubble sort.",
        ]
    )
    assert definition.definition > definition.example
    assert example.example > example.definition
