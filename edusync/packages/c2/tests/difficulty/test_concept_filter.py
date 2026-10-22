"""The heuristic concept filter: generic drop, normalisation merge, abbreviation merge."""

import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from edusync_c2.difficulty import concept_filter as cf
from edusync_c2.difficulty.concept_filter import HeuristicConceptFilter
from edusync_c2.difficulty.concepts import FilterContext
from edusync_c2.difficulty.ids import casefold_id
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    C1GraphResponse,
    ConceptRecord,
    InputSource,
    SlideRecord,
)

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
SOURCE = InputSource(name="c1_graph", kind="real")


def rec(
    label: str,
    freq: int = 1,
    slides: tuple[int, ...] = (),
    *,
    cid: str | None = None,
) -> ConceptRecord:
    return ConceptRecord(
        id=cid or casefold_id(label),
        label=label,
        frequency=freq,
        slide_indexes=list(slides),
        alignment_status=AlignmentStatus.ALIGNED if slides else AlignmentStatus.NOT_FOUND,
    )


def make_ctx(slide_texts: tuple[str, ...] = (), edges: tuple[tuple[str, str, str], ...] = ()):
    slides = tuple(SlideRecord(index=i + 1, text=t) for i, t in enumerate(slide_texts))
    graph = C1GraphResponse.model_validate(
        {
            "graph": {
                "nodes": [],
                "edges": [{"source": s, "target": t, "relation_type": r} for s, t, r in edges],
            }
        }
    )
    return FilterContext(graph=graph, slides=slides, graph_source=SOURCE)


def resolve(
    labels: list[str],
    slide_texts: tuple[str, ...] = (),
    edges: tuple[tuple[str, str, str], ...] = (),
    *,
    extras: bool = False,
) -> list[ConceptRecord]:
    records = [rec(label) for label in labels]
    return HeuristicConceptFilter(extras).resolve(records, make_ctx(slide_texts, edges))


def merged(out: list[ConceptRecord]) -> list[list[str]]:
    """Merged groups as sorted label lists, ordered."""
    return sorted(sorted([r.label, *r.alias_labels]) for r in out if r.merged_from)


def dropped(out: list[ConceptRecord]) -> list[str]:
    return [r.label for r in out if r.filter_status == "dropped"]


def kept_labels(out: list[ConceptRecord]) -> list[str]:
    return [r.label for r in out if r.filter_status == "kept"]


# ====================================================================== group 4: generic drop
def test_flags_and_input_not_mutated() -> None:
    flt = HeuristicConceptFilter()
    assert (flt.name, flt.is_mock, flt.extras) == ("heuristic", False, False)
    assert HeuristicConceptFilter(True).extras is True
    records = [rec("Overview"), rec("Gradient Descent")]
    before = [r.model_copy() for r in records]
    flt.resolve(records, make_ctx())
    assert records == before


@pytest.mark.parametrize(
    "label",
    [
        "Overview",
        "Introduction",
        "Summary",
        "Agenda",
        "Outline",
        "Conclusion",
        "Conclusions",
        "Question",
        "Questions",
        "Q&A",
        "Q & A",
        "Thank You",
        "Thanks",
        "Objective",
        "Objectives",
        "Learning Objectives",
        "Learning Outcomes",
        "Next Steps",
        "Today",
        "Recap",
        "Key Takeaways",
        "Table of Contents",
        "Roadmap",
        "Further Reading",
        "Welcome",
        "Best Practices",
        "Best Practice",
        "Homework",
        "the Agenda",
        "  OVERVIEW ",
        "Summaries",
    ],
)
def test_safe_tier_is_dropped(label: str) -> None:
    out = resolve([label, "Gradient Descent"])
    assert dropped(out) == [label]
    assert out[0].filter_note == "generic slide phrase"
    assert out[1].filter_status == "kept"


@pytest.mark.parametrize(
    "label",
    [
        "Reference",
        "References",
        "Resource",
        "Resources",
        "Background",
        "Review",
        "Discussion",
        "Motivation",
        "Summary Statistics",
        "Introduction to Algorithms",
        "Example-Based Learning",
        "Overview Plus Detail",
        "Question Answering",
        "QA",
        "!!!",
    ],
)
def test_never_dropped(label: str) -> None:
    for extras in (False, True):
        assert dropped(resolve([label], extras=extras)) == []


@pytest.mark.parametrize(
    "label",
    [
        *("Example", "Examples", "Definition", "Definitions", "Exercise", "Exercises"),
        *("Demo", "Quiz", "Quizzes", "Lab", "Labs"),
    ],
)
def test_extras_tier_only_when_enabled(label: str) -> None:
    assert dropped(resolve([label])) == []
    assert dropped(resolve([label], extras=True)) == [label]


def test_dropped_records_take_no_part_in_merging() -> None:
    out = resolve(["Best Practices", "Best Practice", "Conclusion", "Conclusions"])
    assert dropped(out) == ["Best Practices", "Best Practice", "Conclusion", "Conclusions"]
    assert merged(out) == []


def test_dropped_keep_their_position_and_other_fields() -> None:
    records = [rec("A Topic", 3, (1, 2)), rec("Overview", 4, (1,)), rec("Other Topic")]
    out = HeuristicConceptFilter().resolve(records, make_ctx())
    assert [r.label for r in out] == ["A Topic", "Overview", "Other Topic"]
    assert (out[1].frequency, out[1].slide_indexes, out[1].filter_status) == (4, [1], "dropped")


def test_generic_tiers_are_disjoint_and_normalised() -> None:
    assert not cf.SAFE_GENERIC & cf.EXTRAS_GENERIC
    assert cf.is_generic_label("Learning  Objectives")
    assert not cf.is_generic_label("")


# ====================================================================== group 5: normalisation
@pytest.mark.parametrize(
    "labels",
    [
        ["Real-Time", "Real Time", "Realtime"],
        ["Pre-Order", "Preorder", "Pre Order"],
        ["The SQL", "SQL"],
        ["An Index", "Index"],
        ["Dijkstra's Algorithm", "Dijkstra Algorithm"],
        ["Dijkstra\u2019s Algorithm", "Dijkstra Algorithm"],
        ["Three-Way Handshake", "3-Way Handshake"],
        ["Ten Fold", "10 Fold"],
        ["Optimisation", "Optimization"],
        ["Neural Network", "Neural Networks"],
        ["Hash", "Hashes"],
        ["Policy", "Policies"],
        ["Index", "Indexes", "Indices"],
        ["Matrix", "Matrices"],
        ["Analysis", "Analyses"],
        ["Datum", "Data"],
        ["Cluster Centroid", "Cluster Centroids"],
        ["Multi-Threading", "Multithreading", "Multi Threading"],
    ],
)
def test_normalisation_merges(labels: list[str]) -> None:
    out = resolve(labels)
    assert merged(out) == [sorted(labels)]
    assert len(out) == 1


@pytest.mark.parametrize(
    "labels",
    [
        ["First Normal Form", "1 Normal Form"],
        ["Process", "Processing"],
        ["Cache", "Cache Coherence"],
        ["Data Encryption", "Transparent Data Encryption"],
        ["Gradient Descent", "Stochastic Gradient Descent"],
        ["Key", "Public Key"],
        ["C", "C++"],
        ["C#", "C"],
        ["Series", "Time Series"],
        ["Queue", "Queueing"],
        ["Hash", "Hashing"],
        ["Class", "Pass"],
        ["Hashes", "HASH"],
        ["---", "!!!"],
    ],
)
def test_normalisation_negatives(labels: list[str]) -> None:
    assert merged(resolve(labels)) == []


def test_ordinals_are_not_number_words() -> None:
    assert cf._tokens("First Normal Form") == ("first", "normal", "form")
    assert cf._tokens("Three Way") == ("3", "way")
    assert cf._tokens("The") == ("the",)


def test_known_limit_window_windows_false_merge() -> None:
    """Known limitation: the existence-based plural rule cannot tell Window from Windows."""
    assert merged(resolve(["Window", "Windows"])) == [["Window", "Windows"]]


def test_plural_rule_skips_abbreviations() -> None:
    assert merged(resolve(["HTTP", "HTTPS"])) == []
    assert merged(resolve(["DNS", "DN"])) == []
    assert merged(resolve(["OS", "O"])) == []


def test_plural_rule_needs_the_singular_to_exist() -> None:
    assert merged(resolve(["Neural Networks", "Decision Trees"])) == []


def test_canonical_is_highest_frequency_then_slides_then_longest_then_id() -> None:
    flt = HeuristicConceptFilter()
    ctx = make_ctx()
    out = flt.resolve([rec("Index", 1), rec("Indexes", 5)], ctx)
    assert out[0].id == "indexes"
    out = flt.resolve([rec("Index", 5, (1, 2)), rec("Indexes", 5, (3,))], ctx)
    assert out[0].id == "index"
    out = flt.resolve([rec("Index", 5, (1,)), rec("Indexes", 5, (3,))], ctx)
    assert out[0].id == "indexes"  # frequency and slide hits tie: the longer label
    out = flt.resolve([rec("Real-Time", 2), rec("Real Time", 2)], ctx)
    assert out[0].id == "real time"  # everything ties: the smallest id
    out = flt.resolve([rec("Neural Networks", 3), rec("Neural Network", 3)], ctx)
    assert out[0].id == "neural networks"


def test_canonical_record_fields() -> None:
    records = [
        rec("Neural Network", 4, (1, 3)),
        rec("Neural Networks", 2, (3, 5)),
        rec("neural-network", 1),
    ]
    out = HeuristicConceptFilter().resolve(records, make_ctx())
    assert len(out) == 1
    r = out[0]
    assert r.id == "neural network"
    assert r.merged_from == ["neural networks", "neural-network"]
    assert r.alias_labels == ["Neural Networks", "neural-network"]
    assert r.frequency == 4  # max over members, not the sum
    assert r.slide_indexes == [1, 3, 5]
    assert (r.alignment_status, r.filter_status) == (AlignmentStatus.ALIGNED, "kept")
    assert r.filter_note == "merged 2 aliases (normalisation)"


def test_frequency_is_max_even_when_canonical_has_the_lower_value() -> None:
    out = HeuristicConceptFilter().resolve(
        [rec("SQL", 5), rec("Structured Query Language", 3)], make_ctx()
    )
    assert (out[0].label, out[0].frequency) == ("Structured Query Language", 5)


def test_alignment_is_aligned_if_any_member_aligned() -> None:
    flt = HeuristicConceptFilter()
    out = flt.resolve([rec("Index"), rec("Indexes", 1, (2,))], make_ctx())
    assert out[0].alignment_status is AlignmentStatus.ALIGNED
    out = flt.resolve([rec("Index"), rec("Indexes")], make_ctx())
    assert out[0].alignment_status is AlignmentStatus.NOT_FOUND


def test_single_alias_note_is_singular() -> None:
    out = resolve(["Index", "Indexes"])
    assert out[0].filter_note == "merged 1 alias (normalisation)"


def test_output_order_follows_the_earliest_member_not_the_canonical() -> None:
    records = [rec("Index", 1), rec("Other Topic"), rec("Indexes", 9), rec("Last Topic")]
    out = HeuristicConceptFilter().resolve(records, make_ctx())
    assert [r.id for r in out] == ["indexes", "other topic", "last topic"]


def test_normalisation_cluster_cap(caplog: pytest.LogCaptureFixture) -> None:
    labels = ["Hash Table", "Hash Tables", "Hash-Table", "Hashtable", "The Hash Table"]
    with caplog.at_level(logging.DEBUG, logger="edusync_c2.difficulty.concept_filter"):
        out = resolve(labels)
    assert merged(out) == []
    assert kept_labels(out) == labels
    assert "left unmerged" in caplog.text
    assert merged(resolve(labels[:4])) == [sorted(labels[:4])]


def test_no_slide_text_in_filter_note() -> None:
    secret = "Confidential Quarterly Merger Plan"
    out = resolve(
        ["TDE", "Transparent Data Encryption"],
        (f"{secret}. Transparent Data Encryption (TDE) is on.",),
    )
    assert out[0].merged_from == ["tde"]
    assert secret.casefold() not in out[0].filter_note.casefold()
    assert out[0].filter_note == "merged 1 alias (abbreviation)"


# ====================================================================== group 6: abbreviations
@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["SQL", "Structured Query Language"], [["SQL", "Structured Query Language"]]),
        (["TDE", "Transparent Data Encryption"], [["TDE", "Transparent Data Encryption"]]),
        (["BFS", "Breadth-First Search"], [["BFS", "Breadth-First Search"]]),
        (["CNN", "Convolutional Neural Network"], [["CNN", "Convolutional Neural Network"]]),
        (["DMS", "Document Management System"], [["DMS", "Document Management System"]]),
        (["OOP", "Object-Oriented Programming"], [["OOP", "Object-Oriented Programming"]]),
        (["IaaS", "Infrastructure as a Service"], [["IaaS", "Infrastructure as a Service"]]),
        (["NoSQL", "Not Only SQL"], [["NoSQL", "Not Only SQL"]]),
        (["LSTM", "Long Short-Term Memory"], [["LSTM", "Long Short-Term Memory"]]),
        (["TLS", "Transport Layer Security"], [["TLS", "Transport Layer Security"]]),
        (["1NF", "First Normal Form"], [["1NF", "First Normal Form"]]),
        (["3DES", "Triple DES"], [["3DES", "Triple DES"]]),
        (["2PL", "Two-Phase Locking"], [["2PL", "Two-Phase Locking"]]),
        (["OS", "Operating System"], [["OS", "Operating System"]]),
        (["IP", "Internet Protocol"], [["IP", "Internet Protocol"]]),
        (["AI", "Artificial Intelligence"], [["AI", "Artificial Intelligence"]]),
        (["k-NN", "K-Nearest Neighbors"], [["K-Nearest Neighbors", "k-NN"]]),
        (["PKI", "Public Key Infrastructure"], [["PKI", "Public Key Infrastructure"]]),
        (["TLB", "Translation Lookaside Buffer"], [["TLB", "Translation Lookaside Buffer"]]),
        (["Mean Absolute Error", "MAE"], [["MAE", "Mean Absolute Error"]]),
    ],
)
def test_abbreviation_merges(labels: list[str], expected: list[list[str]]) -> None:
    assert merged(resolve(labels)) == expected


def test_abbreviation_with_two_expansion_variants() -> None:
    out = resolve(["CNN", "Convolutional Neural Networks", "Convolutional Neural Network"])
    assert merged(out) == [["CNN", "Convolutional Neural Network", "Convolutional Neural Networks"]]
    assert out[0].label == "Convolutional Neural Networks"  # expansion side, longest label


def test_abbreviation_with_two_independent_pairs() -> None:
    out = resolve(["BFS", "DFS", "Breadth-First Search", "Depth-First Search"])
    assert merged(out) == [["BFS", "Breadth-First Search"], ["DFS", "Depth-First Search"]]


def test_abbreviation_canonical_is_the_expansion() -> None:
    records = [rec("SQL", 9, (1, 2, 3)), rec("Structured Query Language", 1)]
    out = HeuristicConceptFilter().resolve(records, make_ctx())
    assert len(out) == 1
    assert (out[0].label, out[0].merged_from, out[0].alias_labels) == (
        "Structured Query Language",
        ["sql"],
        ["SQL"],
    )
    assert out[0].slide_indexes == [1, 2, 3]
    assert out[0].alignment_status is AlignmentStatus.ALIGNED


def test_abbreviation_canonical_among_expansion_variants_uses_the_ranking() -> None:
    records = [
        rec("CNN", 9),
        rec("Convolutional Neural Network", 1),
        rec("Convolutional Neural Networks", 4),
    ]
    out = HeuristicConceptFilter().resolve(records, make_ctx())
    assert out[0].id == "convolutional neural networks"
    assert out[0].merged_from == ["cnn", "convolutional neural network"]
    assert out[0].filter_note == "merged 2 aliases (normalisation + abbreviation)"


def test_abbreviation_and_expansion_ids_are_kept_as_given() -> None:
    records = [rec("TDE", cid="node-7"), rec("Transparent Data Encryption", cid="node-3")]
    out = HeuristicConceptFilter().resolve(records, make_ctx())
    assert (out[0].id, out[0].merged_from) == ("node-3", ["node-7"])


# --- the guard cases: an abbreviation that fits two expansions merges neither
@pytest.mark.parametrize(
    "labels",
    [
        ["LR", "Linear Regression", "Logistic Regression"],
        ["LR", "L1 Regularization"],
        ["LR", "Linear Regression", "Logistic Regression", "L1 Regularization"],
        ["CD", "Continuous Delivery", "Continuous Deployment"],
        ["ML", "Machine Learning", "Maximum Likelihood"],
        ["TDE", "Transparent Data Encryption", "Triple Data Encryption"],
        ["OS", "Operating System", "Open Source"],
        ["IP", "Internet Protocol", "Intellectual Property"],
    ],
)
def test_guard_blocks_ambiguous_abbreviations(labels: list[str]) -> None:
    assert merged(resolve(labels)) == []


def test_digit_rule_blocks_l1_regularization_but_allows_a_digit_abbreviation() -> None:
    assert merged(resolve(["LR", "L1 Regularization"])) == []
    assert merged(resolve(["L1R", "L1 Regularization"])) == [["L1 Regularization", "L1R"]]
    assert merged(resolve(["LR", "Linear Regression"])) == [["LR", "Linear Regression"]]


def test_expansion_claimed_by_two_abbreviations_merges_neither() -> None:
    assert merged(resolve(["BFS", "BS", "Breadth-First Search"])) == []
    assert merged(resolve(["BFS", "Breadth-First Search"])) != []


def test_slide_evidence_selects_exactly_one_expansion() -> None:
    labels = ["TDE", "Transparent Data Encryption", "Triple Data Encryption"]
    out = resolve(labels, ("Transparent Data Encryption (TDE) encrypts files at rest.",))
    assert merged(out) == [["TDE", "Transparent Data Encryption"]]
    assert "Triple Data Encryption" in kept_labels(out)
    out = resolve(labels, ("Triple Data Encryption (TDE) is old.",))
    assert merged(out) == [["TDE", "Triple Data Encryption"]]


def test_slide_evidence_for_both_expansions_merges_none() -> None:
    labels = ["TDE", "Transparent Data Encryption", "Triple Data Encryption"]
    slides = ("Triple Data Encryption (TDE) and Transparent Data Encryption (TDE) differ.",)
    assert merged(resolve(labels, slides)) == []


def test_evidence_in_the_slide_title_and_either_direction() -> None:
    labels = ["ReLU", "Rectified Linear Unit"]
    assert merged(resolve(labels)) == []  # initials alone do not spell ReLU
    forward = (SlideRecord(index=1, title="Rectified Linear Unit (ReLU)", text="body"),)
    ctx = FilterContext(graph=make_ctx().graph, slides=forward, graph_source=SOURCE)
    out = HeuristicConceptFilter().resolve([rec(x) for x in labels], ctx)
    assert merged(out) == [["ReLU", "Rectified Linear Unit"]]
    out = resolve(labels, ("ReLU (Rectified Linear Unit) is an activation.",))
    assert merged(out) == [["ReLU", "Rectified Linear Unit"]]
    out = resolve(labels, ("ReLU (Rectified Linear Unit)",))
    assert merged(out) == [["ReLU", "Rectified Linear Unit"]]


def test_evidence_needs_a_definition_pattern() -> None:
    labels = ["ReLU", "Rectified Linear Unit"]
    for text in (
        "ReLU and Rectified Linear Unit are mentioned.",  # no parentheses
        "Rectified Linear, Unit (ReLU) is split by a comma.",
        "(ReLU) opens the slide",  # nothing before the bracket
        "Unit (ReLU)",  # a single word before the bracket
        "ReLU (Rectified, Linear Unit)",  # a comma inside the expansion
        "ReLU (Rectified Linear)",  # partial expansion
        "Details (see below) for Rectified Linear Unit",
    ):
        assert merged(resolve(labels, (text,))) == []


def test_evidence_is_only_for_abbreviation_shaped_labels() -> None:
    out = resolve(["Backprop", "Backpropagation"], ("Backpropagation (Backprop) is key",))
    assert merged(out) == []
    out = resolve(
        ["Data Encryption", "Transparent Data Encryption"], ("Transparent (Data Encryption)",)
    )
    assert merged(out) == []


def test_evidence_and_no_labels_to_pair_is_a_no_op() -> None:
    assert merged(resolve(["Alpha", "Beta"], ("Alpha (Beta)",))) == []
    assert merged(resolve(["TDE"], ("(TDE)",))) == []


def test_evidence_whose_expansion_is_already_in_the_same_cluster() -> None:
    out = resolve(["SQL", "S Q L"], ("S Q L (SQL)",))
    assert merged(out) == [["S Q L", "SQL"]]
    out = resolve(
        ["SQL", "The SQL", "Structured Query Language"], ("Structured Query Language (SQL)",)
    )
    assert merged(out) == [["SQL", "Structured Query Language", "The SQL"]]


# --- the negative battery
@pytest.mark.parametrize(
    "labels",
    [
        ["Data Encryption", "Transparent Data Encryption"],
        ["Gradient Descent", "Stochastic Gradient Descent"],
        ["Key", "Public Key"],
        ["SQL", "SQL Injection"],
        ["HTTP", "HTTPS"],
        ["Process", "Processing"],
        ["Cache", "Cache Coherence"],
        ["AES", "Advanced Data Encryption"],
        ["ACID", "Atomicity"],
        ["ACID", "Atomicity", "Consistency", "Isolation", "Durability"],
        ["Subnet", "Subnet Mask"],
        ["DNS", "DN"],
        ["SGD", "Gradient Descent"],
        ["AES", "Advanced Encryption"],
    ],
)
def test_negative_battery(labels: list[str]) -> None:
    assert merged(resolve(labels)) == []


def test_aes_merges_with_the_right_expansion_only() -> None:
    labels = ["AES", "Advanced Encryption Standard", "Advanced Data Encryption"]
    assert merged(resolve(labels)) == [["AES", "Advanced Encryption Standard"]]


def test_acid_names_all_four_properties_when_given_as_one_label() -> None:
    labels = ["ACID", "Atomicity Consistency Isolation Durability"]
    assert merged(resolve(labels)) == [sorted(labels)]


def test_abbreviation_shape() -> None:
    for yes in ("TDE", "ReLU", "k-NN", "3DES", "NoSQL", "DoS", "AES-256", "OS"):
        assert cf.is_abbreviation(yes), yes
    for no in ("Cache", "Backprop", "X", "P-Value", "E-Mail", "TDE Key", "ABCDEFG", "Q&A", "Ab"):
        assert not cf.is_abbreviation(no), no


def test_expansions_need_two_words() -> None:
    assert merged(resolve(["TS", "The SQL"])) == []
    assert merged(resolve(["TS", "SQL"])) == []
    assert merged(resolve(["Of The", "XY"])) == []  # all stop words: the empty initials are skipped


def test_initial_variants() -> None:
    assert "tde" in cf._initial_variants("Transparent Data Encryption", False)
    assert "ie" in cf._initial_variants("Introduction of Encoding", True)
    assert "ioe" in cf._initial_variants("Introduction of Encoding", False)
    assert "bfs" in cf._initial_variants("Breadth-First Search", False)
    assert "bs" in cf._initial_variants("Breadth-First Search", False)  # first part of a compound
    assert "3des" in cf._initial_variants("Triple DES", False)
    assert "nosql" in cf._initial_variants("Not Only SQL", False)
    assert cf._initial_variants("Of The", True) == frozenset({""})
    long = cf._initial_variants("Alpha Beta Gamma Delta Epsilon Zeta Eta Theta", False)
    assert long == frozenset()  # every variant would be longer than six characters
    assert all(
        len(v) <= 6 for v in cf._initial_variants("One-Two-Three Four-Five-Six Seven", False)
    )


def test_part_count_and_digit_helpers() -> None:
    assert cf._part_count("Breadth-First Search") == 3
    assert cf._has_digit("Layer 3")
    assert not cf._has_digit("Layer")


def test_expansion_with_too_many_parts_is_ignored() -> None:
    long_label = "One Two Three Four Five Six Seven Eight Nine"
    assert cf._part_count(long_label) > cf._EXPANSION_MAX_PARTS
    assert merged(resolve(["OTTFFS", long_label])) == []


# --- prerequisite veto, cluster cap, chains
def test_prerequisite_edge_vetoes_the_merge_in_either_direction() -> None:
    labels = ["SQL", "Structured Query Language"]
    assert merged(resolve(labels)) == [sorted(labels)]
    for src, dst in (("sql", "structured query language"), ("structured query language", "sql")):
        edges = ((src, dst, "prerequisite"),)
        assert merged(resolve(labels, edges=edges)) == []
        assert merged(resolve(labels, ("Structured Query Language (SQL)",), edges)) == []
    co = (("sql", "structured query language", "co_mentioned"),)
    assert merged(resolve(labels, edges=co)) == [sorted(labels)]
    other = (("sql", "database", "prerequisite"),)
    assert merged(resolve(labels, edges=other)) == [sorted(labels)]


def test_prerequisite_veto_looks_at_every_member_of_the_clusters() -> None:
    labels = ["SQL", "Structured Query Language", "Structured Query Languages"]
    edges = (("sql", "structured query languages", "prerequisite"),)
    assert merged(resolve(labels, edges=edges)) == [
        ["Structured Query Language", "Structured Query Languages"]
    ]


def test_merge_chain_expansion_plural_and_abbreviation() -> None:
    labels = ["The SQL", "SQL", "Structured Query Language", "Structured Query Languages"]
    out = resolve(labels)
    assert len(out) == 1
    assert out[0].label == "Structured Query Languages"  # the expansion side, longest label
    assert out[0].alias_labels == ["SQL", "Structured Query Language", "The SQL"]
    assert out[0].merged_from == ["sql", "structured query language", "the sql"]


def test_abbreviation_cluster_cap(caplog: pytest.LogCaptureFixture) -> None:
    labels = ["The SQL", "SQL", "Structured Query Language", "Structured Query Languages"]
    labels.append("The Structured Query Language")
    with caplog.at_level(logging.DEBUG, logger="edusync_c2.difficulty.concept_filter"):
        out = resolve(labels)
    # the abbreviation edge would build a cluster of five: the normalisation groups stay as they are
    assert merged(out) == [
        ["SQL", "The SQL"],
        [
            "Structured Query Language",
            "Structured Query Languages",
            "The Structured Query Language",
        ],
    ]
    assert "abbreviation cluster of 5" in caplog.text


def test_cluster_of_exactly_the_cap_is_merged() -> None:
    labels = [
        "SQL",
        "Structured Query Language",
        "Structured Query Languages",
        "The Structured Query Language",
    ]
    out = resolve(labels)
    assert len(out) == 1
    assert len(out[0].merged_from) == cf.MAX_CLUSTER - 1


# --- determinism
LABELS = [
    "SQL",
    "Structured Query Language",
    "TDE",
    "Transparent Data Encryption",
    "Triple Data Encryption",
    "LR",
    "Linear Regression",
    "Logistic Regression",
    "Neural Network",
    "Neural Networks",
    "CNN",
    "Convolutional Neural Network",
    "Real-Time",
    "Real Time",
    "Realtime",
    "Best Practices",
    "Overview",
    "Index",
    "Indexes",
    "Indices",
    "Key",
    "Public Key",
    "ReLU",
    "Rectified Linear Unit",
    "Three-Way Handshake",
    "3-Way Handshake",
]
SLIDES = ("Transparent Data Encryption (TDE) at rest.", "ReLU (Rectified Linear Unit) layers.")


def summary(out: list[ConceptRecord]) -> set[tuple]:
    return {
        (r.id, r.filter_status, tuple(r.merged_from), tuple(r.alias_labels), r.frequency)
        for r in out
    }


def test_shuffled_input_gives_the_same_canonical_set() -> None:
    base = [rec(label, freq=i % 5 + 1, slides=(i % 3 + 1,)) for i, label in enumerate(LABELS)]
    ctx = make_ctx(SLIDES)
    reference = summary(HeuristicConceptFilter().resolve(base, ctx))
    assert any(row[2] for row in reference)
    for seed in range(12):
        shuffled = list(base)
        random.Random(seed).shuffle(shuffled)
        assert summary(HeuristicConceptFilter().resolve(shuffled, ctx)) == reference


def test_output_order_is_the_order_of_the_earliest_member() -> None:
    base = [rec(label) for label in LABELS]
    ctx = make_ctx(SLIDES)
    out = HeuristicConceptFilter().resolve(base, ctx)
    ids = [r.id for r in base]
    firsts = []
    for r in out:
        members = [r.id, *r.merged_from]
        firsts.append(min(ids.index(m) for m in members))
    assert firsts == sorted(firsts)


CHILD = """
import json
from edusync_c2.difficulty.concept_filter import HeuristicConceptFilter
from edusync_c2.difficulty.concepts import FilterContext
from edusync_c2.difficulty.ids import casefold_id
from edusync_c2.difficulty.schemas import (
    AlignmentStatus, C1GraphResponse, ConceptRecord, InputSource, SlideRecord)
labels = json.loads(__LABELS__)
slides = tuple(SlideRecord(index=i + 1, text=t) for i, t in enumerate(json.loads(__SLIDES__)))
records = [ConceptRecord(id=casefold_id(l), label=l, frequency=i % 4 + 1,
           alignment_status=AlignmentStatus.NOT_FOUND) for i, l in enumerate(labels)]
ctx = FilterContext(graph=C1GraphResponse.model_validate({"graph": {}}), slides=slides,
                    graph_source=InputSource(name="c1_graph", kind="real"))
out = HeuristicConceptFilter(True).resolve(records, ctx)
print(json.dumps([r.model_dump(mode="json") for r in out], sort_keys=True))
"""


def run_child(seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR), *sys.path])
    code = CHILD.replace("__LABELS__", repr(json.dumps(LABELS))).replace(
        "__SLIDES__", repr(json.dumps(list(SLIDES)))
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return done.stdout.strip()


def test_same_result_under_different_hash_seeds() -> None:
    first, second = run_child("0"), run_child("1")
    assert first == second
    rows = json.loads(first)
    assert any(r["merged_from"] for r in rows)
    assert any(r["filter_status"] == "dropped" for r in rows)


# --- performance guard
def test_prefilter_keeps_candidate_pairs_small_for_many_labels() -> None:
    abbrs = [a + b for a in "ABCDEF" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"][:150]
    expansions = [f"Zeta Gamma{k}" for k in range(145)]
    ab = [f"Alpha {w}" for w in ("Beta", "Basic", "Binary", "Bounded", "Broad")]
    labels = [*abbrs, *expansions, *ab]
    assert len(labels) == 300
    assert len(set(labels)) == 300
    pairs = HeuristicConceptFilter.candidate_pairs(labels)
    assert pairs == 5  # only "AB" and the five "Alpha B..." labels meet; not 150 * 150
    out = HeuristicConceptFilter().resolve([rec(x) for x in labels], make_ctx())
    assert len(out) == 300  # "AB" fits five expansions: ambiguous, nothing merges


def test_prefilter_on_a_realistic_list() -> None:
    labels = ["SQL", "Structured Query Language", "Key", "Public Key", "TDE"]
    assert HeuristicConceptFilter.candidate_pairs(labels) == 1


# --- helpers coverage
def test_singular_forms() -> None:
    assert cf._singular_forms("policies")[0] == "policy"
    assert cf._singular_forms("data") == ["datum"]
    assert cf._singular_forms("boxes") == ["box", "boxe"]
    assert cf._singular_forms("class") == []
    assert cf._singular_forms("bus") == []
    assert cf._singular_forms("cats") == ["cat"]
    assert cf._singular_forms("ies") == []
