"""Pairwise precision and recall of the heuristic filter on realistic label lists.

``golden/concept_label_lists.json`` holds hand-written label lists of the kind C1 produces for a
lecture (SQL / Structured Query Language, TDE with two possible expansions, plural pairs, generic
slide phrases ...), each with the merges and the generic phrases a person would expect.

The recall bound is optimistic: the rules were tuned on lists like these, so a real C1 output will
score lower. Precision is the number to trust, and it must stay at exactly 1.0 here.
"""

import itertools
import json
from pathlib import Path

import pytest

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

GOLDEN = Path(__file__).parent / "golden" / "concept_label_lists.json"
LISTS = json.loads(GOLDEN.read_text(encoding="utf-8"))["lists"]


def run(entry: dict) -> list[ConceptRecord]:
    seen: dict[str, str] = {}
    for label in entry["labels"]:
        seen.setdefault(casefold_id(label), label)
    records = [
        ConceptRecord(id=k, label=v, frequency=1, alignment_status=AlignmentStatus.NOT_FOUND)
        for k, v in seen.items()
    ]
    slides = (SlideRecord(index=1, text=entry["slides"]),) if entry["slides"] else ()
    ctx = FilterContext(
        graph=C1GraphResponse.model_validate({"graph": {}}),
        slides=slides,
        graph_source=InputSource(name="c1_graph", kind="real"),
    )
    return HeuristicConceptFilter().resolve(records, ctx)


def pairs(groups: list[list[str]]) -> set[frozenset[str]]:
    return {frozenset(p) for g in groups for p in itertools.combinations(g, 2)}


def predicted(out: list[ConceptRecord]) -> set[frozenset[str]]:
    return pairs([[r.label, *r.alias_labels] for r in out if r.merged_from])


def test_there_are_enough_lists() -> None:
    assert len(LISTS) >= 15
    assert sum(len(x["merge_groups"]) for x in LISTS) >= 40


def test_precision_is_one_and_recall_is_high_on_the_committed_lists() -> None:
    tp = fp = fn = 0
    for entry in LISTS:
        got = predicted(run(entry))
        gold = pairs(entry["merge_groups"])
        tp += len(got & gold)
        fp += len(got - gold)
        fn += len(gold - got)
    assert fp == 0
    assert tp / (tp + fn) >= 0.9


@pytest.mark.parametrize("entry", LISTS, ids=[x["name"] for x in LISTS])
def test_each_list_has_no_false_merge_and_the_right_junk(entry: dict) -> None:
    out = run(entry)
    assert predicted(out) <= pairs(entry["merge_groups"])
    assert {r.label for r in out if r.filter_status == "dropped"} == set(entry["junk"])
    for left, right in entry["separate"]:
        assert frozenset((left, right)) not in predicted(out)


def test_dropped_and_absorbed_labels_account_for_every_input_label() -> None:
    for entry in LISTS:
        out = run(entry)
        seen = {casefold_id(x) for x in entry["labels"]}
        covered = {r.id for r in out} | {a for r in out for a in r.merged_from}
        assert covered == seen
