"""Heuristic concept filter: drops generic slide phrases and merges aliases of one concept.

Pure and deterministic: no clocks, no randomness, no iteration over unordered collections that
could change the result. The rules, in order:

1. Generic drop. A label that is, as a whole, a generic slide phrase (``Overview``, ``Best
   Practices``, ``Thank You`` ...) is returned with ``filter_status="dropped"``. Dropped records
   take no part in merging. ``Summary Statistics`` or ``Introduction to Algorithms`` stay because
   only the whole label is compared. An opt-in extras tier adds ``Example``, ``Definition`` ...
2. Normalisation merge: case, hyphens, whitespace, a leading determiner, possessives, number
   words (``three-way`` = ``3-way``), ``-isation``, run-together words (``real time`` =
   ``realtime``) and plurals whose singular exists as another label.
3. Abbreviation merge: an abbreviation-shaped label (``TDE``) joins a multi-word expansion whose
   initials spell it (``Transparent Data Encryption``). An abbreviation that fits two different
   expansions merges neither unless a slide says ``Expansion (ABBR)`` for exactly one. A
   prerequisite edge between the two vetoes the merge. Clusters larger than ``MAX_CLUSTER`` are
   not merged at all.

Known limit: the existence-based plural rule merges ``Window`` and ``Windows``.
"""

import itertools
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

from edusync_c2.difficulty.concepts import FilterContext
from edusync_c2.difficulty.ids import casefold_id
from edusync_c2.difficulty.schemas import AlignmentStatus, ConceptRecord, SlideRecord

logger = logging.getLogger(__name__)

MAX_CLUSTER = 4  # a cluster with more members than this is not merged at all
GENERIC_NOTE = "generic slide phrase"

# Whole-label generic phrases in singular, plural-folded form.
SAFE_GENERIC = frozenset(
    {
        "overview",
        "introduction",
        "summary",
        "agenda",
        "outline",
        "conclusion",
        "question",
        "q a",
        "thank you",
        "thanks",
        "objective",
        "learning objective",
        "learning outcome",
        "next step",
        "today",
        "recap",
        "key takeaway",
        "table of content",
        "roadmap",
        "further reading",
        "welcome",
        "best practice",
        "homework",
    }
)
EXTRAS_GENERIC = frozenset({"example", "definition", "exercise", "demo", "quiz", "quizzes", "lab"})

_STOP = frozenset(
    {"of", "and", "the", "a", "an", "for", "to", "in", "on", "with", "by", "as", "at", "or", "from"}
)
_DETERMINERS = frozenset({"a", "an", "the"})
# Number words folded when comparing labels. Ordinals are deliberately absent ("first" stays).
_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
# Words that stand for a digit inside an abbreviation (1NF = First Normal Form, 3DES = Triple DES).
_INITIAL_DIGITS = {
    **_NUMBER_WORDS,
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "double": "2",
    "triple": "3",
}
_IRREGULAR_SINGULAR = {
    "automata": "automaton",
    "matrices": "matrix",
    "indices": "index",
    "vertices": "vertex",
    "axes": "axis",
    "analyses": "analysis",
    "criteria": "criterion",
    "hypotheses": "hypothesis",
    "data": "datum",
    "policies": "policy",
}

_TOKEN = re.compile(r"[^\W_]+[+#]*")
_RAW_WORD = re.compile(r"[^\W_]+")
_COMPOUND = re.compile(r"[^\W_]+(?:-[^\W_]+)*")
_ISATION = re.compile(r"isation$")
_POSSESSIVE = re.compile(r"'s\b")
_ABBR_SHAPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/\-]*")
_ABBR_MAX_CHARS = 6
_EXPANSION_MIN_WORDS = 2
_EXPANSION_MAX_PARTS = 8
_PAREN = re.compile(r"\(([^()]{1,120})\)")
_PRECEDING_RUN = re.compile(r"((?:[^\W_]+[\s\-]+)*[^\W_]+)\s*$")
_WORD_RUN = re.compile(r"[^\W_]+(?:[\s\-]+[^\W_]+)*")
_ABBR_BEFORE_PAREN = re.compile(r"([^\W_]+(?:[-/][^\W_]+)*)\s*$")
_EVIDENCE_MAX_WORDS = 8
_EVIDENCE_LOOKBACK = 160


def _strip_possessive(text: str) -> str:
    return _POSSESSIVE.sub("", text.replace("\u2019", "'")).replace("'", "")


def _tokens(label: str) -> tuple[str, ...]:
    """Comparison tokens: casefolded words, no leading determiner, number words as digits."""
    found = _TOKEN.findall(_strip_possessive(label.casefold()))
    while len(found) > 1 and found[0] in _DETERMINERS:
        found = found[1:]
    folded = (_ISATION.sub("ization", w) for w in found)
    return tuple(_NUMBER_WORDS.get(w, w) for w in folded)


def _singular_forms(word: str) -> list[str]:
    """Candidate singulars of ``word`` in a fixed order (not a lemmatiser, only candidates)."""
    out: list[str] = []
    if word in _IRREGULAR_SINGULAR:
        out.append(_IRREGULAR_SINGULAR[word])
    if word.endswith("ies") and len(word) > 4:
        out.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        out.append(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        out.append(word[:-1])
    return out


def is_generic_label(label: str, extras: bool = False) -> bool:
    """Is the whole label a generic slide phrase (plural and determiner folded)?"""
    words = _tokens(label.replace("&", " "))
    if not words:
        return False
    tier = SAFE_GENERIC | EXTRAS_GENERIC if extras else SAFE_GENERIC
    last = words[-1]
    forms = [last, *_singular_forms(last)]
    return any(" ".join((*words[:-1], f)) in tier for f in forms)


def is_abbreviation(label: str) -> bool:
    """Short alphanumeric label that is mostly upper case (``TDE``, ``ReLU``, ``k-NN``)."""
    text = label.strip()
    if not _ABBR_SHAPE.fullmatch(text):
        return False
    alnum = [ch for ch in text if ch.isalnum()]
    upper = sum(ch.isupper() for ch in alnum)
    return 2 <= len(alnum) <= _ABBR_MAX_CHARS and upper >= 2 and 2 * upper >= len(alnum)


def _compact(label: str) -> str:
    return "".join(ch for ch in label.casefold() if ch.isalnum())


def _compound_words(label: str) -> list[str]:
    return _COMPOUND.findall(_strip_possessive(label))


def _part_count(label: str) -> int:
    return sum(len(w.split("-")) for w in _compound_words(label))


def _has_digit(label: str) -> bool:
    return any(ch.isdigit() for ch in label)


def _initial_variants(label: str, skip_stop: bool) -> frozenset[str]:
    """Strings the label's initials can spell (at most ``_ABBR_MAX_CHARS`` long).

    Every word contributes its first letter. An all-caps word contributes itself, a number word
    contributes its digit, and a hyphen compound contributes the initials of all parts or of its
    first part. With ``skip_stop`` stop words (``of``, ``and`` ...) contribute nothing.
    """
    words = _compound_words(label)
    if skip_stop:
        words = [w for w in words if w.casefold() not in _STOP]
    variants = {""}
    for word in words:
        parts = word.split("-")

        def one(part: str) -> set[str]:
            low = part.casefold()
            spelled = {low[0]}
            if len(part) >= 2 and part.isupper():
                spelled.add(low)
            if low in _INITIAL_DIGITS:
                spelled.add(_INITIAL_DIGITS[low])
            return spelled

        options = {"".join(c) for c in itertools.product(*(sorted(one(p)) for p in parts))}
        options |= one(parts[0])
        variants = {v + o for v in variants for o in options if len(v + o) <= _ABBR_MAX_CHARS}
    return frozenset(variants)


@dataclass
class _Cluster:
    """One final cluster: its normalisation groups and the members acting as expansions."""

    groups: list[list[int]]
    expansions: set[int] = field(default_factory=set)

    @property
    def members(self) -> list[int]:
        return sorted(m for g in self.groups for m in g)


class _UnionFind:
    """Union-find whose root is always the smallest member index."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra < rb:
            self.parent[rb] = ra
        elif rb < ra:
            self.parent[ra] = rb


def _slide_strings(slides: tuple[SlideRecord, ...]) -> list[str]:
    out: list[str] = []
    for slide in slides:
        for raw in (slide.title or "", slide.text):
            flat = " ".join(raw.split())
            if flat:
                out.append(flat)
    return out


def _evidence(
    labels: list[str], abbrs: list[int], expansions: list[int], slides: tuple[SlideRecord, ...]
) -> dict[int, set[int]]:
    """Abbreviation index -> expansion indexes paired on a slide.

    The two patterns are ``Expansion (ABBR)`` and ``ABBR (Expansion)``.
    """
    by_abbr: dict[str, list[int]] = defaultdict(list)
    for i in abbrs:
        by_abbr[labels[i].strip().casefold()].append(i)
    by_words: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for j in expansions:
        by_words[tuple(_RAW_WORD.findall(labels[j].casefold()))].append(j)
    found: dict[int, set[int]] = defaultdict(set)
    if not by_abbr or not by_words:
        return found
    for text in _slide_strings(slides):
        for match in _PAREN.finditer(text):
            inner = match.group(1).strip()
            forward = by_abbr.get(inner.casefold())
            if forward:
                before = text[max(0, match.start() - _EVIDENCE_LOOKBACK) : match.start()]
                run = _PRECEDING_RUN.search(before)
                words = _RAW_WORD.findall(run.group(1).casefold()) if run else []
                for size in range(_EXPANSION_MIN_WORDS, min(_EVIDENCE_MAX_WORDS, len(words)) + 1):
                    for j in by_words.get(tuple(words[-size:]), ()):
                        for i in forward:
                            found[i].add(j)
            head = _ABBR_BEFORE_PAREN.search(text[max(0, match.start() - 40) : match.start()])
            backward = by_abbr.get(head.group(1).casefold()) if head else None
            if backward and _WORD_RUN.fullmatch(inner):
                for j in by_words.get(tuple(_RAW_WORD.findall(inner.casefold())), ()):
                    for i in backward:
                        found[i].add(j)
    return found


def _prerequisite_pairs(ctx: FilterContext) -> frozenset[frozenset[str]]:
    return frozenset(
        frozenset((casefold_id(e.source), casefold_id(e.target)))
        for e in ctx.graph.graph.edges
        if e.relation_type == "prerequisite"
    )


class HeuristicConceptFilter:
    """Generic-phrase drop plus alias merge; see the module docstring for the rules."""

    name = "heuristic"
    is_mock = False

    def __init__(self, extras: bool = False) -> None:
        self.extras = extras

    def resolve(self, concepts: list[ConceptRecord], ctx: FilterContext) -> list[ConceptRecord]:
        records = list(concepts)
        labels = [c.label for c in records]
        dropped = {i for i, label in enumerate(labels) if is_generic_label(label, self.extras)}
        live = [i for i in range(len(records)) if i not in dropped]

        groups = self._normalisation_groups(labels, live)
        edges = self._abbreviation_edges(records, live, groups, ctx)
        components = self._components(groups, edges)

        anchor = {cluster.members[0]: cluster for cluster in components}
        out: list[ConceptRecord] = []
        for i, record in enumerate(records):
            if i in dropped:
                note = {"filter_status": "dropped", "filter_note": GENERIC_NOTE}
                out.append(record.model_copy(update=note))
            elif i in anchor:
                cluster = anchor[i]
                members = cluster.members
                out.append(record if len(members) == 1 else self._merge(records, members, cluster))
        return out

    # ------------------------------------------------------------------ stage 1
    @staticmethod
    def _normalisation_groups(labels: list[str], live: list[int]) -> list[list[int]]:
        """Clusters of live indexes (ascending) joined by the normalisation rules."""
        uf = _UnionFind(len(labels))
        tokens = {i: _tokens(labels[i]) for i in live}
        shaped = {i: is_abbreviation(labels[i]) for i in live}
        first_by_join: dict[str, int] = {}
        by_tokens: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i in live:
            if not tokens[i]:
                continue
            uf.union(first_by_join.setdefault("".join(tokens[i]), i), i)
            by_tokens[tokens[i]].append(i)
        for i in live:
            if not tokens[i] or shaped[i]:
                continue
            for form in _singular_forms(tokens[i][-1]):
                for j in by_tokens.get((*tokens[i][:-1], form), ()):
                    if not shaped[j]:
                        uf.union(i, j)
        grouped: dict[int, list[int]] = defaultdict(list)
        for i in live:
            grouped[uf.find(i)].append(i)
        out: list[list[int]] = []
        for members in grouped.values():
            if len(members) > MAX_CLUSTER:
                logger.debug("normalisation cluster of %d labels left unmerged", len(members))
                out.extend([m] for m in members)
            else:
                out.append(members)
        return sorted(out)

    # ------------------------------------------------------------------ stage 2
    @staticmethod
    def candidate_pairs(labels: list[str]) -> int:
        """How many (abbreviation, expansion) pairs the initials prefilter lets through."""
        live = list(range(len(labels)))
        abbrs = [i for i in live if is_abbreviation(labels[i])]
        buckets = _expansion_buckets(labels, live)
        return sum(len(buckets.get(_compact(labels[i]), ())) for i in abbrs)

    @staticmethod
    def _abbreviation_edges(
        records: list[ConceptRecord],
        live: list[int],
        groups: list[list[int]],
        ctx: FilterContext,
    ) -> list[tuple[int, int, int]]:
        """Accepted (abbreviation group, expansion group) merges as group positions."""
        labels = [c.label for c in records]
        group_of = {m: g for g, members in enumerate(groups) for m in members}
        abbrs = [i for i in live if is_abbreviation(labels[i])]
        buckets = _expansion_buckets(labels, live)
        candidates: dict[int, set[int]] = {}
        claimed: dict[int, set[int]] = defaultdict(set)
        for i in abbrs:
            abbr = _compact(labels[i])
            for j in buckets.get(abbr, ()):
                if group_of[j] == group_of[i] or (_has_digit(labels[j]) and not _has_digit(abbr)):
                    continue
                candidates.setdefault(i, set()).add(group_of[j])
                claimed[group_of[j]].add(group_of[i])
        expansion_idx = sorted({j for js in buckets.values() for j in js})
        evidence = _evidence(labels, abbrs, expansion_idx, ctx.slides)
        prerequisite = _prerequisite_pairs(ctx)

        def linked(ga: int, gb: int) -> bool:
            return any(
                frozenset((casefold_id(records[a].id), casefold_id(records[b].id))) in prerequisite
                for a in groups[ga]
                for b in groups[gb]
            )

        edges: list[tuple[int, int, int]] = []
        for i in abbrs:
            by_slide = {group_of[j] for j in evidence.get(i, ())} - {group_of[i]}
            if len(by_slide) == 1:
                target = next(iter(by_slide))
            else:
                options = candidates.get(i, set())
                if len(options) != 1:
                    continue
                target = next(iter(options))
                if len(claimed[target]) != 1:
                    continue
            if not linked(group_of[i], target):
                edges.append((group_of[i], target, i))
        return edges

    @staticmethod
    def _components(groups: list[list[int]], edges: list[tuple[int, int, int]]) -> list[_Cluster]:
        """Final clusters: groups joined by the abbreviation edges, minus oversized ones."""
        uf = _UnionFind(len(groups))
        for ga, gb, _ in edges:
            uf.union(ga, gb)
        joined: dict[int, list[int]] = defaultdict(list)
        for g in range(len(groups)):
            joined[uf.find(g)].append(g)
        expansion_groups: dict[int, set[int]] = defaultdict(set)
        for ga, gb, _ in edges:
            expansion_groups[uf.find(ga)].add(gb)
        out: list[_Cluster] = []
        for root in sorted(joined):
            gs = joined[root]
            size = sum(len(groups[g]) for g in gs)
            if size > MAX_CLUSTER:
                logger.debug("abbreviation cluster of %d labels left unmerged", size)
                out.extend(_Cluster([groups[g]]) for g in gs)
            else:
                exp = {m for g in expansion_groups[root] for m in groups[g]}
                out.append(_Cluster([groups[g] for g in gs], exp))
        return out

    # ------------------------------------------------------------------ canonical record
    @staticmethod
    def _merge(records: list[ConceptRecord], members: list[int], comp: _Cluster) -> ConceptRecord:
        pool = sorted(comp.expansions) or members

        def rank(i: int) -> tuple[int, int, int, str]:
            r = records[i]
            return (-r.frequency, -len(r.slide_indexes), -len(r.label), r.id)

        canonical = records[min(pool, key=rank)]
        absorbed = [records[m] for m in members if records[m].id != canonical.id]
        kinds = []
        if any(len(g) > 1 for g in comp.groups):
            kinds.append("normalisation")
        if comp.expansions:
            kinds.append("abbreviation")
        n = len(absorbed)
        note = f"merged {n} alias{'es' if n > 1 else ''} ({' + '.join(kinds)})"
        everyone = [records[m] for m in members]
        return canonical.model_copy(
            update={
                "merged_from": sorted(r.id for r in absorbed),
                "alias_labels": sorted(
                    (r.label for r in absorbed), key=lambda s: (s.casefold(), s)
                ),
                "frequency": max(r.frequency for r in everyone),
                "slide_indexes": sorted({s for r in everyone for s in r.slide_indexes}),
                "alignment_status": (
                    AlignmentStatus.ALIGNED
                    if any(r.alignment_status == AlignmentStatus.ALIGNED for r in everyone)
                    else canonical.alignment_status
                ),
                "filter_status": "kept",
                "filter_note": note,
            }
        )


def _expansion_buckets(labels: list[str], live: list[int]) -> dict[str, list[int]]:
    """Initials string -> indexes of multi-word, non-abbreviation labels that can spell it."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for j in live:
        label = labels[j]
        if is_abbreviation(label) or len(_tokens(label)) < _EXPANSION_MIN_WORDS:
            continue
        if _part_count(label) > _EXPANSION_MAX_PARTS:
            continue
        for variant in _initial_variants(label, False) | _initial_variants(label, True):
            if variant:
                buckets[variant].append(j)
    return buckets
