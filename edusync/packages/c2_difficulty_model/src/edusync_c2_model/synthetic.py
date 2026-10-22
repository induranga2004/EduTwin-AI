"""SYNTHETIC training data, one separate dataset per model (A, B, C, D and the combiner).

SYNTHETIC DATA: no real course material is used. The labels are the transparent
``synthetic_label`` rule of ``edusync_c2.difficulty.synthetic`` plus seeded label noise and two
simulated raters (an adjudicator row settles their disagreements), so anything measured on these
datasets tests the plumbing only and is NOT accuracy. Every label row is ``synthetic=true``.

Each dataset has its own seed range and its own mix of deck shapes, so the five datasets differ:

====  ============  =====================================================================
 key   model         what is emphasised
====  ============  =====================================================================
 A     language      wordier decks with more formulas, code and deeper bullet levels
 B     prerequisites v1 graphs with prerequisite chains (every concept needs the previous one)
 C     explanation   definition, example and step sentences (no semantic scorer is needed)
 D     density       layout-varied PPTX (box sizes, fonts, pictures, shapes, tables)
 M     combiner      a balanced mix of the four shapes (graph only on B-shaped decks)
====  ============  =====================================================================

Only the parameters of ``edusync_c2.difficulty.synthetic`` are used (``ConceptSpec`` fields,
``vary_layout``, the graph version); that module is not edited. Generation is deterministic for a
given ``base_seed`` (PPTX bytes embed timestamps, so existing files are reused, not rewritten).
"""

import csv
import io
import json
import random
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from edusync_c2.difficulty.synthetic import (
    ConceptSpec,
    DeckSpec,
    build_c1_graph,
    build_pptx,
    make_deck_spec,
    synthetic_label,
)
from edusync_c2_model.examples import Example
from edusync_c2_model.extract import ExtractStats, JoinResult, extract_all, join_examples
from edusync_c2_model.labels import LABEL_ID, LABEL_ORDER, read_labels
from edusync_c2_model.manifest import ManifestEntry, ManifestError, read_manifest, write_manifest

DATASET_VERSION = 1
DEFAULT_ROWS = 5000
DEFAULT_BASE_SEED = 20260
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
N_COURSES = 5
RATERS = ("R1", "R2")
ADJUDICATOR = "ADJ"
LABEL_COLUMNS = (  # docs/edusync/c2-label-collection.md section 9
    "lecture_id",
    "concept_id",
    "concept_slug",
    "concept_label",
    "slide_indexes",
    "difficulty_label",
    "rater_id",
    "student_level",
    "confidence",
    "notes",
    "labelled_at",
    "rubric_version",
    "synthetic",
)


class SyntheticError(RuntimeError):
    """A dataset cannot be generated or reused."""


# ------------------------------------------------------------------------------------ shapes


def _shape_language(c: ConceptSpec, i: int, labels: list[str], rng: random.Random) -> ConceptSpec:
    return c.model_copy(
        update={
            "words": rng.randint(20, 110),
            "formula": rng.random() < 0.3,
            "code": rng.random() < 0.15,
            "bullet_depth": rng.randint(0, 3),
            "definition": rng.random() < 0.5,
            "examples": rng.randint(0, 2),
            "steps": rng.choice([0, 0, 2]),
        }
    )


def _shape_prerequisites(
    c: ConceptSpec, i: int, labels: list[str], rng: random.Random
) -> ConceptSpec:
    prerequisites = [labels[i - 1]] if i else []
    extra = min(max(i - 1, 0), rng.choice([0, 1, 2]))
    prerequisites += rng.sample(labels[: max(i - 1, 0)], extra)
    return c.model_copy(
        update={
            "prerequisites": prerequisites,
            "words": rng.randint(25, 95),
            "formula": rng.random() < 0.2,
            "code": rng.random() < 0.15,
            "definition": rng.random() < 0.55,
        }
    )


def _shape_explanation(
    c: ConceptSpec, i: int, labels: list[str], rng: random.Random
) -> ConceptSpec:
    return c.model_copy(
        update={
            "definition": rng.random() < 0.7,
            "examples": rng.randint(0, 2),
            "steps": rng.choice([0, 2, 3]),
            "words": rng.randint(50, 150),
            "formula": rng.random() < 0.15,
            "code": rng.random() < 0.1,
        }
    )


def _shape_density(c: ConceptSpec, i: int, labels: list[str], rng: random.Random) -> ConceptSpec:
    return c.model_copy(
        update={
            "picture": rng.random() < 0.6,
            "formula": rng.random() < 0.25,
            "code": rng.random() < 0.25,
            "bullet_depth": rng.randint(0, 3),
        }
    )


_SHAPES: dict[str, Callable[[ConceptSpec, int, list[str], random.Random], ConceptSpec]] = {
    "language": _shape_language,
    "prerequisites": _shape_prerequisites,
    "explanation": _shape_explanation,
    "density": _shape_density,
}
_GRAPH_SHAPES = frozenset({"prerequisites"})  # shapes whose decks ship a v1 C1 graph
_LAYOUT_SHAPES = frozenset({"density"})  # shapes built with vary_layout=True


@dataclass(frozen=True)
class ExpertProfile:
    letter: str
    name: str
    seed_offset: int  # disjoint seed range per dataset
    shapes: tuple[str, ...]  # deck shapes, cycled by deck number
    n_concepts: tuple[int, ...]
    label_noise: float  # chance the first rater flips to an adjacent class
    rater_flip: float  # chance the second rater flips to an adjacent class


EXPERTS: dict[str, ExpertProfile] = {
    p.letter: p
    for p in (
        ExpertProfile("A", "language", 1_000_000, ("language",), (8,), 0.10, 0.08),
        ExpertProfile("B", "prerequisites", 2_000_000, ("prerequisites",), (8, 10, 12), 0.12, 0.08),
        ExpertProfile("C", "explanation", 3_000_000, ("explanation",), (8,), 0.08, 0.06),
        ExpertProfile("D", "density", 4_000_000, ("density",), (8,), 0.10, 0.08),
        ExpertProfile(
            "M",
            "combiner",
            5_000_000,
            ("language", "prerequisites", "explanation", "density"),
            (6, 8, 10),
            0.10,
            0.08,
        ),
    )
}


@dataclass(frozen=True)
class DeckPlan:
    spec: DeckSpec
    shape: str
    course_id: str
    graph: bool
    layout: bool
    label_seed: str
    label_noise: float
    rater_flip: float


def plan_dataset(profile: ExpertProfile, rows: int, base_seed: int) -> list[DeckPlan]:
    """Deck specs for ``rows`` labelled concepts (the last deck is shortened to fit exactly)."""
    if rows < 1:
        raise SyntheticError("rows must be at least 1")
    plans: list[DeckPlan] = []
    remaining = rows
    number = 0
    while remaining > 0:
        number += 1
        seed = base_seed + profile.seed_offset + number
        rng = random.Random(f"{seed}:shape")
        n = min(rng.choice(profile.n_concepts), remaining)
        remaining -= n
        shape = profile.shapes[(number - 1) % len(profile.shapes)]
        base = make_deck_spec(seed, n_concepts=n, lecture_id=f"S{profile.letter}-{number}")
        labels = [c.label for c in base.concepts]
        concepts = [_SHAPES[shape](c, i, labels, rng) for i, c in enumerate(base.concepts)]
        plans.append(
            DeckPlan(
                spec=base.model_copy(update={"concepts": concepts}),
                shape=shape,
                course_id=f"S{profile.letter}-C{(number - 1) % N_COURSES + 1}",
                graph=shape in _GRAPH_SHAPES,
                layout=shape in _LAYOUT_SHAPES,
                label_seed=f"{seed}:labels",
                label_noise=profile.label_noise,
                rater_flip=profile.rater_flip,
            )
        )
    return plans


# ------------------------------------------------------------------------------------ labels


def _adjacent(label_id: int, rng: random.Random) -> int:
    if label_id == 0:
        return 1
    if label_id == len(LABEL_ORDER) - 1:
        return label_id - 1
    return rng.choice((label_id - 1, label_id + 1))


def label_rows(plan: DeckPlan) -> list[list[str]]:
    """CSV rows (section 9 columns): two simulated raters plus ADJ rows where they disagree."""
    rng = random.Random(plan.label_seed)
    use_slug = plan.graph
    rows = []
    for position, c in enumerate(plan.spec.concepts):
        truth = LABEL_ID[synthetic_label(c, len(c.prerequisites)).value]
        first = _adjacent(truth, rng) if rng.random() < plan.label_noise else truth
        second = _adjacent(first, rng) if rng.random() < plan.rater_flip else first
        given = [(RATERS[0], first), (RATERS[1], second)]
        if first != second:
            given.append((ADJUDICATOR, first))
        for rater, label_id in given:
            rows.append(
                [
                    plan.spec.lecture_id,
                    c.slug if use_slug else c.concept_id,
                    c.slug,
                    c.label,
                    str(position + 2),
                    LABEL_ORDER[label_id],
                    rater,
                    "mixed",
                    "3",
                    "",
                    "2026-01-01",
                    "r1",
                    "true",
                ]
            )
    return rows


def labels_text(plans: Sequence[DeckPlan]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(LABEL_COLUMNS)
    for plan in plans:
        writer.writerows(label_rows(plan))
    return buf.getvalue()


# ------------------------------------------------------------------------------------- files


def _build_deck(plan: DeckPlan) -> tuple[bytes, bytes | None]:
    pptx = build_pptx(plan.spec, vary_layout=plan.layout)
    graph = None
    if plan.graph:
        graph = json.dumps(build_c1_graph(plan.spec, "v1"), sort_keys=True).encode("utf-8")
    return pptx, graph


@dataclass(frozen=True)
class DatasetPaths:
    letter: str
    name: str
    root: Path
    manifest: Path
    labels: Path
    rows: int
    decks: int
    reused: bool


def _write_dataset(
    profile: ExpertProfile, root: Path, rows: int, base_seed: int, jobs: int
) -> DatasetPaths:
    plans = plan_dataset(profile, rows, base_seed)
    (root / "lectures").mkdir(parents=True)
    (root / "graphs").mkdir()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            built = list(pool.map(_build_deck, plans, chunksize=8))
    else:
        built = [_build_deck(p) for p in plans]
    entries = []
    for plan, (pptx, graph) in zip(plans, built, strict=True):
        lecture = plan.spec.lecture_id
        file = root / "lectures" / f"{lecture}.pptx"
        file.write_bytes(pptx)
        graph_path = None
        if graph is not None:
            graph_path = root / "graphs" / f"{lecture}.json"
            graph_path.write_bytes(graph)
        entries.append(ManifestEntry(lecture, plan.course_id, file, graph_path))
    manifest = root / "manifest.csv"
    write_manifest(manifest, entries)
    labels = root / "labels.csv"
    labels.write_text(labels_text(plans), encoding="utf-8", newline="\n")
    meta = _meta(profile, rows, base_seed)
    (root / "dataset.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    return DatasetPaths(
        profile.letter, profile.name, root, manifest, labels, rows, len(plans), reused=False
    )


def _meta(profile: ExpertProfile, rows: int, base_seed: int) -> dict:
    return {
        "version": DATASET_VERSION,
        "expert": profile.letter,
        "rows": rows,
        "base_seed": base_seed,
        "synthetic": True,
    }


def _try_reuse(
    profile: ExpertProfile, root: Path, rows: int, base_seed: int
) -> DatasetPaths | None:
    try:
        meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        entries = read_manifest(root / "manifest.csv")
        labels = root / "labels.csv"
        if meta != _meta(profile, rows, base_seed) or not labels.is_file():
            return None
    except (OSError, ValueError, ManifestError):
        return None
    return DatasetPaths(
        profile.letter, profile.name, root, root / "manifest.csv", labels, rows, len(entries), True
    )


def build_expert_datasets(
    out_dir: Path | None = None,
    rows_per_expert: int = DEFAULT_ROWS,
    base_seed: int = DEFAULT_BASE_SEED,
    *,
    jobs: int = 1,
    experts: Sequence[str] = tuple(EXPERTS),
    force: bool = False,
) -> dict[str, DatasetPaths]:
    """Write one synthetic dataset per expert under ``out_dir/<name>/`` and return their paths.

    An existing dataset with identical parameters is reused as is (so the feature cache stays
    valid); one with different parameters raises unless ``force`` (then it is replaced).
    """
    out_dir = Path(DEFAULT_DATA_DIR if out_dir is None else out_dir)
    unknown = [e for e in experts if e not in EXPERTS]
    if unknown:
        raise SyntheticError(f"unknown expert(s) {unknown}; valid: {list(EXPERTS)}")
    if jobs < 1:
        raise SyntheticError("jobs must be at least 1")
    result = {}
    for letter in experts:
        profile = EXPERTS[letter]
        root = out_dir / profile.name
        reused = _try_reuse(profile, root, rows_per_expert, base_seed) if root.exists() else None
        if reused is None and root.exists():
            if not force:
                raise SyntheticError(
                    f"{root} exists with different parameters; pass force=True to replace it"
                )
            shutil.rmtree(root)
        result[letter] = reused or _write_dataset(profile, root, rows_per_expert, base_seed, jobs)
    return result


# ---------------------------------------------------------------------------------- loading


def load_dataset(
    paths: DatasetPaths, cache_dir: Path
) -> tuple[list[Example], JoinResult, ExtractStats]:
    """Extract features (cached) and join them to the dataset's labels."""
    features, stats = extract_all(read_manifest(paths.manifest), cache_dir)
    labels = read_labels([paths.labels])
    joined = join_examples(features, labels)
    return joined.examples, joined, stats


def class_balance(examples: Sequence[Example]) -> dict[str, float]:
    """Share of each class (zeros when there are no examples)."""
    counts = Counter(e.label for e in examples)
    total = len(examples)
    return {name: (counts[name] / total if total else 0.0) for name in LABEL_ORDER}
