"""Real-deck ingestion: draft curated graphs, extract label-free features, summarise (counts only).

For each deck ``<stem>.pptx`` or ``.pdf``:

* ``<graphs>/<stem>.draft.json`` is drafted from the slide titles by the C2 tool
  ``make_curated_graph.py`` (``build_draft``) when missing;
* ``<graphs>/<stem>.graph.json`` is the reviewed copy used for extraction. When absent it is
  created from the draft with ``"curated": true`` and ``"review_status": "auto-draft,
  unreviewed"``; an existing file is never touched (so the team can edit it freely), except with
  ``refresh_auto`` for files that still carry that status;
* features are extracted through the frozen pipeline and the JSONL cache (no network, no semantic
  scorer: the semantic features are unavailable), with the default concept filter.

Reports hold counts and numeric ranges only, never slide text. Errors are reported by type only.
"""

import importlib.util
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import edusync_c2
from edusync_c2.difficulty.contract_check import check_c1_graph
from edusync_c2.difficulty.materials import MaterialsError
from edusync_c2.difficulty.registry import FEATURE_SPECS
from edusync_c2_model.examples import ConceptFeatures
from edusync_c2_model.extract import (
    DEFAULT_CONCEPT_FILTER,
    ExtractError,
    extract_all,
)
from edusync_c2_model.ids import is_valid_id
from edusync_c2_model.manifest import ManifestEntry, read_manifest, write_manifest

DECK_SUFFIXES = (".pptx", ".pdf")
REVIEW_STATUS = "auto-draft, unreviewed"
STATUSES = ("computed", "mocked", "unavailable")
_SCRIPT = "make_curated_graph.py"


class IngestError(ValueError):
    """A deck or its graph could not be prepared."""


def load_draft_module() -> ModuleType:
    """The C2 draft tool, loaded from its script file (it is not an importable package)."""
    path = Path(edusync_c2.__file__).resolve().parents[2] / "scripts" / _SCRIPT
    spec = importlib.util.spec_from_file_location("c2_make_curated_graph", path)
    if spec is None or spec.loader is None or not path.is_file():
        raise IngestError(f"{_SCRIPT} not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_auto_draft(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("review_status") == REVIEW_STATUS


def ensure_graphs(
    deck: Path, graphs_dir: Path, *, refresh_auto: bool = False
) -> tuple[Path, bool, bool]:
    """Make sure the draft and the reviewed copy exist.

    Returns ``(reviewed_path, draft_created, reviewed_created)``. Existing files are never
    overwritten, except with ``refresh_auto``: then the draft is regenerated and the reviewed
    copy rewritten, but only when the reviewed copy still has ``review_status`` equal to
    ``REVIEW_STATUS`` (a copy a human changed is left alone). Raises ``MaterialsError`` for an
    unreadable deck and ``IngestError`` when the draft breaks the C1 contract.
    """
    graphs_dir.mkdir(parents=True, exist_ok=True)
    draft_path = graphs_dir / f"{deck.stem}.draft.json"
    reviewed_path = graphs_dir / f"{deck.stem}.graph.json"
    draft_created = reviewed_created = False
    if refresh_auto and reviewed_path.exists() and _is_auto_draft(reviewed_path):
        draft_path.unlink(missing_ok=True)
        reviewed_path.unlink()
    if not draft_path.exists():
        draft = load_draft_module().build_draft(deck.read_bytes(), deck.name)
        report = check_c1_graph(draft, "v1")
        if not report.ok:
            raise IngestError("draft graph breaks the C1 contract")
        draft_path.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", "utf-8")
        draft_created = True
    if not reviewed_path.exists():
        reviewed = json.loads(draft_path.read_text(encoding="utf-8-sig"))
        reviewed["curated"] = True
        reviewed["review_status"] = REVIEW_STATUS
        with open(reviewed_path, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(reviewed, indent=2, ensure_ascii=False) + "\n")
        reviewed_created = True
    return reviewed_path, draft_created, reviewed_created


@dataclass
class DeckResult:
    lecture_id: str
    slides: int = 0
    nodes: int = 0
    edges: int = 0
    rows: list[ConceptFeatures] = field(default_factory=list)
    cache_hit: bool = False
    draft_created: bool = False
    reviewed_created: bool = False
    refreshed: bool = False
    refresh_skipped: bool = False  # --refresh-auto left a human-edited graph alone
    seconds: float = 0.0
    error: str | None = None  # exception type name only


def _graph_counts(path: Path) -> tuple[int, int, int]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    graph = payload.get("graph", {})
    return (
        int(payload.get("page_count") or 0),
        len(graph.get("nodes", [])),
        len(graph.get("edges", [])),
    )


def list_decks(decks_dir: Path) -> list[Path]:
    return sorted(p for p in Path(decks_dir).iterdir() if p.suffix.lower() in DECK_SUFFIXES)


def ingest_decks(
    decks_dir: Path,
    graphs_dir: Path,
    cache_dir: Path,
    manifest_path: Path,
    *,
    course: str,
    concept_filter: str = DEFAULT_CONCEPT_FILTER,
    refresh_auto: bool = False,
) -> list[DeckResult]:
    """Prepare graphs, extract features per deck and write the manifest of the decks that worked.

    ``manifest_path`` must sit in a folder that contains the decks and graphs (paths are written
    relative to it).
    """
    results: list[DeckResult] = []
    entries: list[ManifestEntry] = []
    for deck in list_decks(decks_dir):
        result = DeckResult(lecture_id=deck.stem)
        results.append(result)
        started = time.perf_counter()
        try:
            if not is_valid_id(deck.stem):
                raise IngestError("deck file name is not a valid lecture id")
            existing = graphs_dir / f"{deck.stem}.graph.json"
            if refresh_auto and existing.exists():
                result.refreshed = _is_auto_draft(existing)
                result.refresh_skipped = not result.refreshed
            graph, result.draft_created, result.reviewed_created = ensure_graphs(
                deck, graphs_dir, refresh_auto=refresh_auto
            )
            result.slides, result.nodes, result.edges = _graph_counts(graph)
            entry = ManifestEntry(deck.stem, course, deck, graph)
            rows, stats = extract_all([entry], cache_dir, concept_filter=concept_filter)
            result.rows = rows
            result.cache_hit = stats.hits == 1
            entries.append(entry)
        except (IngestError, ExtractError, MaterialsError, OSError, ValueError) as exc:
            result.error = type(exc).__name__
        result.seconds = time.perf_counter() - started
    if entries:
        write_manifest(manifest_path, entries)
    return results


def load_reference_features(
    cache_dir: Path, manifest: Path | Sequence[ManifestEntry]
) -> list[ConceptFeatures]:
    """Label-free features of every lecture in the manifest, sorted by (lecture, concept).

    Uses the extraction cache (recomputing, and caching, any lecture that is not cached).
    """
    entries = read_manifest(Path(manifest)) if isinstance(manifest, str | Path) else list(manifest)
    rows, _ = extract_all(entries, Path(cache_dir))
    return sorted(rows, key=lambda r: r.key)


# ---------------------------------------------------------------------------------- summary


@dataclass(frozen=True)
class FeatureStat:
    name: str
    group: str
    counts: dict[str, int]
    low: float | None
    median: float | None
    high: float | None

    @property
    def flag(self) -> str:
        if self.counts["computed"] == 0 and self.counts["mocked"] == 0:
            return "all unavailable"
        if self.counts["computed"] == 0:
            return "never computed"
        return "constant" if self.low == self.high else ""


def feature_stats(rows: Sequence[ConceptFeatures]) -> list[FeatureStat]:
    """Per feature: status counts and min/median/max over the computed values."""
    stats = []
    for i, spec in enumerate(FEATURE_SPECS):
        counts = dict.fromkeys(STATUSES, 0)
        computed: list[float] = []
        for row in rows:
            counts[row.statuses[i]] += 1
            if row.statuses[i] == "computed":
                computed.append(row.values[i])
        stats.append(
            FeatureStat(
                name=spec.name,
                group=spec.group.value,
                counts=counts,
                low=min(computed) if computed else None,
                median=statistics.median(computed) if computed else None,
                high=max(computed) if computed else None,
            )
        )
    return stats


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def render_summary(results: Sequence[DeckResult], *, course: str) -> str:
    """Markdown with counts and ranges only (no slide text)."""
    ok = [r for r in results if r.error is None]
    rows = [row for r in ok for row in r.rows]
    lines = [
        "# Real-deck ingestion summary",
        "",
        f"Course: {course}. Decks: {len(results)} ({len(ok)} ingested, "
        f"{len(results) - len(ok)} failed). Concepts: {len(rows)}.",
        "Graphs are auto-drafted from slide titles and marked curated but UNREVIEWED; "
        "the semantic scorer is off, so semantic features are unavailable.",
        "",
        "| Deck | Slides | Concepts | Graph nodes | Graph edges | Cache | Seconds | Error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.lecture_id} | {r.slides} | {len(r.rows)} | {r.nodes} | {r.edges} | "
            f"{'hit' if r.cache_hit else 'miss'} | {r.seconds:.2f} | {r.error or ''} |"
        )
    lines += [
        "",
        "## Features across all concepts",
        "",
        "| Feature | Group | Computed | Mocked | Unavailable | Min | Median | Max | Flag |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in feature_stats(rows):
        c = s.counts
        lines.append(
            f"| {s.name} | {s.group} | {c['computed']} | {c['mocked']} | {c['unavailable']} | "
            f"{_num(s.low)} | {_num(s.median)} | {_num(s.high)} | {s.flag} |"
        )
    return "\n".join(lines) + "\n"
