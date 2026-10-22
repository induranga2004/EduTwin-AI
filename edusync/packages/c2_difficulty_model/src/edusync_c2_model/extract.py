"""Feature extraction: run the frozen ``DifficultyPipeline`` and record exactly what a model sees.

A ``RecordingModel`` stands in for the difficulty model: it captures every ``FusionInput`` the
pipeline builds and delegates to ``RuleBaselineModel``, so training uses the same inputs, in
registry order, as inference. Results are cached per lecture as JSONL; the cache key covers the
lecture bytes, the graph bytes, the lecture/course ids, the feature schema version and hash, the
pipeline version and the concept filter, so any change misses the cache and recomputes.
"""

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from edusync_c2.difficulty import DifficultyPipeline, RuleBaselineModel
from edusync_c2.difficulty.concepts import create_concept_filter
from edusync_c2.difficulty.fusion import to_vector
from edusync_c2.difficulty.materials import MaterialsError
from edusync_c2.difficulty.providers import FileGraphProvider, is_curated
from edusync_c2.difficulty.registry import FEATURE_SCHEMA_VERSION, feature_schema_hash
from edusync_c2.difficulty.schemas import (
    C1GraphResponse,
    DifficultyResponse,
    FusionInput,
    ModelOutput,
)
from edusync_c2_model.examples import FeatureRow
from edusync_c2_model.manifest import ManifestEntry

DEFAULT_CONCEPT_FILTER = "mock_keep_all"
ROW_FORMAT = "2"  # bump when the cached row fields change
PIPELINE_VERSION: str = DifficultyResponse.model_fields["pipeline_version"].default


class ExtractError(ValueError):
    """A lecture or graph could not be turned into feature rows."""


class RecordingModel:
    """Captures the pipeline's ``FusionInput`` list and delegates to the rule baseline."""

    def __init__(self) -> None:
        self._inner = RuleBaselineModel()
        self.info = self._inner.info
        self.captured: list[FusionInput] = []

    def predict(self, inputs: list[FusionInput]) -> list[ModelOutput | None]:
        self.captured = list(inputs)
        return self._inner.predict(inputs)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_lecture(
    entry: ManifestEntry,
    *,
    data: bytes | None = None,
    graph_bytes: bytes | None = None,
    concept_filter: str = DEFAULT_CONCEPT_FILTER,
) -> list[FeatureRow]:
    """Run the pipeline on one lecture (no semantic scorer, no network) and return its rows."""
    data = entry.file.read_bytes() if data is None else data
    graph = None
    if entry.graph is not None:
        raw = entry.graph.read_bytes() if graph_bytes is None else graph_bytes
        try:
            graph = C1GraphResponse.model_validate(json.loads(raw.decode("utf-8-sig")))
        except (ValueError, ValidationError) as exc:
            raise ExtractError(
                f"{entry.lecture_id}: graph is not a C1 graph ({type(exc).__name__})"
            ) from exc
    recorder = RecordingModel()
    factory = None
    if graph is not None and entry.graph is not None and is_curated(graph):
        # a hand-made graph goes through the file provider, so the pipeline labels it "curated"
        graph_path = entry.graph
        factory = lambda _data, _name: FileGraphProvider(graph_path)  # noqa: E731
        graph = None
    pipeline = DifficultyPipeline(
        model=recorder,
        concept_filter=create_concept_filter(concept_filter),
        graph_provider_factory=factory,
    )
    try:
        response = pipeline.run(data, entry.file.name, graph)
    except MaterialsError as exc:
        raise ExtractError(f"{entry.lecture_id}: lecture file rejected: {exc.message}") from exc
    labels = {p.concept_id: p.label for p in response.predictions}
    slides = {p.concept_id: tuple(p.slide_indexes) for p in response.predictions}
    graph_kind = next((i.kind for i in response.inputs if i.name == "c1_graph"), "none")
    rows = []
    for fusion in recorder.captured:
        values, mask = to_vector(fusion)
        rows.append(
            FeatureRow(
                lecture_id=entry.lecture_id,
                course_id=entry.course_id,
                concept_id=fusion.concept_id,
                concept_label=labels[fusion.concept_id],
                text=fusion.text,
                values=tuple(values),
                missing=tuple(mask),
                statuses=tuple(f.status.value for f in fusion.features),
                slide_indexes=slides[fusion.concept_id],
                graph_kind=graph_kind,
            )
        )
    return rows


def cache_key(
    entry: ManifestEntry, data: bytes, graph_bytes: bytes | None, concept_filter: str
) -> dict[str, str]:
    return {
        "lecture_id": entry.lecture_id,
        "course_id": entry.course_id,
        "lecture_sha256": sha256_hex(data),
        "graph_sha256": "" if graph_bytes is None else sha256_hex(graph_bytes),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "pipeline_version": PIPELINE_VERSION,
        "concept_filter": concept_filter,
        "row_format": ROW_FORMAT,
    }


def _cache_path(cache_dir: Path, key: dict[str, str]) -> Path:
    digest = sha256_hex(json.dumps(key, sort_keys=True).encode("utf-8"))[:24]
    return cache_dir / f"{key['lecture_id']}-{digest}.jsonl"


def _read_cache(path: Path, key: dict[str, str]) -> list[FeatureRow] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        if not lines or json.loads(lines[0]) != key:
            return None
        return [FeatureRow.from_json(line) for line in lines[1:]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(path: Path, key: dict[str, str], rows: list[FeatureRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    body = [json.dumps(key, sort_keys=True), *(row.to_json() for row in rows)]
    tmp.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


@dataclass
class ExtractStats:
    hits: int = 0
    misses: int = 0


def extract_all(
    entries: Iterable[ManifestEntry],
    cache_dir: Path,
    *,
    concept_filter: str = DEFAULT_CONCEPT_FILTER,
) -> tuple[list[FeatureRow], ExtractStats]:
    """Rows of every lecture (manifest order), using and filling the cache."""
    stats = ExtractStats()
    rows: list[FeatureRow] = []
    for entry in entries:
        data = entry.file.read_bytes()
        graph_bytes = entry.graph.read_bytes() if entry.graph is not None else None
        key = cache_key(entry, data, graph_bytes, concept_filter)
        path = _cache_path(Path(cache_dir), key)
        cached = _read_cache(path, key)
        if cached is None:
            stats.misses += 1
            cached = extract_lecture(
                entry, data=data, graph_bytes=graph_bytes, concept_filter=concept_filter
            )
            _write_cache(path, key, cached)
        else:
            stats.hits += 1
        rows.extend(cached)
    return rows, stats
