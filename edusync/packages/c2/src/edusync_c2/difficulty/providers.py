"""Graph providers: where the C1 graph comes from (caller payload, mock, later the C1 API)."""

import hashlib
import itertools
import re
from importlib import resources
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from edusync_c2.difficulty.c1_client import Transport
from edusync_c2.difficulty.concepts import locate_slides, normalize
from edusync_c2.difficulty.errors import GraphProviderError
from edusync_c2.difficulty.ids import casefold_id
from edusync_c2.difficulty.schemas import (
    C1GraphResponse,
    GraphContext,
    InputSource,
    MaterialsDoc,
)

_EVIDENCE_MAX = 200
_DIRECTION = "source_is_prerequisite_of_target"
_SAMPLE_RESOURCE = ("resources", "sample_c1_graph_v1.json")


def is_curated(response: C1GraphResponse) -> bool:
    """Is the graph marked as hand-made: top-level ``"curated": true`` or ``"note": "curated"``."""
    extra = response.model_extra or {}
    return extra.get("curated") is True or extra.get("note") == "curated"


def curated_graph_path(directory: Path, deck_filename: str) -> Path:
    """``<directory>/<deck stem>.graph.json`` for a deck file name (any directory part ignored)."""
    return directory / f"{Path(deck_filename.replace(chr(92), '/')).stem}.graph.json"


MOCKED_GRAPH_PACKAGE_DIR = "mocked_knowledge_graph"
_LECTURE_NO = re.compile(r"lecture\s*0*(\d+)", re.IGNORECASE)
_COPY_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


def _graph_key(name: str) -> str:
    stem = Path(name.replace(chr(92), "/")).name
    stem = stem[: -len(".graph.json")] if stem.lower().endswith(".graph.json") else Path(stem).stem
    return re.sub(r"[^a-z0-9]+", "-", _COPY_SUFFIX.sub("", stem).casefold()).strip("-")


def find_mocked_graph(deck_filename: str, directory: Path | None = None) -> Path | None:
    """The hand-made mocked graph of a lecture file, matched by lecture name, then number.

    Looks in ``directory`` (if given) and then in the package folder ``mocked_knowledge_graph``.
    Matching ignores case, punctuation, a trailing ``(2)`` copy marker and the file extension; if
    the names differ it falls back to the ``Lecture NN`` number. Returns None when nothing fits.
    """
    folders = [directory] if directory is not None else []
    folders.append(Path(str(resources.files("edusync_c2").joinpath(MOCKED_GRAPH_PACKAGE_DIR))))
    key = _graph_key(deck_filename)
    number = _LECTURE_NO.search(deck_filename)
    for folder in folders:
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.graph.json"))
        for f in files:
            if _graph_key(f.name) == key:
                return f
        for f in files:
            if number and (m := _LECTURE_NO.search(f.name)) and m.group(1) == number.group(1):
                return f
        default = folder / "default.graph.json"
        if default.is_file() and directory is not None and folder == directory:
            return default
    return None


def slugify(label: str) -> str:
    """Alias of ``casefold_id``: the v0/mock/fallback id of a label (casefolded, spaces kept).

    The name is historical. It is not a hyphen slug; for the C1 v1 id use ``ids.slug_id``.
    """
    return casefold_id(label)


class GraphProvider(Protocol):
    name: str

    def build(self, materials: MaterialsDoc) -> GraphContext: ...


class PayloadGraphProvider:
    """Uses a C1 response supplied by the caller (v0 dummy or v1)."""

    name = "payload"

    def __init__(self, payload: C1GraphResponse) -> None:
        self._payload = payload

    def build(self, materials: MaterialsDoc) -> GraphContext:
        return GraphContext(
            response=self._payload,
            source=InputSource(
                name="c1_graph",
                kind="real",
                detail="C1 graph supplied by the caller (v0 dummy or v1); not verified by C2",
            ),
        )


def _usable_title(title: str | None) -> bool:
    t = (title or "").strip()
    return bool(t) and not t.endswith("?") and any(ch.isalnum() for ch in t)


def _sentence_with_both(text: str, first: str, second: str) -> str:
    """First sentence containing both labels; falls back to the start of the slide text."""
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        low = normalize(s)
        if normalize(first) in low and normalize(second) in low:
            return s.strip()[:_EVIDENCE_MAX]
    return " ".join(text.split())[:_EVIDENCE_MAX]


class MockC1GraphProvider:
    """Fabricates a C1-v1-shaped graph from the deck's slide titles (Phase 1 placeholder)."""

    name = "mock_from_slide_titles"

    def build(self, materials: MaterialsDoc) -> GraphContext:
        labels: dict[str, str] = {}
        for s in materials.slides:
            if _usable_title(s.title):
                labels.setdefault(slugify(s.title or ""), (s.title or "").strip())
        ids = list(labels)
        slides = {s.index: s for s in materials.slides}
        found = {i: locate_slides(labels[i], i, materials.slides) for i in ids}
        nodes = [
            {"id": i, "label": labels[i], "description": None, "frequency": max(1, len(found[i]))}
            for i in ids
        ]
        edges: list[dict] = []
        for a, ida in enumerate(ids):
            for idb in ids[a + 1 :]:
                pages = sorted(set(found[ida]) & set(found[idb]))
                if not pages:
                    continue
                evidence = [
                    {
                        "page": p,
                        "sentence": _sentence_with_both(slides[p].text, labels[ida], labels[idb]),
                    }
                    for p in pages
                ]
                edges.append(
                    {
                        "source": ida,
                        "target": idb,
                        "relation_type": "co_mentioned",
                        "evidence_count": len(evidence),
                        "evidence": evidence,
                    }
                )
        for prev, nxt in itertools.pairwise(ids):
            edges.append(
                {
                    "source": prev,
                    "target": nxt,
                    "relation_type": "prerequisite",
                    "direction": _DIRECTION,
                    "confidence": 0.5,
                    "source_type": "inferred",
                    "evidence_count": 0,
                    "evidence": [],
                }
            )
        response = C1GraphResponse.model_validate(
            {
                "lecture_id": None,
                "filename": materials.filename,
                "page_count": materials.slide_count,
                "concept_count": len(ids),
                "concepts": [{"name": n["label"], "frequency": n["frequency"]} for n in nodes],
                "graph": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "nodes": nodes,
                    "edges": edges,
                    "note": "Mock graph: prerequisite edges are placeholders taken from slide "
                    "order, not real prerequisites.",
                },
                "message": "Mock graph generated by C2 (Phase 1 placeholder, not from C1).",
            }
        )
        return GraphContext(
            response=response,
            source=InputSource(
                name="c1_graph",
                kind="mock",
                detail="mock graph: concepts = slide titles, prerequisites = slide order "
                "(placeholder, not real C1 data)",
            ),
        )


class C1ApiGraphProvider:
    """Asks C1 for the graph of the uploaded file through an injectable transport.

    Built per request because ``build`` receives the parsed deck, not the raw bytes. The reported
    ``lecture_id`` is derived from the bytes, so identical uploads give identical responses.
    """

    name = "c1_api"

    def __init__(self, data: bytes, filename: str, transport: Transport) -> None:
        self._data = data
        self._filename = filename
        self._transport = transport

    def build(self, materials: MaterialsDoc) -> GraphContext:
        raw = self._transport(self._data, self._filename)
        try:
            response = C1GraphResponse.model_validate(raw)
        except ValidationError:
            raise GraphProviderError(502, "C1 returned an invalid graph") from None
        response = response.model_copy(
            update={"lecture_id": f"c1-{hashlib.sha256(self._data).hexdigest()[:12]}"}
        )
        nodes, edges = len(response.graph.nodes), len(response.graph.edges)
        return GraphContext(
            response=response,
            source=InputSource(
                name="c1_graph",
                kind="real",
                detail=f"C1 API graph ({nodes} nodes, {edges} edges)",
            ),
        )


class FileGraphProvider:
    """Serves a C1-shaped graph from a JSON file (default: the bundled synthetic sample).

    Labelled ``mock``: the file is not C1 output for the uploaded deck, so the mock-graph warning
    applies. A file marked curated (``"curated": true`` at the top level, or a top-level ``note``
    equal to ``"curated"``) was written or corrected by people: it is labelled ``curated``, which
    features treat as real, and the response says it is hand-made. The file's own ``lecture_id``
    is kept.
    """

    name = "file"

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @classmethod
    def for_deck(cls, directory: Path, deck_filename: str) -> "FileGraphProvider":
        """Provider for the per-lecture graph ``<deck stem>.graph.json`` in ``directory``."""
        return cls(curated_graph_path(directory, deck_filename))

    def build(self, materials: MaterialsDoc) -> GraphContext:
        ref = self._path or resources.files("edusync_c2.difficulty").joinpath(*_SAMPLE_RESOURCE)
        try:
            response = C1GraphResponse.model_validate_json(ref.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # ValidationError and JSON/Unicode errors are ValueErrors
            raise GraphProviderError(500, "graph file could not be loaded") from None
        if is_curated(response):
            source = InputSource(
                name="c1_graph", kind="curated", detail=f"graph file: {ref.name} (curated)"
            )
        else:
            source = InputSource(name="c1_graph", kind="mock", detail=f"graph file: {ref.name}")
        return GraphContext(response=response, source=source)
