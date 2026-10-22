"""The score artifact folder: reference, combiner, text-model tables and the T1/T2 models.

Files: ``reference.json``, ``combiner.json``, ``text_tables.json`` (z quantiles of T1 and T2 over
the public training texts), ``t2.npz`` + ``t2_config.json``, ``t1_int8.onnx`` (or ``t1_fp32.onnx``),
``tokenizer.json``, ``inference_config.json``, ``artifact_manifest.json``. A missing text model only
removes the text signal's model (with a warning); a missing reference or combiner is an error.
"""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from edusync_c2.difficulty.features.semantic_loader import get_default_encoder
from edusync_c2.difficulty.registry import feature_schema_hash
from edusync_c2.difficulty.scoring.combiner import COMBINER_NAME, Combiner
from edusync_c2.difficulty.scoring.reference import Reference, ScoringError
from edusync_c2.difficulty.scoring.text_models import (
    SessionFactory,
    T1Scorer,
    T2Scorer,
    TokenizerFactory,
    default_session_factory,
    default_tokenizer_factory,
)
from edusync_c2.logger import get_logger

logger = get_logger(__name__)

REFERENCE_NAME = "reference.json"
TABLES_NAME = "text_tables.json"
MANIFEST_NAME = "artifact_manifest.json"


@dataclass
class ScoreArtifact:
    reference: Reference
    combiner: Combiner
    t1: T1Scorer | None = None
    t2: T2Scorer | None = None
    manifest: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def load_artifact(
    directory: Path,
    *,
    model_dir: Path | None = None,
    encoder_factory: Callable[[Path | None], object] = get_default_encoder,
    session_factory: SessionFactory = default_session_factory,
    tokenizer_factory: TokenizerFactory = default_tokenizer_factory,
) -> ScoreArtifact:
    """Load the artifact folder; ``model_dir`` is the MiniLM folder T2 needs for embeddings."""
    directory = Path(directory)
    reference = Reference.load(directory / REFERENCE_NAME)
    if reference.feature_schema_hash != feature_schema_hash():
        raise ScoringError("the artifact was built for a different feature schema")
    combiner = Combiner.load(directory / COMBINER_NAME)
    try:
        tables = json.loads((directory / TABLES_NAME).read_text(encoding="utf-8"))
        t1_table = [float(x) for x in tables["t1_z_table"]]
        t2_table = [float(x) for x in tables["t2_z_table"]]
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ScoringError(f"cannot read the text tables: {type(exc).__name__}") from exc
    warnings: list[str] = []
    t1: T1Scorer | None = None
    t2: T2Scorer | None = None
    try:
        t1 = T1Scorer.load(
            directory,
            t1_table,
            session_factory=session_factory,
            tokenizer_factory=tokenizer_factory,
        )
    except ScoringError as exc:
        logger.warning("t1_unavailable reason=%s", exc)
        warnings.append("Text model T1 is unavailable; the text signal uses T2 only")
    encoder = encoder_factory(model_dir)
    if encoder is None:
        logger.warning("t2_unavailable reason=encoder")
        warnings.append("Text model T2 is unavailable (MiniLM weights not configured)")
    else:
        try:
            t2 = T2Scorer.load(directory, encoder.encode, t2_table)  # type: ignore[attr-defined]
        except ScoringError as exc:
            logger.warning("t2_unavailable reason=%s", exc)
            warnings.append("Text model T2 is unavailable; the text signal uses T1 only")
    if t1 is None and t2 is None:
        warnings.append("No text model is available; the text signal is missing")
    return ScoreArtifact(reference, combiner, t1, t2, manifest, warnings)


_cache: dict[str, ScoreArtifact] = {}
_lock = threading.Lock()


def get_artifact(directory: Path, *, model_dir: Path | None = None, **factories) -> ScoreArtifact:
    """The process-wide artifact for ``directory``; failures are not cached."""
    key = str(Path(directory).resolve())
    with _lock:
        if key not in _cache:
            _cache[key] = load_artifact(Path(directory), model_dir=model_dir, **factories)
        return _cache[key]


def clear_artifact_cache() -> None:
    with _lock:
        _cache.clear()
