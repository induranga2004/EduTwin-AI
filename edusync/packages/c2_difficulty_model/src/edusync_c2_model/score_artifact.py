"""Build the score artifact folder served by C2's layered analysis (``runs/score``).

Inputs: label-free reference rows (the real-deck feature cache), the trained T1 (ONNX) and T2
(numpy) folders, and the CLEAR train texts. Output: ``reference.json``, ``combiner.json``,
``text_tables.json``, the model files under the names C2 expects, ``artifact_manifest.json`` and
``model_card.md``. Nothing here reads labels; the combiner is fitted from the reference structure.
"""

import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from edusync_c2.difficulty.registry import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SPECS,
    feature_schema_hash,
)
from edusync_c2.difficulty.schemas import FeatureStatus, FeatureValue
from edusync_c2.difficulty.scoring.artifact import (
    MANIFEST_NAME,
    REFERENCE_NAME,
    TABLES_NAME,
    load_artifact,
)
from edusync_c2.difficulty.scoring.combiner import COMBINER_NAME, fit_combiner
from edusync_c2.difficulty.scoring.confidence import WEIGHTS as CONFIDENCE_WEIGHTS
from edusync_c2.difficulty.scoring.reference import build_reference, quantile_table
from edusync_c2.difficulty.scoring.signals import group_scores, score_features
from edusync_c2.difficulty.scoring.text_models import TEXT_CAP, T1Scorer, T2Scorer
from edusync_c2_model.examples import FeatureRow
from edusync_c2_model.public_data.clear import load_clear
from edusync_c2_model.public_data.licences import SOURCES
from edusync_c2_model.public_data.splits import partition, read_split

ARTIFACT_FORMAT = 1
LICENCE_NOTE = (
    "T1/T2 are derived from CLEAR (CC BY-NC-SA 4.0) and OneStopEnglish (CC BY-SA 4.0): "
    "research and non-commercial use only unless the CLEAR terms are cleared."
)
COPIES = {  # source file name in the model folders -> name in the artifact
    "t1": {
        "model_int8.onnx": "t1_int8.onnx",
        "tokenizer.json": "tokenizer.json",
        "inference_config.json": "inference_config.json",
    },
    "t1_fp32": {"model_fp32.onnx": "t1_fp32.onnx"},
    "t2": {"t2.npz": "t2.npz", "t2_config.json": "t2_config.json"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row_features(row: FeatureRow) -> list[FeatureValue]:
    """Rebuild ``FeatureValue`` objects (registry order) from a cached feature row."""
    out = []
    for spec, value, status in zip(FEATURE_SPECS, row.values, row.statuses, strict=True):
        computed = status == "computed" and value is not None
        out.append(
            FeatureValue(
                name=spec.name,
                group=spec.group,
                value=value if computed else None,
                status=FeatureStatus(status) if computed else FeatureStatus.UNAVAILABLE,
                source="reference",
            )
        )
    return out


def copy_models(t1_dir: Path, t2_dir: Path, out_dir: Path, *, include_fp32: bool) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = [(Path(t1_dir), COPIES["t1"]), (Path(t2_dir), COPIES["t2"])]
    if include_fp32:
        plan.append((Path(t1_dir), COPIES["t1_fp32"]))
    names = []
    for source, files in plan:
        for src, dst in files.items():
            shutil.copyfile(source / src, out_dir / dst)
            names.append(dst)
    return names


class _EncoderShim:
    """Makes a bare ``embed`` callable look like ``MiniLMEncoder`` for ``load_artifact``."""

    def __init__(self, embed: Callable) -> None:
        self.encode = embed


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def model_card(manifest: dict) -> str:
    weights = ", ".join(f"{k} {v:.3f}" for k, v in manifest["weights"].items())
    state = "provisional" if manifest["provisional"] else "established"
    lines = [
        "# Score artifact (EduSync C2 layered analysis)",
        "",
        f"- Reference: {manifest['n_decks']} decks, {manifest['n_concepts']} concepts ({state}).",
        f"- Feature schema {manifest['feature_schema_version']} "
        f"(hash {manifest['feature_schema_hash'][:12]}).",
        f"- Combiner weights: {weights}. Cut-offs: Low below {manifest['cutoffs']['low']}, "
        f"High from {manifest['cutoffs']['high']} (percentiles of the combined score).",
        "- Text signal: mean percentile of T1 (INT8 ONNX, fp32 fallback) and T2 (numpy ridge), "
        f"each against its own predictions on public training texts. {manifest['train_note']}",
        "- Confidence is evidence quality (coverage, agreement, reference size), "
        "not a probability.",
        "- Weights, cut-offs and the reference are unvalidated; the text models were trained "
        "on general English, not lecture slides.",
        "",
        f"Licence: {manifest['licence_note']}",
        "",
    ]
    return "\n".join(lines)


def build_artifact(
    *,
    rows: Sequence[FeatureRow],
    train_texts: Sequence[str],
    t1_dir: Path,
    t2_dir: Path,
    out_dir: Path,
    embed: Callable,
    session_factory: Callable,
    tokenizer_factory: Callable,
    include_fp32: bool = False,
    train_note: str = "",
) -> dict:
    """Write the artifact folder and return its manifest (weights, cut-offs, sizes, hashes)."""
    out_dir = Path(out_dir)
    files = copy_models(t1_dir, t2_dir, out_dir, include_fp32=include_fp32)
    reference = build_reference(
        rows, schema_version=FEATURE_SCHEMA_VERSION, schema_hash=feature_schema_hash()
    )
    reference.save(out_dir / REFERENCE_NAME)

    t1 = T1Scorer.load(
        out_dir, [0.0], session_factory=session_factory, tokenizer_factory=tokenizer_factory
    )
    t2 = T2Scorer.load(out_dir, embed, [0.0])
    t1.z_table = quantile_table(t1.z(list(train_texts)))
    t2.z_table = quantile_table(t2.z(list(train_texts)))
    tables = {"t1_z_table": t1.z_table, "t2_z_table": t2.z_table}
    (out_dir / TABLES_NAME).write_text(json.dumps(tables), encoding="utf-8", newline="\n")

    texts = [r.text[:TEXT_CAP] for r in rows]
    distinct = sorted({t for t in texts if t.strip()})
    p1 = dict(zip(distinct, t1.percentiles(distinct), strict=True))
    p2 = dict(zip(distinct, t2.percentiles(distinct), strict=True))
    signal_rows = []
    for row, text in zip(rows, texts, strict=True):
        groups = group_scores(score_features(row_features(row), reference))
        signal_rows.append({**groups, "text": _mean([p1.get(text), p2.get(text)])})
    combiner = fit_combiner(signal_rows)
    combiner.save(out_dir / COMBINER_NAME)

    names = [REFERENCE_NAME, COMBINER_NAME, TABLES_NAME, *files]
    manifest = {
        "format": ARTIFACT_FORMAT,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "n_features": len(FEATURE_NAMES),
        "n_decks": reference.n_decks,
        "n_concepts": reference.n_concepts,
        "provisional": reference.provisional,
        "weights": combiner.weights,
        "cutoffs": {"low": combiner.low, "high": combiner.high},
        "confidence_weights": CONFIDENCE_WEIGHTS,
        "n_train_texts": len(train_texts),
        "train_note": train_note,
        "licence_note": LICENCE_NOTE,
        "files": {n: sha256_file(out_dir / n) for n in sorted(names)},
    }
    (out_dir / "model_card.md").write_text(model_card(manifest), encoding="utf-8", newline="\n")
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    load_artifact(  # fail now, not at the first request, if the folder is not loadable
        out_dir,
        model_dir=out_dir,
        encoder_factory=lambda _dir: _EncoderShim(embed),
        session_factory=session_factory,
        tokenizer_factory=tokenizer_factory,
    )
    return manifest


def clear_train_texts(
    data_dir: Path, limit: int | None, loader: Callable = load_clear
) -> list[str]:
    """CLEAR texts of the sealed TRAIN part (first ``limit`` in file order; None = all)."""
    source = next(src for src in SOURCES if src.key == "clear")
    examples = loader(Path(data_dir) / source.filename)
    split = read_split(Path(data_dir) / "splits" / "clear_split.json")
    texts = [e.text for e in partition(examples, split)["train"]]
    return texts if limit is None else texts[:limit]
