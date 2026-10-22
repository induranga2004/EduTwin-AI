"""Build the score artifact folder (``runs/score``) used by C2's layered analysis.

Reads the label-free real-deck feature cache (reference), ``runs/t1`` (INT8 ONNX) and ``runs/t2``
(numpy), computes the T1/T2 difficulty distributions over the CLEAR TRAIN texts (the first
``--train-limit`` passages, to bound the run time), fits the combiner from the reference structure
and writes everything plus a manifest and model card. Needs the MiniLM base weights for T2.
"""

import argparse
import json
import sys
from pathlib import Path

from edusync_c2.difficulty.scoring.text_models import (
    default_session_factory,
    default_tokenizer_factory,
)
from edusync_c2_model.ingest import load_reference_features
from edusync_c2_model.models.embeddings import encoder_embedder
from edusync_c2_model.score_artifact import build_artifact, clear_train_texts

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    PACKAGE_ROOT.parent / "blueprint_model/artifacts/base/sentence-transformers--all-MiniLM-L6-v2"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PACKAGE_ROOT / "data/real/manifest.csv")
    parser.add_argument("--cache-dir", type=Path, default=PACKAGE_ROOT / "cache/real")
    parser.add_argument("--data-dir", type=Path, default=PACKAGE_ROOT / "data/public")
    parser.add_argument("--t1-dir", type=Path, default=PACKAGE_ROOT / "runs/t1")
    parser.add_argument("--t2-dir", type=Path, default=PACKAGE_ROOT / "runs/t2")
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT / "runs/score")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--include-fp32", action="store_true")
    args = parser.parse_args(argv)
    try:
        rows = load_reference_features(args.cache_dir, args.manifest)
        texts = clear_train_texts(args.data_dir, args.train_limit)
        embed, _ = encoder_embedder(args.model_dir)
        manifest = build_artifact(
            rows=rows,
            train_texts=texts,
            t1_dir=args.t1_dir,
            t2_dir=args.t2_dir,
            out_dir=args.out,
            embed=embed,
            session_factory=default_session_factory,
            tokenizer_factory=default_tokenizer_factory,
            include_fp32=args.include_fp32,
            train_note=f"Percentile tables use the first {len(texts)} CLEAR train passages.",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {k: manifest[k] for k in ("n_decks", "n_concepts", "weights", "cutoffs", "provisional")}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
