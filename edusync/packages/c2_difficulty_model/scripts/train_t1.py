"""Train T1 (MiniLM regressor) on CLEAR + OneStopEnglish, export to ONNX, quantize to INT8, compare.

Needs torch, transformers, tokenizers, onnx and onnxruntime, so run it inside the training
container (see ``Dockerfile.train`` and ``scripts/docker_train_t1.sh``). Splits are the sealed
ones in ``<data-dir>/splits``. Smoke run: ``--limit-rows 64 --epochs 1``.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    PACKAGE_ROOT.parent
    / "blueprint_model"
    / "artifacts"
    / "base"
    / "sentence-transformers--all-MiniLM-L6-v2"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE_ROOT / "data" / "public")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=PACKAGE_ROOT / "runs" / "t1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--freeze-encoder", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    from edusync_c2_model.t1 import pipeline, torch_impl
    from edusync_c2_model.t1.config import T1Config

    config = T1Config(
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        threads=args.threads,
        loss=args.loss,
        freeze_encoder=args.freeze_encoder,
        limit_rows=args.limit_rows,
    )
    parts, higher_is_easier = pipeline.load_sources(args.data_dir)
    summary = pipeline.run(
        config,
        parts,
        higher_is_easier,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        impl=torch_impl,
        n_boot=args.n_boot,
        log=lambda message: print(message, flush=True),
    )
    print((args.out_dir / "report.md").read_text(encoding="utf-8"))
    print(f"shipped: {summary['quantization']['decision']['ship']}; artifacts in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
