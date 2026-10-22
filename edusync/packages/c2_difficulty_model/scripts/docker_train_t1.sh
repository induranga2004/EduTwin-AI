#!/usr/bin/env bash
# Entry point inside the T1 training container (see Dockerfile.train for the docker run command).
# Copies the code and data into the container filesystem (bind mounts are slow on Windows), then
# trains, or with --pytest runs the ml-marked tests. Outputs go to /runs/<name>.
set -euo pipefail

mkdir -p /work
tar -C /src -cf - \
  --exclude=__pycache__ \
  --exclude=packages/c2_difficulty_model/data/real \
  --exclude=packages/c2_difficulty_model/runs \
  --exclude=packages/c2_difficulty_model/cache \
  packages/core packages/c2 packages/c2_difficulty_model \
  packages/blueprint_model/artifacts/base | tar -C /work -xf -

cd /work/packages/c2_difficulty_model
export PYTHONPATH=/work/packages/core/src:/work/packages/c2/src:/work/packages/c2_difficulty_model/src
export OMP_NUM_THREADS="${T1_THREADS:-$(nproc)}"

if [ "${1:-}" = "--pytest" ]; then
  exec python -m pytest tests/test_t1_ml.py -m ml -q -p no:cacheprovider --no-cov \
    --import-mode=importlib -o addopts=""
fi

out=/runs/t1
for arg in "$@"; do
  if [ "$arg" = "--limit-rows" ]; then out=/runs/t1_smoke; fi
done
exec python scripts/train_t1.py --out-dir "$out" "$@"
