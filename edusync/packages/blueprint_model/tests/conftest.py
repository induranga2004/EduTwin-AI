from pathlib import Path

import pytest

TOML = """
[model]
name = "org/tiny-model"
revision = "abc1234567"
max_length = 16

[data]
path = "data.csv"
text_column = "text"
label_column = "label"
val_fraction = 0.2
test_fraction = 0.2
seed = 1

[train]
epochs = 1
batch_size = 4
encoder_lr = 3e-5
head_lr = 1e-3
weight_decay = 0.01
warmup_ratio = 0.1
dropout = 0.1
threads = 0
seed = 1

[quantize]
weight_type = "qint8"
per_channel = false
op_types = ["MatMul"]
benchmark_runs = 2

[paths]
artifacts_dir = "out"
"""


def write_csv(path: Path, per_class: int = 10) -> None:
    rows = ["text,label"]
    for label in ("a", "b", "c"):
        rows += [f"sample {i} for {label},{label}" for i in range(per_class)]
    path.write_text("\n".join(rows), encoding="utf-8")


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    (tmp_path / "cfg").mkdir()
    path = tmp_path / "cfg" / "bp.toml"
    path.write_text(TOML.replace('"data.csv"', '"../data.csv"').replace('"out"', '"../out"'))
    write_csv(tmp_path / "data.csv")
    return path
