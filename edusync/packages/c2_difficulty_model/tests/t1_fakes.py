"""A torch-free stand-in for ``edusync_c2_model.t1.torch_impl`` plus synthetic public examples."""

from pathlib import Path

import numpy as np

from edusync_c2_model.public_data.schema import TextExample
from edusync_c2_model.public_data.splits import make_group_split, partition
from edusync_c2_model.t1.quantize import remove_new_files, snapshot_dir


def encode_words(texts: list[str]) -> list[list[int]]:
    return [[101, *[sum(map(ord, w)) % 900 + 1 for w in t.split()], 102] for t in texts]


class FakeModel:
    pass


class FakeBackend:
    """Predicts from sequence length; ``improve`` False makes val get worse after epoch 1."""

    def __init__(self, model, config, improve=True):
        self.config, self.improve, self.state, self.steps, self.scales = config, improve, 0, 0, []

    def train_step(self, batch, lr_scale):
        self.steps += 1
        self.scales.append(lr_scale)
        return 1.0 / self.steps

    def predict_batch(self, batch):
        sign = 1.0 if (self.improve or self.state == 0) else -1.0
        return sign * batch.attention_mask.sum(axis=1).astype(np.float64) / 10.0

    def snapshot(self):
        self.state += 1
        return self.state

    def restore(self, state):
        self.state = state


def make_examples(source, n_groups=20, per_group=1, sign=1.0):
    out = []
    for g in range(n_groups):
        for k in range(per_group):
            n = 3 + (g * 7 + k * 3) % 11
            target = sign * (float(n) + (g % 3) * 0.1 + k)
            out.append(
                TextExample(source, " ".join(f"w{i}" for i in range(n)), target, k, f"{source}{g}")
            )
    return out


def make_parts():
    parts = {}
    for source, per_group in (("clear", 1), ("onestop", 3)):
        examples = make_examples(source, 20, per_group, -1.0 if source == "clear" else 1.0)
        split = make_group_split({e.group_id for e in examples}, seed=1, source=source)
        parts[source] = partition(examples, split)
    return parts, {"clear": True, "onestop": False}


class FakeImpl:
    """Mimics the torch_impl module surface used by ``pipeline.run``."""

    def __init__(self, improve=True):
        self.improve = improve
        self.calls = []

    def set_threads(self, threads):
        self.calls.append(("threads", threads))

    def seed_everything(self, seed):
        self.calls.append(("seed", seed))

    def load_encoder(self, tokenizer_json, max_length):
        return encode_words

    def build_model(self, base_dir, dropout):
        return FakeModel()

    def TorchBackend(self, model, config):
        return FakeBackend(model, config, self.improve)

    def export_onnx(self, model, encode, path):
        Path(path).write_bytes(b"x" * 4096)
        return "legacy"

    def torch_predictor(self, model, encode):
        return lambda texts: np.array([len(i) / 10.0 for i in encode(texts)])

    def open_session(self, path, threads):
        shift = 0.0005 if "int8" in Path(path).name else 0.0

        def run(feeds):
            return (feeds["attention_mask"].sum(axis=1, keepdims=True) / 10.0 + shift).astype(
                np.float32
            )

        io_in = [
            ("input_ids", "tensor(int64)", ["batch", "seq"]),
            ("attention_mask", "tensor(int64)", ["batch", "seq"]),
        ]
        return run, io_in, [("score", "tensor(float)", ["batch", 1])]

    def quantize_int8(self, fp32, int8, workdir):
        before = snapshot_dir(workdir)
        Path(workdir, "model_prep.onnx").write_bytes(b"tmp")
        Path(int8).write_bytes(b"y" * 1024)
        return remove_new_files(workdir, before, keep=[int8])

    def versions(self):
        return {"fake": "1"}
