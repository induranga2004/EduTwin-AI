import numpy as np
import pytest

from edusync_blueprint_model.predict import OnnxClassifier, softmax


class Enc:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)


class FakeTokenizer:
    def encode_batch(self, texts):
        return [Enc(list(range(1, len(t.split()) + 1))) for t in texts]


class FakeSession:
    def __init__(self):
        self.feeds = None

    def run(self, _outputs, feeds):
        self.feeds = feeds
        n = feeds["input_ids"].shape[0]
        return [np.tile(np.array([[0.0, 2.0, 1.0]]), (n, 1))]


def make(max_length=4):
    session = FakeSession()
    return OnnxClassifier(session, FakeTokenizer(), ["a", "b", "c"], max_length), session


def test_softmax_sums_to_one_and_is_stable() -> None:
    p = softmax(np.array([[1000.0, 1001.0], [0.0, 0.0]]))
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert p[0, 1] > p[0, 0]
    assert p[1, 0] == pytest.approx(0.5)


def test_predict_labels_and_probabilities() -> None:
    clf, session = make()
    preds = clf.predict(["one two", "x"])
    assert [p.label for p in preds] == ["b", "b"]
    assert sum(preds[0].probabilities.values()) == pytest.approx(1.0)
    assert preds[0].score == pytest.approx(preds[0].probabilities["b"])
    assert session.feeds["input_ids"].dtype == np.int64


def test_padding_and_truncation() -> None:
    clf, session = make(max_length=3)
    clf.logits(["a b c d e f", "a"])
    ids, mask = session.feeds["input_ids"], session.feeds["attention_mask"]
    assert ids.shape == (2, 3)
    assert mask.shape == (2, 3)
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert ids[1].tolist() == [1, 0, 0]


def test_empty_input() -> None:
    clf, _ = make()
    assert clf.logits([]).shape == (0, 3)
    assert clf.predict([]) == []
