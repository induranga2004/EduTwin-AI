from edusync_c1.schemas import Concept
from edusync_c1.semantic import EXACT_METHOD, SEMANTIC_METHOD, SemanticMatcher

CONCEPTS = [Concept(name="Gradient Descent"), Concept(name="Loss Function"), Concept(name=" ")]


def test_empty_inputs():
    matcher = SemanticMatcher(enabled=False)
    assert matcher.match("", CONCEPTS).method == "none"
    assert matcher.match("text", []).method == "none"
    assert matcher.match("text", [Concept(name=" ")]).matches == []


def test_exact_fallback_when_disabled():
    result = SemanticMatcher(enabled=False).match("We study gradient descent today", CONCEPTS)
    assert result.method == EXACT_METHOD
    assert [m.name for m in result.matches] == ["Gradient Descent"]
    assert result.matches[0].similarity == 1.0
    assert result.model is None


def test_fallback_when_loader_fails_only_tried_once():
    calls = []

    def loader(name):
        calls.append(name)
        raise ImportError("no sentence-transformers")

    matcher = SemanticMatcher(loader=loader)
    assert matcher.match("loss function", CONCEPTS).method == EXACT_METHOD
    assert matcher.match("loss function", CONCEPTS).method == EXACT_METHOD
    assert len(calls) == 1


class _Rows(list):
    """Minimal stand-in for a numpy matrix: supports ``rows @ vector``."""

    def __matmul__(self, vec):
        return [sum(a * b for a, b in zip(row, vec, strict=True)) for row in self]

    def __getitem__(self, item):
        result = super().__getitem__(item)
        return _Rows(result) if isinstance(item, slice) else result


class FakeModel:
    def encode(self, texts, **kwargs):
        # text vector first, then one vector per concept
        vecs = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]
        return _Rows(vecs[: len(texts)])


def test_semantic_matching_with_injected_model():
    matcher = SemanticMatcher(loader=lambda name: FakeModel(), threshold=0.5)
    result = matcher.match("anything", CONCEPTS[:2])
    assert result.method == SEMANTIC_METHOD
    assert [m.name for m in result.matches] == ["Gradient Descent"]
    assert result.matches[0].similarity == 0.8
    assert result.model == matcher.model_name
    # explicit threshold overrides the default
    assert len(matcher.match("anything", CONCEPTS[:2], threshold=0.0).matches) == 2
    assert matcher.match("anything", CONCEPTS[:2], threshold=0.99).matches == []


def test_encode_failure_falls_back():
    class Broken:
        def encode(self, *args, **kwargs):
            raise RuntimeError("boom")

    matcher = SemanticMatcher(loader=lambda name: Broken())
    result = matcher.match("loss function here", CONCEPTS)
    assert result.method == EXACT_METHOD
    assert [m.name for m in result.matches] == ["Loss Function"]
