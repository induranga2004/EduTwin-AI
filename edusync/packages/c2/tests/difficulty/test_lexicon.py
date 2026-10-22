"""Domain lexicon: resource file format, parser, cached loader and plural folding."""

import re
from importlib.resources import files

import pytest

from edusync_c2.difficulty.features import lexicon
from edusync_c2.difficulty.features.lexicon import (
    LexiconError,
    in_lexicon,
    load_lexicon,
    parse_lexicon,
)

TERM_RE = re.compile(r"^[a-z][a-z0-9]*$")
MUST_CONTAIN = [
    "encryption",
    "gradient",
    "authentication",
    "concurrency",
    "kernel",
    "database",
    "protocol",
    "algorithm",
    "tensor",
    "sql",
    "api",
    "cpu",
    "tde",
    "aes",
    "rsa",
    "ldap",
]
STOP_WORDS = [
    "the",
    "and",
    "of",
    "because",
    "system",
    "information",
    "example",
    "data",
    "model",
]


def _raw() -> bytes:
    return files("edusync_c2.difficulty").joinpath("resources", "domain_terms.txt").read_bytes()


def _term_lines() -> list[str]:
    return [ln for ln in _raw().decode("utf-8").split("\n")[:-1] if not ln.startswith("#")]


@pytest.fixture(autouse=True)
def _fresh_cache():
    load_lexicon.cache_clear()
    yield
    load_lexicon.cache_clear()


class _FakeResource:
    def __init__(self, outcomes: list):
        self.outcomes = outcomes
        self.reads = 0

    def joinpath(self, *parts: str) -> "_FakeResource":
        assert parts == ("resources", "domain_terms.txt")
        return self

    def read_text(self, encoding: str) -> str:
        assert encoding == "utf-8"
        outcome = self.outcomes[min(self.reads, len(self.outcomes) - 1)]
        self.reads += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_file_ends_with_single_lf_and_has_no_cr_or_trailing_space():
    raw = _raw()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b"\r" not in raw
    for line in raw.decode("utf-8").split("\n")[:-1]:
        assert line == line.rstrip(), repr(line)


def test_comment_lines_only_at_top():
    lines = _raw().decode("utf-8").split("\n")[:-1]
    first_term = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
    assert first_term > 0
    assert all(not ln.startswith("#") for ln in lines[first_term:])


def test_terms_sorted_unique_lowercase_alnum():
    terms = _term_lines()
    assert terms == sorted(terms)
    assert len(terms) == len(set(terms))
    bad = [t for t in terms if not TERM_RE.match(t)]
    assert bad == []


def test_term_count_in_range():
    assert 400 <= len(_term_lines()) <= 1000


def test_contains_required_terms_and_no_stop_words():
    terms = set(_term_lines())
    assert [t for t in MUST_CONTAIN if t not in terms] == []
    assert [w for w in STOP_WORDS if w in terms] == []


def test_load_lexicon_reads_real_file():
    lex = load_lexicon()
    assert isinstance(lex, frozenset)
    assert lex == frozenset(_term_lines())


def test_parse_skips_comments_and_blank_lines():
    text = "# header\n\n  kernel  \n#hidden\nsql\n\n"
    assert parse_lexicon(text) == frozenset({"kernel", "sql"})


def test_parse_does_not_lowercase():
    assert parse_lexicon("SQL\n") == frozenset({"SQL"})


@pytest.mark.parametrize("text", ["", "\n\n", "# only comments\n# more\n", "   \n"])
def test_parse_empty_raises(text):
    with pytest.raises(LexiconError):
        parse_lexicon(text)


def test_lexicon_error_is_runtime_error():
    assert issubclass(LexiconError, RuntimeError)


def test_load_missing_file_raises_lexicon_error(monkeypatch):
    fake = _FakeResource([FileNotFoundError("gone")])
    monkeypatch.setattr(lexicon, "files", lambda pkg: fake)
    with pytest.raises(LexiconError, match="gone") as info:
        load_lexicon()
    assert isinstance(info.value.__cause__, FileNotFoundError)


def test_load_empty_file_raises_lexicon_error(monkeypatch):
    fake = _FakeResource(["# nothing\n"])
    monkeypatch.setattr(lexicon, "files", lambda pkg: fake)
    with pytest.raises(LexiconError):
        load_lexicon()


def test_load_is_cached(monkeypatch):
    fake = _FakeResource(["kernel\nsql\n"])
    monkeypatch.setattr(lexicon, "files", lambda pkg: fake)
    first = load_lexicon()
    assert load_lexicon() is first
    assert fake.reads == 1


def test_load_failure_is_not_cached(monkeypatch):
    fake = _FakeResource([OSError("busy"), "kernel\n"])
    monkeypatch.setattr(lexicon, "files", lambda pkg: fake)
    with pytest.raises(LexiconError):
        load_lexicon()
    assert load_lexicon() == frozenset({"kernel"})
    assert fake.reads == 2


LEX = frozenset({"kernel", "index", "process", "class", "sql", "os"})


@pytest.mark.parametrize(
    "word",
    ["kernel", "kernels", "index", "indexes", "process", "processes", "classes", "sqls", "os"],
)
def test_in_lexicon_true(word):
    assert in_lexicon(word, LEX)


@pytest.mark.parametrize(
    "word",
    ["", "proces", "clas", "classs", "kernelz", "banana", "bananas", "ss", "s", "es"],
)
def test_in_lexicon_false(word):
    assert not in_lexicon(word, LEX)


def test_ss_guard_does_not_fold_to_truncated_stem():
    lex = frozenset({"proces", "clas"})
    assert not in_lexicon("process", lex)
    assert not in_lexicon("class", lex)
    assert in_lexicon("processes", frozenset({"process"}))


def test_in_lexicon_with_real_lexicon():
    lex = load_lexicon()
    assert in_lexicon("kernels", lex)
    assert in_lexicon("databases", lex)
    assert in_lexicon("indexes", lex)
    assert not in_lexicon("because", lex)
