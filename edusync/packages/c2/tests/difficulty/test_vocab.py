"""Vocabulary analyzer: exact numbers on a fake tokenizer, real-resource ordering, loading."""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from importlib import resources
from pathlib import Path
from typing import ClassVar

import pytest

from edusync_c2.difficulty.features import vocab
from edusync_c2.difficulty.features.lexicon import LexiconError
from edusync_c2.difficulty.features.vocab import (
    MIN_PIECES_UNFAMILIAR,
    VocabStats,
    VocabularyAnalyzer,
    get_default_analyzer,
    reset_default_analyzer,
)
from edusync_c2.difficulty.features.wordpiece import UNK, WordPieceTokenizer

pytestmark = pytest.mark.usefixtures("fresh_default_analyzer")

LOGGER = "edusync_c2.difficulty.features.vocab"
SRC_DIR = Path(__file__).resolve().parents[2] / "src"

EASY = (
    "We went to the park on a sunny day. The kids ran to the lake and played with a ball. "
    "My mother made lunch for all of us and we sat on the grass to eat it. After that we "
    "walked home and watched a movie. It was a good day and everyone was happy. "
    "Tomorrow we will go to the beach with our friends and swim in the sea."
)
MEDIUM = (
    "A database stores data in tables. Each table has rows and columns, and queries ask the "
    "database for the rows that match a rule. Indexing makes a query faster because the server "
    "does not scan every row. When two users change the same row at once, the system uses a "
    "lock so that one change waits for the other. Backups copy the data to another disk in "
    "case the server fails."
)
HARD = (
    "Backpropagation computes the gradient of the loss with respect to every parameter by "
    "applying the chain rule layer by layer. The Hessian eigenvalue spectrum governs "
    "hyperparameter sensitivity, while regularization and normalization stabilize convergence. "
    "In cryptography, the decryption oracle exposes ciphertext malleability, so authenticated "
    "encryption and nonce reuse resistance matter. Asymptotic complexity and idempotency "
    "constrain the scheduler and the serializable transaction isolation."
)
CS = (
    "The kernel schedules each process on the cpu while the database enforces transactions. "
    "An api exposes the protocol to the client and the algorithm encrypts the packets."
)


class FakeTokenizer:
    """Maps a few words to fixed pieces; every other word is a single piece."""

    TABLE: ClassVar = {
        "alpha": ["al", "##pha"],
        "delta": ["d", "##e", "##l"],
        "zeta": [UNK],
        "skip": [],
    }

    def tokenize(self, text: str) -> list[str]:
        return self.TABLE.get(text, [text])


def fake_analyzer(lexicon: frozenset[str] = frozenset({"beta", "gamma"})) -> VocabularyAnalyzer:
    return VocabularyAnalyzer(FakeTokenizer(), lexicon)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def real() -> VocabularyAnalyzer:
    return VocabularyAnalyzer.from_resources()


def stats(analyzer: VocabularyAnalyzer, text: str) -> VocabStats:
    result = analyzer.analyze(text)
    assert result is not None
    return result


# ------------------------------------------------------------------ fake tokenizer, exact numbers
def test_exact_numbers_with_fake_tokenizer():
    # alpha=2 pieces, beta=1, gamma=1, delta=3, zeta=[UNK]; "x" is ignored (one letter)
    result = stats(fake_analyzer(), "Alpha beta x gamma-delta zeta")
    assert result == VocabStats(
        words=5, fragmentation=8 / 5, unfamiliar_share=3 / 5, domain_share=2 / 5
    )


def test_min_pieces_constant():
    assert MIN_PIECES_UNFAMILIAR == 2


def test_single_piece_word_is_familiar():
    result = stats(fake_analyzer(), "beta gamma")
    assert result.unfamiliar_share == 0.0
    assert result.fragmentation == 1.0


def test_unk_alone_is_unfamiliar_with_one_piece():
    result = stats(fake_analyzer(), "zeta")
    assert result == VocabStats(words=1, fragmentation=1.0, unfamiliar_share=1.0, domain_share=0.0)


def test_word_with_no_pieces_is_skipped():
    result = stats(fake_analyzer(), "skip beta")
    assert result.words == 1
    assert result.fragmentation == 1.0


def test_only_skipped_words_is_none():
    assert fake_analyzer().analyze("skip skip") is None


def test_lookup_is_lowercased_and_plural_folded():
    result = stats(fake_analyzer(), "BETA Gammas")
    assert result.domain_share == 1.0


def test_tokenizer_receives_lowercase_words():
    seen: list[str] = []

    class Spy:
        def tokenize(self, text: str) -> list[str]:
            seen.append(text)
            return [text]

    VocabularyAnalyzer(Spy(), frozenset()).analyze("Hello WORLD x")  # type: ignore[arg-type]
    assert seen == ["hello", "world"]


def test_stats_are_frozen():
    with pytest.raises(AttributeError):
        stats(fake_analyzer(), "beta").words = 3  # type: ignore[misc]
    assert replace(stats(fake_analyzer(), "beta"), words=2).words == 2


# ------------------------------------------------------------------ real resources
def test_empty_and_wordless_text_is_none(real):
    assert real.analyze("") is None
    assert real.analyze("123 ... !!! a b _ -") is None


def test_single_letter_words_are_ignored(real):
    assert stats(real, "a b c kernel").words == 1


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("real-time", 2),
        ("don't", 1),  # "don" and the single letter "t"
        ("gpu8memory", 2),
        ("snake_case", 2),
        ("one,two;three", 3),
        ("naive", 1),
        ("na\u00efve", 1),
        ("\u00c9cole caf\u00e9", 2),
    ],
)
def test_word_splitting(real, text, words):
    assert stats(real, text).words == words


def test_accent_is_folded_by_the_tokenizer(real):
    assert stats(real, "na\u00efve") == stats(real, "naive")


def test_very_long_word_is_unk_and_one_piece(real):
    result = stats(real, "a" * 120)
    assert real.tokenizer.tokenize("a" * 120) == [UNK]
    assert result == VocabStats(words=1, fragmentation=1.0, unfamiliar_share=1.0, domain_share=0.0)


def test_difficulty_ordering_on_paragraphs(real):
    easy, medium, hard = (stats(real, t) for t in (EASY, MEDIUM, HARD))
    for text in (EASY, MEDIUM, HARD):
        assert 55 <= len(text.split()) <= 80
    assert easy.fragmentation < medium.fragmentation < hard.fragmentation
    assert easy.unfamiliar_share < medium.unfamiliar_share < hard.unfamiliar_share
    assert easy.unfamiliar_share == 0.0
    assert easy.fragmentation >= 1.0


def test_hard_text_pieces_are_fragmented(real):
    for word in ("backpropagation", "eigenvalue", "hyperparameter", "decryption", "malleability"):
        assert len(real.tokenizer.tokenize(word)) >= MIN_PIECES_UNFAMILIAR


def test_domain_share_is_higher_for_technical_text(real):
    assert stats(real, CS).domain_share > stats(real, EASY).domain_share
    assert stats(real, EASY).domain_share == 0.0


def test_analysis_is_repeatable(real):
    assert real.analyze(HARD) == real.analyze(HARD)


def test_case_does_not_matter(real):
    assert real.analyze(HARD.upper()) == real.analyze(HARD)


# ------------------------------------------------------------------ default analyzer loading
def counting_loader(monkeypatch, delay: float = 0.0):
    calls: list[int] = []

    def fake(cls):
        calls.append(1)
        if delay:
            time.sleep(delay)
        return VocabularyAnalyzer(WordPieceTokenizer([UNK, "a"]), frozenset({"x", "y"}))

    monkeypatch.setattr(VocabularyAnalyzer, "from_resources", classmethod(fake))
    return calls


def records(caplog, level: str):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelname == level]


def test_default_analyzer_loads_once(monkeypatch, caplog):
    caplog.set_level("DEBUG", logger="edusync_c2")
    calls = counting_loader(monkeypatch)
    first = get_default_analyzer()
    assert first is not None
    assert get_default_analyzer() is first
    assert get_default_analyzer() is first
    assert len(calls) == 1
    info = records(caplog, "INFO")
    assert len(info) == 1
    assert re.fullmatch(r"vocab_loaded words=2 lexicon=2 sha=[0-9a-f]{8}", info[0])
    assert not records(caplog, "WARNING")


def test_real_load_logs_sizes_and_hash(caplog):
    caplog.set_level("INFO", logger="edusync_c2")
    analyzer = get_default_analyzer()
    assert analyzer is not None
    expected = hashlib.sha256(
        resources.files("edusync_c2.difficulty")
        .joinpath("resources", "bert_uncased_vocab.txt")
        .read_bytes()
    ).hexdigest()[:8]
    assert expected == "07eced37"
    assert records(caplog, "INFO") == [
        f"vocab_loaded words=30522 lexicon={len(analyzer.lexicon)} sha={expected}"
    ]
    assert analyzer.tokenizer.vocab_size == 30522


def test_reset_forces_a_new_load(monkeypatch):
    calls = counting_loader(monkeypatch)
    first = get_default_analyzer()
    reset_default_analyzer()
    second = get_default_analyzer()
    assert len(calls) == 2
    assert first is not second


def test_first_load_is_race_free(monkeypatch, caplog):
    caplog.set_level("INFO", logger="edusync_c2")
    calls = counting_loader(monkeypatch, delay=0.05)
    barrier = threading.Barrier(16)

    def work(_: int):
        barrier.wait()
        return get_default_analyzer()

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(work, range(16)))
    assert len(calls) == 1
    assert results[0] is not None
    assert all(r is results[0] for r in results)
    assert len(records(caplog, "INFO")) == 1


def test_waiting_thread_reuses_the_result_loaded_meanwhile(monkeypatch):
    calls = counting_loader(monkeypatch)
    winner = fake_analyzer()

    class LoadsWhileWaiting:
        def __enter__(self):
            vocab._cache.analyzer = winner
            vocab._cache.loaded = True

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(vocab, "_lock", LoadsWhileWaiting())
    assert get_default_analyzer() is winner
    assert not calls


FAILURES = [
    LexiconError("cannot read E:/secret/path/domain_terms.txt"),
    FileNotFoundError("E:/secret/path/bert_uncased_vocab.txt"),
    OSError("disk"),
    UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ValueError("bad"),
]


@pytest.mark.parametrize("error", FAILURES, ids=[type(e).__name__ for e in FAILURES])
def test_failure_is_cached_and_logged_once(monkeypatch, caplog, error):
    caplog.set_level("DEBUG", logger="edusync_c2")
    calls: list[int] = []

    def boom(cls):
        calls.append(1)
        raise error

    monkeypatch.setattr(VocabularyAnalyzer, "from_resources", classmethod(boom))
    assert [get_default_analyzer() for _ in range(4)] == [None] * 4
    assert len(calls) == 1
    assert records(caplog, "WARNING") == [f"vocab_unavailable reason={type(error).__name__}"]
    assert not records(caplog, "INFO")
    assert all("secret" not in r.getMessage() for r in caplog.records)


def test_reset_retries_after_a_failure(monkeypatch):
    def boom(cls):
        raise OSError("down")

    monkeypatch.setattr(VocabularyAnalyzer, "from_resources", classmethod(boom))
    assert get_default_analyzer() is None
    monkeypatch.undo()
    reset_default_analyzer()
    assert get_default_analyzer() is not None


def test_hash_failure_is_also_unavailable(monkeypatch, caplog):
    caplog.set_level("INFO", logger="edusync_c2")
    counting_loader(monkeypatch)

    def boom():
        raise PermissionError("no")

    monkeypatch.setattr(vocab, "_vocab_sha", boom)
    assert get_default_analyzer() is None
    assert records(caplog, "WARNING") == ["vocab_unavailable reason=PermissionError"]
    assert not records(caplog, "INFO")


def test_default_analyzer_is_usable_from_many_threads():
    results = []

    def work():
        a = get_default_analyzer()
        results.append(a.analyze(HARD) if a else None)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8
    assert all(r == results[0] for r in results)
    assert results[0] is not None


def test_logger_is_namespaced():
    assert vocab.logger.name == LOGGER
    assert isinstance(vocab.logger, logging.Logger)


# ------------------------------------------------------------------ determinism across processes
CHILD = """
import json, sys
from dataclasses import asdict
from edusync_c2.difficulty.features.vocab import VocabularyAnalyzer
print(json.dumps(asdict(VocabularyAnalyzer.from_resources().analyze(sys.argv[1]))))
"""


def run_child(seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR), *sys.path])
    done = subprocess.run(
        [sys.executable, "-c", CHILD, HARD],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return done.stdout.strip()


def test_same_result_under_different_hash_seeds(real):
    first, second = run_child("0"), run_child("1")
    assert first == second
    assert json.loads(first) == asdict(stats(real, HARD))
