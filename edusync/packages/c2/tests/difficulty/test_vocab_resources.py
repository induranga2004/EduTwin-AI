"""The vendored WordPiece vocabulary is byte-exact and carries its licence notice."""

import hashlib
from importlib import resources

import pytest

VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _read(name: str) -> bytes:
    ref = resources.files("edusync_c2.difficulty").joinpath("resources", name)
    return ref.read_bytes()


@pytest.fixture(scope="module")
def vocab_bytes() -> bytes:
    return _read("bert_uncased_vocab.txt")


def test_vocab_hash_is_pinned(vocab_bytes: bytes) -> None:
    normalised = vocab_bytes.replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalised).hexdigest() == VOCAB_SHA256


def test_vocab_entries(vocab_bytes: bytes) -> None:
    entries = vocab_bytes.decode("utf-8").replace("\r\n", "\n").split("\n")
    assert entries[-1] == ""
    entries = entries[:-1]
    assert len(entries) == 30522
    assert len(set(entries)) == len(entries)
    assert {"[UNK]", "[PAD]", "##s"} <= set(entries)


def test_notice_names_source_and_licence() -> None:
    notice = _read("NOTICE").decode("utf-8")
    assert REVISION in notice
    assert VOCAB_SHA256 in notice
    assert "Apache" in notice
    assert "sentence-transformers/all-MiniLM-L6-v2" in notice


def test_licence_text_is_apache_2() -> None:
    text = _read("LICENSE-APACHE-2.0.txt").decode("utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
