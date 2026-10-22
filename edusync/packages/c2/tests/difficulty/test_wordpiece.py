"""Pure-Python WordPiece: golden parity with HF tokenizers plus branch tests."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from edusync_c2.difficulty.features import wordpiece
from edusync_c2.difficulty.features.wordpiece import UNK, WordPieceTokenizer

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_wordpiece_golden.py"
GOLDEN = Path(__file__).resolve().parent / "golden" / "wordpiece_pieces.json"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@pytest.fixture(scope="module")
def real() -> WordPieceTokenizer:
    return WordPieceTokenizer.from_resources()


@pytest.fixture
def tiny() -> WordPieceTokenizer:
    return WordPieceTokenizer([UNK, "a", "b", "ab", "##a", "##b", "##c"])


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("make_wordpiece_golden", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


# golden parity


def test_golden_metadata() -> None:
    data = _golden()
    assert data["revision"] == REVISION
    assert len(data["cases"]) >= 50


@pytest.mark.parametrize("index", range(len(_golden()["cases"])))
def test_golden_case(real: WordPieceTokenizer, index: int) -> None:
    case = _golden()["cases"][index]
    assert real.tokenize(case["text"]) == case["tokens"]


@pytest.mark.ml
def test_live_parity_with_hf(real: WordPieceTokenizer, script) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer.from_file(str(script.DEFAULT_TOKENIZER))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    for text in script.SENTENCES:
        expected = tokenizer.encode(text, add_special_tokens=False).tokens
        assert real.tokenize(text) == expected, text


# segmentation


def test_pieces_whole_word_and_continuation(tiny: WordPieceTokenizer) -> None:
    assert tiny.pieces("ab") == ("ab",)
    assert tiny.pieces("abc") == ("ab", "##c")
    assert tiny.pieces("bab") == ("b", "##a", "##b")


def test_pieces_prefers_longest_match(tiny: WordPieceTokenizer) -> None:
    assert tiny.pieces("aab") == ("a", "##a", "##b")
    assert tiny.pieces("ab") != ("a", "##b")


def test_pieces_unsegmentable_is_unk(tiny: WordPieceTokenizer) -> None:
    assert tiny.pieces("x") == (UNK,)
    assert tiny.pieces("abx") == (UNK,)
    assert tiny.pieces("xab") == (UNK,)


def test_pieces_empty_word(tiny: WordPieceTokenizer) -> None:
    assert tiny.pieces("") == ()


def test_pieces_length_limit(tiny: WordPieceTokenizer) -> None:
    assert len(tiny.pieces("a" * 100)) == 100
    assert tiny.pieces("a" * 101) == (UNK,)


def test_pieces_are_cached(tiny: WordPieceTokenizer) -> None:
    first = tiny.pieces("abc")
    assert tiny.pieces("abc") is first
    assert tiny.pieces("") == ()
    assert tiny.pieces("") == ()


def test_cache_is_bounded(monkeypatch: pytest.MonkeyPatch, tiny: WordPieceTokenizer) -> None:
    monkeypatch.setattr(wordpiece, "_CACHE_MAX", 1)
    assert tiny.pieces("ab") == ("ab",)
    assert tiny.pieces("abc") == ("ab", "##c")
    assert tiny.pieces("abc") == ("ab", "##c")
    assert len(tiny._cache) == 1


# normalisation


def test_normalize_lowercases_and_splits_whitespace(tiny: WordPieceTokenizer) -> None:
    assert tiny.normalize("Hello\tWORLD\n x") == ["hello", "world", "x"]


def test_normalize_empty_and_blank(tiny: WordPieceTokenizer) -> None:
    assert tiny.normalize("") == []
    assert tiny.normalize(" \t\n\u00a0\u2003 ") == []


def test_normalize_strips_accents(tiny: WordPieceTokenizer) -> None:
    assert tiny.normalize("Na\u00efve caf\u00e9 Z\u00fcrich") == ["naive", "cafe", "zurich"]


def test_normalize_splits_punctuation(tiny: WordPieceTokenizer) -> None:
    assert tiny.normalize("real-time (O(n))") == ["real", "-", "time", "(", "o", "(", "n", ")", ")"]
    assert tiny.normalize("a_b+c$d") == ["a", "_", "b", "+", "c", "$", "d"]
    assert tiny.normalize("\u201cq\u201d\u2014x") == ["\u201c", "q", "\u201d", "\u2014", "x"]


def test_normalize_pads_cjk(tiny: WordPieceTokenizer) -> None:
    assert tiny.normalize("ab\u65e5\u672cc") == ["ab", "\u65e5", "\u672c", "c"]


def test_normalize_drops_control_and_replacement(tiny: WordPieceTokenizer) -> None:
    assert tiny.normalize("a\x00b\x01c\ufffdd\u200be\u00adf") == ["abcdef"]


def test_normalize_maps_whitespace_kinds_to_space(tiny: WordPieceTokenizer) -> None:
    assert tiny.normalize("a\u00a0b\u3000c\rd") == ["a", "b", "c", "d"]


def test_normalize_unicode_lowering_and_nfd(tiny: WordPieceTokenizer) -> None:
    assert tiny.normalize("\u0130") == ["i"]
    assert tiny.normalize("\uff33\uff31\uff2c") == ["\uff53\uff51\uff4c"]


# real vocabulary


def test_from_resources_loads_full_vocab(real: WordPieceTokenizer) -> None:
    assert len(real._vocab) == 30522
    assert real._vocab[UNK] == 100
    assert real.tokenize("Hello, world") == ["hello", ",", "world"]


def test_real_vocab_unk_and_length(real: WordPieceTokenizer) -> None:
    assert real.pieces("a" * 101) == (UNK,)
    assert len(real.pieces("a" * 100)) >= 1
    assert real.pieces("a" * 100) != (UNK,)
    assert real.tokenize("") == []


def test_from_resources_keeps_entries_without_trailing_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRef:
        def joinpath(self, *parts: str) -> "FakeRef":
            return self

        def read_bytes(self) -> bytes:
            return b"[UNK]\na\n##b"

    monkeypatch.setattr(wordpiece.resources, "files", lambda _pkg: FakeRef())
    assert WordPieceTokenizer.from_resources().tokenize("ab") == ["a", "##b"]


# golden script


def test_script_paths(script) -> None:
    assert script.DEFAULT_OUT == GOLDEN
    assert script.DEFAULT_TOKENIZER.name == "tokenizer.json"
    assert script.REVISION == REVISION


def test_script_help_needs_no_tokenizers(script, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        script.main(["--help"])
    assert exc.value.code == 0
    assert "--tokenizer" in capsys.readouterr().out


def test_script_missing_tokenizer_file(script, tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    assert script.main(["--tokenizer", str(missing), "--out", str(tmp_path / "o.json")]) == 2


def test_script_inputs_are_sane(script) -> None:
    sentences = script.SENTENCES
    assert len(sentences) == len(set(sentences))
    assert {"", "   ", "a" * 99, "a" * 100, "a" * 101} <= set(sentences)
    assert not any("[SEP]" in s or "[MASK]" in s for s in sentences)


def test_script_writes_json_with_fake_tokenizers(
    script, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeEncoding:
        def __init__(self, text: str) -> None:
            self.tokens = text.split()

    class FakeTokenizer:
        def no_padding(self) -> None:
            pass

        def no_truncation(self) -> None:
            pass

        def encode(self, text: str, add_special_tokens: bool) -> FakeEncoding:
            assert add_special_tokens is False
            return FakeEncoding(text)

        @classmethod
        def from_file(cls, path: str) -> "FakeTokenizer":
            return cls()

    module = types.ModuleType("tokenizers")
    module.Tokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "tokenizers", module)
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}", encoding="utf-8")
    out = tmp_path / "sub" / "golden.json"
    assert script.main(["--tokenizer", str(tok), "--out", str(out)]) == 0
    raw = out.read_bytes()
    assert raw.endswith(b"}\n")
    assert b"\r" not in raw
    data = json.loads(raw)
    assert data["revision"] == REVISION
    assert [c["text"] for c in data["cases"]] == script.SENTENCES


@pytest.mark.ml
def test_script_regenerates_committed_golden(script, tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    out = tmp_path / "golden.json"
    assert script.main(["--out", str(out)]) == 0
    assert out.read_bytes() == GOLDEN.read_bytes()


def test_vocab_size_counts_distinct_entries(tiny: WordPieceTokenizer) -> None:
    assert tiny.vocab_size == 7
