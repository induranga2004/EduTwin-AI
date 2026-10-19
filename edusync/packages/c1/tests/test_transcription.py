import builtins
import sys

import pytest

from edusync_c1 import transcription
from edusync_c1.transcription import WhisperTranscriber, get_transcriber
from edusync_core.settings import CoreSettings


def test_fake_returns_segments(fake_transcriber, tmp_path):
    assert fake_transcriber.transcribe(tmp_path / "a.wav", "en")["segments"]


def test_whisper_does_not_import_faster_whisper_eagerly():
    sys.modules.pop("faster_whisper", None)
    transcriber = WhisperTranscriber("tiny", "cpu", "int8")
    assert transcriber._model is None
    assert "faster_whisper" not in sys.modules


def test_missing_and_empty_recordings(tmp_path):
    transcriber = WhisperTranscriber()
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(tmp_path / "missing.wav")
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        transcriber.transcribe(empty)


def test_lazy_import_error_when_package_absent(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("faster_whisper not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"data")
    with pytest.raises(ImportError):
        WhisperTranscriber().transcribe(audio)


def test_get_transcriber_reads_settings_and_caches(monkeypatch):
    transcription._whisper.cache_clear()
    settings = CoreSettings(whisper_model="tiny", whisper_device="cpu", whisper_compute_type="int8")
    monkeypatch.setattr(transcription, "get_settings", lambda: settings)
    first = get_transcriber()
    assert isinstance(first, WhisperTranscriber)
    assert first.model_size == "tiny"
    assert get_transcriber() is first
    transcription._whisper.cache_clear()
