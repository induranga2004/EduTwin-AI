"""Speech-to-text: a ``Transcriber`` protocol and a lazy faster-whisper implementation."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from edusync_core.settings import get_settings


class Transcriber(Protocol):
    """Anything that turns a recording into ``{"text", "language", "segments"}``."""

    def transcribe(self, audio_path: Path, language: str = "en") -> dict[str, Any]: ...


class WhisperTranscriber:
    """faster-whisper backed transcriber; the model loads on first use (needs the ``ml`` extra)."""

    def __init__(
        self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def _get_model(self) -> Any:  # pragma: no cover - needs real model weights
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio_path: Path, language: str = "en") -> dict[str, Any]:
        """Return transcript text, detected language and timestamped segments."""
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Recording not found: {path.name}")
        if path.stat().st_size == 0:
            raise ValueError("The recording file is empty.")
        return self._run(path, language)

    def _run(self, path: Path, language: str) -> dict[str, Any]:  # pragma: no cover
        model = self._get_model()
        raw_segments, info = model.transcribe(
            str(path), language=language or None, beam_size=5, vad_filter=True
        )
        segments = []
        for index, seg in enumerate(raw_segments):
            start = max(0.0, float(seg.start))
            segments.append(
                {
                    "id": index,
                    "start": start,
                    "end": max(start, float(seg.end)),
                    "text": (seg.text or "").strip(),
                }
            )
        text = " ".join(s["text"] for s in segments if s["text"]).strip()
        return {
            "text": text,
            "language": getattr(info, "language", language or "unknown"),
            "segments": segments,
        }


@lru_cache
def _whisper(model_size: str, device: str, compute_type: str) -> WhisperTranscriber:
    """One transcriber per configuration so the model is loaded once per process."""
    return WhisperTranscriber(model_size, device, compute_type)


def get_transcriber() -> Transcriber:
    """FastAPI dependency; tests override it with a fake."""
    s = get_settings()
    return _whisper(s.whisper_model, s.whisper_device, s.whisper_compute_type)
