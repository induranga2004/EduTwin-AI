from pathlib import Path
from faster_whisper import WhisperModel

MODEL_SIZE = "base"
_whisper_model = None


def get_whisper_model():
    """Load the local Whisper model once per Python process."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
    return _whisper_model


def transcribe(audio_path: str | Path, language: str = "en") -> dict:
    """Return transcript text, detected language, and timestamped segments."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError("The recording file is empty.")

    model = get_whisper_model()
    segments_generator, info = model.transcribe(
        str(path),
        language=language or None,
        beam_size=5,
        vad_filter=True,
    )

    segments = []
    text_parts = []
    for index, segment in enumerate(segments_generator):
        segment_text = (segment.text or "").strip()
        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))
        segments.append({
            "id": index,
            "start": start,
            "end": end,
            "text": segment_text,
        })
        if segment_text:
            text_parts.append(segment_text)

    return {
        "text": " ".join(text_parts).strip(),
        "language": getattr(info, "language", language or "unknown"),
        "segments": segments,
    }
