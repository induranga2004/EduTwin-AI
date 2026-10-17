
from pathlib import Path
from faster_whisper import WhisperModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "processed"


class LectureTranscriber:
    """Transcribe a lecture recording into timestamped segments."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = "en",
    ) -> dict:
        """Return transcript text and timestamped speech segments."""

        path = Path(audio_path)

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(
                f"Recording not found: {path}"
            )

        model = self._load_model()

        segments, info = model.transcribe(
            str(path),
            language=language,
            vad_filter=True,
            beam_size=5,
        )

        results = []
        full_text = []

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            item = {
                "segment_id": len(results) + 1,
                "start_seconds": round(segment.start, 2),
                "end_seconds": round(segment.end, 2),
                "text": text,
            }

            results.append(item)
            full_text.append(text)

        return {
            "filename": path.name,
            "language": info.language,
            "language_probability": round(
                info.language_probability, 4
            ),
            "segment_count": len(results),
            "transcript": " ".join(full_text),
            "segments": results,
        }

    def save_transcript(
        self,
        transcript_data: dict,
        output_path: str | Path | None = None,
    ) -> Path:
        """Save the generated transcript as a JSON file."""

        import json

        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            safe_stem = Path(
                transcript_data["filename"]
            ).stem.replace(" ", "_")

            output_path = (
                TRANSCRIPT_DIR / f"{safe_stem}_transcript.json"
            )

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                transcript_data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return destination
