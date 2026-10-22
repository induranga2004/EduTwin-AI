"""C4 speech helper. Runs INSIDE the edusync container (see scripts/c4-speech.ps1).

Transcribes a lecture recording with C1's Whisper transcriber (faster-whisper, runs locally) and
stores the timestamped segments in C4 through its HTTP API. This is glue code: the C4 package
itself never imports C1. Standard library plus C1 only; ASCII output; no secrets.

  --check              load the speech model and run it on 2 seconds of silence (downloads the
                       model on first use, so run this once with internet before the demo)
  --file PATH          transcribe PATH and store the result for the lecture
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

# Keep downloaded models in the persistent volume so a container rebuild does not lose them.
os.environ.setdefault("HF_HOME", "/data/hf")

C4 = "http://127.0.0.1:8100/api/c4"


def http_json(method, url, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:500]


def fail(message):
    print("ERROR: " + message)
    sys.exit(1)


def ascii_safe(text):
    return text.encode("ascii", "replace").decode("ascii")


def patch_audio_decoding():
    """Work around a library mismatch in this image.

    faster-whisper 1.2.x opens recordings with av.open(metadata_errors=...), which the installed
    PyAV no longer accepts, so C1's transcriber fails on every file. Decode with PyAV directly
    instead (all audio tracks are mixed, so a recording with separate microphone and system
    tracks still works) and let faster-whisper use that.
    """
    import av
    import faster_whisper.audio as audio_module
    import faster_whisper.transcribe as transcribe_module
    import numpy as np

    def decode_audio(input_file, sampling_rate=16000, split_stereo=False):
        tracks = []
        with av.open(str(input_file), mode="r") as container:
            for stream in [s for s in container.streams if s.type == "audio"]:
                resampler = av.audio.resampler.AudioResampler(
                    format="s16", layout="mono", rate=sampling_rate
                )
                parts = []
                for frame in container.decode(stream):
                    parts.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
                parts.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
                if parts:
                    tracks.append(np.concatenate(parts).astype(np.float32) / 32768.0)
        if not tracks:
            raise ValueError("The recording has no audio track.")
        length = max(len(t) for t in tracks)
        mixed = np.zeros(length, dtype=np.float32)
        for track in tracks:
            mixed[: len(track)] += track
        peak = float(np.max(np.abs(mixed))) if length else 0.0
        return mixed / peak if peak > 1.0 else mixed

    audio_module.decode_audio = decode_audio
    transcribe_module.decode_audio = decode_audio


def make_transcriber(model):
    try:
        from edusync_c1.transcription import WhisperTranscriber
        from edusync_core.settings import get_settings
    except ImportError as exc:
        fail("Speech support is not installed in this container: %s" % exc)
    patch_audio_decoding()
    settings = get_settings()
    return WhisperTranscriber(
        model or settings.whisper_model, settings.whisper_device, settings.whisper_compute_type
    )


STOP_WORDS = {
    "about", "above", "across", "after", "again", "against", "always", "because", "before",
    "between", "could", "every", "other", "should", "their", "there", "these", "those",
    "through", "using", "where", "which", "while", "within", "without", "would",
}


def vocabulary_hint(lecture, extra):
    """A short list of the lecture's own terms, so Whisper spells technical words correctly.

    Built from the text of the slides that have annotations, plus any words given by the user.
    Returns None when there is nothing useful.
    """
    words = []
    status, slides = http_json("GET", "%s/lectures/%s/slides" % (C4, lecture))
    if status == 200:
        counts = {}
        for slide in slides:
            for raw in (slide.get("text") or "").replace("\n", " ").split():
                word = "".join(ch for ch in raw if ch.isalpha() or ch == "-").strip("-")
                if len(word) >= 6 and word.lower() not in STOP_WORDS:
                    counts[word] = counts.get(word, 0) + 1
        words = sorted(counts, key=lambda w: (-counts[w], w))[:40]
    for term in (extra or "").split(","):
        if term.strip():
            words.insert(0, term.strip())
    if not words:
        return None
    unique = list({w.lower(): w for w in reversed(words)}.values())[::-1]
    return ("Lecture terms: " + ", ".join(unique))[:400]


def add_prompt(transcriber, prompt):
    """Pass the vocabulary hint to Whisper (C1's transcriber has no option for it)."""
    model = transcriber._get_model()
    original = model.transcribe

    def with_prompt(*args, **kwargs):
        kwargs.setdefault("initial_prompt", prompt)
        return original(*args, **kwargs)

    model.transcribe = with_prompt


def cmd_check(model):
    transcriber = make_transcriber(model)
    print("model: %s (device %s)" % (transcriber.model_size, transcriber.device))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "silence.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000 * 2)
        started = time.time()
        result = transcriber.transcribe(path, "en")
    print(
        "ok: model loaded and ran in %.1f s; the silent test found %d speech segments (expected 0)"
        % (time.time() - started, len(result["segments"]))
    )
    check_mp4_decoding()


def check_mp4_decoding():
    """Make a 1.5 s MP4 with AAC audio (what Windows screen recording produces) and decode it."""
    import av
    import faster_whisper.audio as audio_module
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "tone.mp4")
        with av.open(path, mode="w") as out:
            stream = out.add_stream("aac", rate=16000)
            stream.layout = "mono"
            samples = (np.sin(np.arange(24000) * 2 * np.pi * 440 / 16000) * 8000).astype("int16")
            frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = 16000
            for packet in stream.encode(frame):
                out.mux(packet)
            for packet in stream.encode(None):
                out.mux(packet)
        decoded = audio_module.decode_audio(path)
    seconds = len(decoded) / 16000.0
    print("ok: decoded an MP4/AAC test file, %.1f s of audio (expected about 1.5)" % seconds)


def latest_lecture():
    status, lectures = http_json("GET", C4 + "/lectures")
    if status != 200 or not lectures:
        fail("No lecture is stored in C4 yet. Run scripts\\c4-capture.ps1 during the lecture.")
    return lectures[0]["lecture_id"]


def cmd_transcribe(args):
    path = Path(args.file)
    if not path.is_file():
        fail("Recording not found inside the container: %s" % path)
    lecture = args.lecture or latest_lecture()
    transcriber = make_transcriber(args.model)
    if args.hint or args.vocabulary:
        hint = vocabulary_hint(lecture, args.vocabulary)
        if hint:
            add_prompt(transcriber, hint)
            print("vocabulary hint: " + ascii_safe(hint)[:160])
    print("transcribing %s with model %s (this can take a few minutes)" % (path.name, transcriber.model_size))
    started = time.time()
    result = transcriber.transcribe(path, args.language)
    segments = result["segments"]
    print("done in %.0f s: %d segments, language %s" % (time.time() - started, len(segments), result.get("language")))
    if not segments:
        fail("No speech was found. Check that the recording has sound from your microphone.")

    offset = args.offset
    if offset is None and args.recording_start_epoch is not None:
        # BBB meeting ids end with the meeting's creation time in epoch milliseconds.
        try:
            meeting_start = float(lecture.rsplit("-", 1)[-1]) / 1000.0
        except ValueError:
            fail("Cannot read the meeting start time from the lecture id: " + lecture)
        offset = args.recording_start_epoch - meeting_start
        print("the recording started %.1f s after the meeting was created" % offset)

    body = {
        "segments": [
            {"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in segments
        ],
        "source": args.source or path.name,
    }
    if offset is not None:
        body["offset_seconds"] = offset
    elif args.sync_word:
        body["sync_word"] = args.sync_word
    status, resp = http_json("PUT", "%s/lectures/%s/speech" % (C4, lecture), body)
    if status != 200:
        print("storing the speech FAILED (%s): %s" % (status, ascii_safe(str(resp))))
        sys.exit(1)
    print(
        "stored %d segments for lecture %s; alignment: %s, offset %.1f s"
        % (resp["segment_count"], lecture, resp["method"], resp["offset_seconds"])
    )
    for seg in segments[:5]:
        print("  %6.1fs  %s" % (seg["start"], ascii_safe(seg["text"].strip())[:100]))
    print("report-path: /edusync/api/c4/lectures/%s/report" % lecture)


def cmd_transcribe_bbb(args):
    """Transcribe BigBlueButton's own microphone recordings and store them on the meeting clock.

    BBB names each file microphone-<user>-<track>-<epoch milliseconds>.webm, where the number is
    the moment the track started. The lecture id ends with the meeting's creation time in epoch
    milliseconds, so every segment can be placed on the meeting clock with no manual offset.
    Several files (a microphone muted and unmuted again) are merged in time order.
    """
    lecture = args.lecture or latest_lecture()
    try:
        meeting_start = float(lecture.rsplit("-", 1)[-1]) / 1000.0
    except ValueError:
        fail("Cannot read the meeting start time from the lecture id: " + lecture)
    transcriber = make_transcriber(args.model)
    merged = []
    for name in args.bbb_audio:
        path = Path(name)
        match = re.search(r"-(\d{13})\.[A-Za-z0-9]+$", path.name)
        if not match:
            fail("Cannot read the start time from the audio file name: " + path.name)
        shift = int(match.group(1)) / 1000.0 - meeting_start
        started = time.time()
        result = transcriber.transcribe(path, args.language)
        print(
            "%s: %d segments, track started %.1f s after the meeting began (%.0f s to transcribe)"
            % (ascii_safe(path.name)[:40], len(result["segments"]), shift, time.time() - started)
        )
        for seg in result["segments"]:
            merged.append(
                {"start": seg["start"] + shift, "end": seg["end"] + shift, "text": seg["text"]}
            )
    if not merged:
        fail(
            "No speech was found in the BBB recording. BBB only records a microphone that is "
            "UNMUTED, so check the microphone button while you speak."
        )
    merged.sort(key=lambda s: s["start"])
    body = {
        "segments": [dict(id=i, **seg) for i, seg in enumerate(merged)],
        "offset_seconds": 0.0,
        "source": "BigBlueButton recording (%d audio file(s))" % len(args.bbb_audio),
    }
    status, resp = http_json("PUT", "%s/lectures/%s/speech" % (C4, lecture), body)
    if status != 200:
        print("storing the speech FAILED (%s): %s" % (status, ascii_safe(str(resp))))
        sys.exit(1)
    print("stored %d segments for lecture %s (BBB clock, no offset needed)" % (len(merged), lecture))
    for seg in merged[:6]:
        print("  %6.1fs  %s" % (seg["start"], ascii_safe(seg["text"].strip())[:100]))
    print("report-path: /edusync/api/c4/lectures/%s/report" % lecture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbb-audio", dest="bbb_audio", nargs="+", metavar="FILE")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--file")
    parser.add_argument("--lecture")
    parser.add_argument("--model")
    parser.add_argument("--language", default="en")
    parser.add_argument("--offset", type=float)
    parser.add_argument("--recording-start-epoch", dest="recording_start_epoch", type=float)
    parser.add_argument("--sync-word", dest="sync_word")
    parser.add_argument("--source")
    parser.add_argument("--vocabulary", help="extra terms, comma separated")
    parser.add_argument(
        "--hint", action="store_true", help="give Whisper a word list made from the slides"
    )
    args = parser.parse_args()
    if args.check:
        cmd_check(args.model)
    elif args.bbb_audio:
        cmd_transcribe_bbb(args)
    elif args.file:
        cmd_transcribe(args)
    else:
        parser.error("use --check, --file PATH or --bbb-audio FILE...")


if __name__ == "__main__":
    main()
