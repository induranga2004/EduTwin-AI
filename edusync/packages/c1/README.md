# EduSync C1: Lecturer Monitoring

C1 turns a lecture document (PDF or PPTX) and, optionally, a recording of the lecture into:

- extracted page/slide text and a ranked list of candidate **concepts** (a frequency heuristic);
- a **knowledge graph** whose edges mean "mentioned in the same sentence";
- a **timeline** of cumulative graph snapshots, one per transcript segment, showing which
  concepts have been mentioned so far (semantic matching with an exact-phrase fallback);
- core `EvidenceItem`s (`source=SPEECH`) via `CoverageTracker.evidence_items(...)`.

> Everything here is a baseline. Concept extraction is heuristic, semantic matching uses a
> pretrained MiniLM model with an uncalibrated threshold (0.45), and the research classifier is
> trained on a tiny dataset. A match is evidence of a possible *mention*, not of coverage,
> understanding, mastery or a prerequisite relationship. None of it has been validated yet.

## Routes

The app mounts the router under `/api/c1` (`COMPONENT.router`).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/c1/health` | Liveness check |
| POST | `/api/c1/lectures/upload` | Multipart `file` (PDF only): stores it, returns pages and concepts |
| POST | `/api/c1/lectures/graph` | Multipart `file` (PDF or PPTX): returns concepts and the candidate graph |
| POST | `/api/c1/lectures/transcribe` | Multipart `file` (recording) + `document_file` (PDF/PPTX), query `language` (default `en`): returns transcript, graph and graph timeline |

Errors: 400 (bad name/type/signature/empty/encrypted), 413 (over `EDUSYNC_MAX_UPLOAD_MB`;
documents are additionally capped at 100 MB), 422 (no extractable text, unreadable document, no
speech segments or invalid parameters), 500 (unexpected failure, details are only logged).
Files are stored under `<EDUSYNC_DATA_DIR>/c1/{uploads,recordings,processed}`.

## Modules

`api` (routes), `schemas`, `validation`, `documents`, `concepts`, `graph`, `semantic`
(`SemanticMatcher`), `timeline`, `coverage` (`CoverageTracker`), `transcription`
(`Transcriber`, `WhisperTranscriber`, `get_transcriber`), `storage`, `research/`.

## Optional extras

| Extra | Installs | Needed for |
| --- | --- | --- |
| `ml` | faster-whisper | real speech-to-text (`WhisperTranscriber`) |
| `semantic` | sentence-transformers | MiniLM matching (otherwise exact-phrase fallback) |
| `research` | scikit-learn, pandas, joblib | training the baseline coverage classifier |

Whisper settings come from `EDUSYNC_WHISPER_MODEL`, `EDUSYNC_WHISPER_DEVICE` and
`EDUSYNC_WHISPER_COMPUTE_TYPE`. Heavy libraries are imported lazily.

## Research tooling

`research/dataset.py` defines the single label schema and validation (stdlib only) for
human-labelled examples (`save_annotation`, `load_dataset`). Train the baseline classifier with:

```
uv --directory edusync run --extra research python -m edusync_c1.research.train_model \
    --dataset path/to/coverage_examples.csv --model-dir models --report-dir reports
```

It needs at least 20 examples from 3 lectures, and evaluates on held-out lectures.

## Tests

```
uv --directory edusync run pytest packages/c1 -q --cov=edusync_c1 --cov-report=term-missing
```

Tests use in-memory PDFs/PPTX, a fake transcriber and a disabled semantic model; they need no
network or model weights. The end-to-end training test is skipped unless the `research` extra is
installed.
