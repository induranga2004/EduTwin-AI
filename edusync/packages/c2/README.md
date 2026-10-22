# edusync-c2

C2 - Learning Difficulty Prediction + Comprehension Validation (component of EduSync AI).

C2 has two stages:

| Stage | What | State |
|-------|------|-------|
| 1. Pre-lecture difficulty | Lecture file (PPTX/PDF) + optional C1 graph -> features -> model -> Low / Medium / High per concept, before the lecture | Phase 1 pipeline exists, end to end. The model is a **rule baseline, not trained**; some inputs and features are **mocked** or **TODO** and are labelled so in every response |
| 2. Live comprehension | Compare the prediction with poll/quiz results during the lecture | Only a rule-based placeholder (`comprehension.py`); not validated. The C1 -> C2 runtime hand-off is not implemented |

No accuracy is claimed anywhere: there is no real labelled data yet.

## Quickstart

From `edusync/` (run `uv sync` once):

```bash
uv run pytest packages/c2                      # tests
uv run python -m edusync_api                   # API on 127.0.0.1:8100
curl -F "file=@lecture.pptx" http://localhost:8100/api/c2/difficulty
curl -F "file=@lecture.pptx" -F "c1_graph=<graph.json" http://localhost:8100/api/c2/difficulty
uv run python packages/c2/scripts/make_synthetic_data.py   # synthetic decks, graphs, labels
```

Without `c1_graph` the graph is a mock built from slide titles (labelled `mock` in the response).

## Endpoints (mounted under `/api/c2`)

| Route | Description |
|-------|-------------|
| `POST /difficulty` | multipart: `file` (.pptx/.pdf), optional `c1_graph` (C1's JSON as returned). Errors: 400 wrong type, corrupt or encrypted; 413 too large; 422 no text or invalid `c1_graph` |
| `GET /difficulty/info` | model, graph provider, concept filter, the 26 features, TODO list |
| `GET /health` | component health |
| `POST /comprehension/evaluate` | rule-based poll flagging (later stage, unchanged) |

The old `{text}` body of `/difficulty` no longer works.

## Layout

```
packages/c2/
  pyproject.toml
  scripts/make_synthetic_data.py      # CLI: seeded synthetic decks, C1 graphs, labels
  scripts/validate_c1_graph.py        # CLI: check a C1 graph file against the contract
  data/synthetic/                     # generated; .pptx/.pdf git-ignored; labels are circular
  src/edusync_c2/
    router.py                         # HTTP routes
    comprehension.py, schemas.py      # later live stage (rule-based placeholder)
    difficulty/
      materials.py                    # PPTX/PDF parser
      graph.py, providers.py          # C1 graph view, payload/mock/TODO providers
      concepts.py                     # concepts on slides, mock filter
      registry.py                     # 26 features, PHASE1_TODO
      features/                       # groups A-D; vocab.py, wordpiece.py, lexicon.py for the vocabulary features
      resources/                      # vendored vocabulary + domain lexicon (see Vendored resources)
      fusion.py, models.py            # ordered features + mask; RuleBaselineModel
      pipeline.py, schemas.py         # orchestration; contracts
      synthetic.py                    # synthetic data helpers
  tests/                              # test_c2.py, difficulty/
```

## Vendored resources

`src/edusync_c2/difficulty/resources/` holds the data of the three vocabulary features of group A (`lang.unfamiliar_vocab_ratio`,
`lang.subword_fragmentation`, `lang.domain_term_ratio`):

- `bert_uncased_vocab.txt`: unmodified copy of the `vocab.txt` of `sentence-transformers/all-MiniLM-L6-v2` (30,522 entries, SHA-256
  `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`), Apache License 2.0. `NOTICE` records the source and revision and
  `LICENSE-APACHE-2.0.txt` is the licence text; keep both with the file. A nested `.gitattributes` forces LF line endings. No model
  weights are included and no dependency was added; the word-piece tokenizer is pure Python.
- `domain_terms.txt`: 851 CS/IT/ML terms written for this project (not copied from a glossary).

If these files cannot be loaded, the three features become `unavailable` and the rest of the response is unaffected.
Method and limits: [design document](../../../docs/edusync/c2-difficulty.md), section 4, "Feature A method".

## Docs

- [C2 difficulty pipeline design](../../../docs/edusync/c2-difficulty.md)
- [C2 model interface, artifact spec and blueprint assessment](../../../docs/edusync/c2-model-and-blueprint.md)
- [C1 -> C2 graph contract](../../../docs/edusync/c1-c2-graph-contract.md)
- [Architecture](../../../docs/edusync/architecture.md)

## Settings

Read from environment variables (prefix `EDUSYNC_C2_`, see `src/edusync_c2/settings.py`).

| Variable | Values | Default | Meaning |
|----------|--------|---------|---------|
| `EDUSYNC_C2_LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR | `INFO` | Level of the `edusync_c2` logger; configured once when the pipeline is built |
| `EDUSYNC_C2_MODEL` | `baseline`, `minilm` | `baseline` | Model used by `/difficulty`; `minilm` answers HTTP 501 until Phase 3 |
| `EDUSYNC_C2_MODEL_DIR` | folder path | unset | Model artifact folder for `minilm` (spec in the model doc above) |

Logging conventions and the checklist for new features: see the "Logging and development conventions" section of the [design document](../../../docs/edusync/c2-difficulty.md).

## Validate a C1 graph

Check a C1 graph JSON file against the contract (exit code 0 ok, 1 contract errors, 2 unreadable file or invalid JSON). From `edusync/`:

```bash
uv run python packages/c2/scripts/validate_c1_graph.py --file path/to/graph.json   # optional: --expected v0|v1, --json
```
