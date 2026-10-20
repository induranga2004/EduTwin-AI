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
| `GET /difficulty/info` | model, graph provider, concept filter, the 24 features, TODO list |
| `GET /health` | component health |
| `POST /comprehension/evaluate` | rule-based poll flagging (later stage, unchanged) |

The old `{text}` body of `/difficulty` no longer works.

## Layout

```
packages/c2/
  pyproject.toml
  scripts/make_synthetic_data.py      # CLI: seeded synthetic decks, C1 graphs, labels
  data/synthetic/                     # generated; .pptx/.pdf git-ignored; labels are circular
  src/edusync_c2/
    router.py                         # HTTP routes
    comprehension.py, schemas.py      # later live stage (rule-based placeholder)
    difficulty/
      materials.py                    # PPTX/PDF parser
      graph.py, providers.py          # C1 graph view, payload/mock/TODO providers
      concepts.py                     # concepts on slides, mock filter
      registry.py                     # 24 features, PHASE1_TODO
      features/                       # groups A-D
      fusion.py, models.py            # ordered features + mask; RuleBaselineModel
      pipeline.py, schemas.py         # orchestration; contracts
      synthetic.py                    # synthetic data helpers
  tests/                              # test_c2.py, difficulty/
```

## Docs

- [C2 difficulty pipeline design](../../../docs/edusync/c2-difficulty.md)
- [C1 -> C2 graph contract](../../../docs/edusync/c1-c2-graph-contract.md)
- [Architecture](../../../docs/edusync/architecture.md)
