# edusync-c2-model

Phase 3 data layer and (later) trainer for the C2 difficulty model. One-way dependency: this package
imports `edusync_c2`; C2 never imports it (a test enforces that). No torch.

Modules (`src/edusync_c2_model/`):

- `labels.py`: read filled label CSVs and resolve one label per `(lecture_id, concept_id)`.
- `manifest.py`: the lecture manifest CSV (`lecture_id,course_id,file,graph`).
- `examples.py`: the `Example` record (text, 35 feature values, missing mask, label).
- `extract.py`: run the frozen `DifficultyPipeline` per lecture, cache the features, join to labels.
- `split.py`: seeded group split (by lecture or course), written once, plus grouped k-fold.
- `metrics.py`: QWK, MAE, adjacent and balanced accuracy, per-class report, bootstrap CIs.
- `synthetic.py`, `report.py`: plumbing-only synthetic data and a markdown report with a banner.

Local, git-ignored folders: `data/`, `cache/`, `runs/`. Real labels and caches derived from real
lectures must never be committed.

Test: `pytest packages/c2_difficulty_model -q --cov=edusync_c2_model` (100% line and branch).
