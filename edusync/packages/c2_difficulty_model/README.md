# edusync-c2-model

Phase 3 (in progress) data layer and trainer for the C2 difficulty models: label-free difficulty
scoring; see the plan. One-way dependency: this package imports `edusync_c2`; C2 never imports it
(a test enforces that). No torch.

The earlier lecturer-label training path (label reader, synthetic expert datasets, experts,
combiner, ensemble and the synthetic demo) was removed.

Modules (`src/edusync_c2_model/`):

- `manifest.py`: the lecture manifest CSV (`lecture_id,course_id,file,graph`).
- `examples.py`: the `FeatureRow` and `Example` records (text, 35 feature values, missing mask).
- `extract.py`: run the frozen `DifficultyPipeline` per lecture and cache the features.
- `split.py`: seeded group split (by lecture or course), written once, plus grouped k-fold.
- `metrics.py`: QWK, MAE, adjacent and balanced accuracy, per-class report, bootstrap CIs.
- `report.py`: markdown report with a banner for synthetic data.
- `models/`: numpy softmax, scaling and embedding-cache helpers.

Local, git-ignored folders: `data/`, `cache/`. Caches derived from real lectures must never be
committed.

Test: `pytest packages/c2_difficulty_model -q --cov=edusync_c2_model` (100% line and branch).
