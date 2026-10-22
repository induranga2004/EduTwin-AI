Synthetic lecture data for the EduSync C2 difficulty pipeline.

- Everything here is synthetic: no real course material is used.
- The difficulty labels in labels.csv come from a transparent rule over the same properties the
  generated decks expose (see synthetic_label in edusync_c2/difficulty/synthetic.py).
- Therefore any model score measured against these labels is circular. Do not use it for
  accuracy claims; the data only exercises the pipeline end to end.

Regenerate (from the repository root, with the workspace environment active):

    python edusync/packages/c2/scripts/make_synthetic_data.py         --seed 42 --lectures 3 --concepts 8

Files per lecture k: lecture_<k>.pptx, lecture_<k>.pdf, lecture_<k>_graph_v0.json,
lecture_<k>_graph_v1.json. Shared: labels.csv, README.txt.
