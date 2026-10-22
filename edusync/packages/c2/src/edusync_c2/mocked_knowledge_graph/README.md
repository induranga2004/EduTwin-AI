# Mocked knowledge graphs (demo lectures)

Hand-curated, mocked knowledge graphs for the demo lectures of the ISSA module (Lectures 01, 03, 04,
05, 06 and 07). They stand in for the real C1 output while C1 still emits the v0 format.

- Format: C1 contract v1 (`nodes` with ids, labels and frequencies; `prerequisite` edges from
  prerequisite to dependent concept with a confidence of 0.9 for `curriculum_defined` and 0.7 for
  `inferred`). Each file has `curated: true` and a `review_status` saying it is a hand-curated demo
  graph that the team should review.
- Non-concept slides (title, agenda, topics, summary, questions, end of lecture, housekeeping) were
  dropped; the remaining nodes are the lecture concepts.
- Matching: a graph is picked for an uploaded deck by lecture name or number. The file name is the
  lecture name, `Lecture NN - <title>.graph.json` (ASCII hyphen, no copy suffix such as `(2)`).
- Adding a new one: copy an existing file, keep the node ids from the deck's slide titles, write the
  prerequisite edges by hand (no cycles), set `curated` to true, name it after the lecture, then run
  `python packages/c2/scripts/validate_c1_graph.py --file <graph>`; it must report 0 errors and
  0 warnings.
- Limits: these graphs are only as good as their authors. The edges come from the subject matter and
  the lecture order, not from a trained model.
