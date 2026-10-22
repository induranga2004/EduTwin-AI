# C2 difficulty labels: collection protocol (Phase 3 training data)

Companion to [c2-difficulty.md](c2-difficulty.md) and [c2-model-and-blueprint.md](c2-model-and-blueprint.md).
Code: `edusync/packages/c2/scripts/{export_label_sheet,validate_labels,write_feature_schema}.py`, template
`edusync/packages/c2/data/labels/labels_template.csv`.

Nothing in this document has been run on real lecturer data. The sample sizes below are a **plan**, not a measurement,
and the agreement bar is a **heuristic**, not a validated threshold.

## 1. Purpose and unit

The Phase 3 model is trained on lecturer labels (Low / Medium / High). Until real labels exist, the only labels in the
repository are the circular synthetic ones in `data/synthetic/labels.csv` (a rule over the same properties the decks
expose), which exercise the plumbing and must never be reported as accuracy.

- **Unit of labelling:** one concept in one lecture, using the concept id and label exactly as C2 emits them (the sheet
  exporter writes them).
- **Question asked:** "How hard will this concept be for the students named in `student_level`, *as presented in this
  lecture*, before they have seen it?" It is a pre-lecture judgement, the same moment the model predicts for.

## 2. Student level

A label is meaningless without the audience it is for. The recommended default is **the typical enrolled student
before the lecture**: someone who passed the course prerequisites and has followed the course so far, but has not seen
this lecture. Record it for every row in `student_level`:

`first_year`, `second_year`, `third_year`, `postgraduate`, `mixed`.

Use `mixed` only when the course genuinely has no typical year. Do not change the level within a lecture, and do not
label for the strongest or the weakest student.

## 3. Scale and anchors

| Label | Anchor | Example (illustrative) |
|---|---|---|
| `Low` | Most of the target students will follow it on first exposure from the slides alone. Few new terms, little or no reliance on concepts not yet taught. | "Loss function" in a course where regression has just been introduced and the slide gives a plain definition and one worked example. |
| `Medium` | Needs the lecturer's explanation or some prior material; a typical student will need one pass of effort or one clarifying question. | "Gradient descent" with a learning-rate formula, building on the loss function. |
| `High` | Many students will struggle even with the explanation: several prerequisite concepts, dense notation, or a large conceptual jump. | "Backpropagation" shown as a chain-rule derivation over a multi-layer network. |
| `CannotJudge` | See below. Not a difficulty level. | |

Judge the concept as taught on its slides, not the subject in general. Compare within the course, not against other
courses.

**CannotJudge.** Use it when the rater cannot give an honest label: the slides give too little to tell what is taught
(a title-only slide), the concept is outside the rater's expertise, or the extracted concept is not really a concept
(a heading such as "Agenda" that slipped through). Do not use it for "I am unsure between two levels": pick the closer
one and lower `confidence`. `CannotJudge` rows are excluded from training and from agreement, and their share is
reported by the validator (a high share points to a bad concept list or an unclear rubric).

**Merged concepts.** C2 may merge aliases into one canonical concept (`merged_from` in the response). Label the
canonical concept that the sheet shows, as it appears across its slides. Do not add rows for the merged aliases.

**Confidence** (optional, integer 1 to 5): how sure the rater is of the label (1 guess, 5 certain). It is not part of
the label and is not averaged into it.

## 4. Raters, agreement and adjudication

- At least **two raters on at least 30% of lectures**, chosen before labelling starts (by lecture id, not by how hard
  the lecture looks). Raters label independently and do not see each other's sheets.
- **Agreement** is measured with the quadratic weighted kappa (Low=0, Medium=1, High=2) between two raters, with
  `CannotJudge` pairs excluded, plus the exact agreement rate. Both come from `validate_labels.py --agreement`, which
  uses the first two raters (sorted by `rater_id`) of each lecture and pools the pairs. It is a diagnostic.
- **Go/no-go heuristic:** pooled quadratic weighted kappa of **0.6 or higher** before large-scale labelling or
  training. The bar is a common rule of thumb for "substantial" agreement, not a validated requirement for this task.
  Below it, revise the rubric (a new `rubric_version`), calibrate on a few lectures together and re-label; do not
  train on labels that the raters themselves do not reproduce.
- **Disagreements** (labels that differ for the same concept) are resolved by a third rater or by discussion. Record
  both the original labels and the adjudicated one: keep the raters' rows as they are and add the adjudicated label as
  a row with `rater_id` `ADJ` (ignored by the agreement calculation) and a note naming how it was settled (`third rater` or
  `discussion`). The training package chooses which rows to use; the original rows are never overwritten.
- Report the agreement in the Phase 3 evaluation next to the model metrics: a model cannot be expected to beat the
  agreement between humans.

## 5. Sample-size plan

Plan (to be revised after a pilot, not an estimate of what is enough):

- At least **30 lectures from at least 3 courses**, roughly **300 to 600 labelled concepts** in total (about 10 to 20
  concepts per lecture).
- Start with a **pilot of about 5 lectures** labelled by two raters. Use it to check the rubric, the agreement and the
  time per lecture, then revise this plan.
- Aim for all three levels to be present. If the pilot shows a very skewed distribution or a high `CannotJudge` share,
  say so in the evaluation rather than rebalancing silently.

## 6. Splits

- **Split by lecture, never by concept**, so a lecture is not in both train and test. When there are enough courses,
  also hold out **whole courses** for part of the test set.
- Make the **sealed test split once, with a seed**, before any model work, and record it in a file
  (`data/labels/real/split.json`: the seed, the lecture ids per split and the date). Nobody tunes on the sealed split;
  it is read once for the final report.
- Illustrative stdlib recipe: sort the lecture ids (per course), `random.Random(seed).shuffle(...)`, and take the
  first share as the test split. The Phase 3 training package owns the final implementation.

## 7. Privacy and consent

- Obtain the **lecturer's consent** to use the lecture file for labelling and training before exporting anything.
  Record that consent outside the repository.
- Use **pseudonymous ids**: lectures `L001`, `L002`, ...; raters `R01`, `R02`, ... (and `ADJ` for adjudication). Do not
  put course names, lecturer names or file names in ids, notes or commit messages. `export_label_sheet.py` rejects a
  lecture id with spaces or path separators and `validate_labels.py` warns about ids that are not short tokens.
- The sheet contains concept ids, concept labels and slide numbers only, **never slide text**. Concept labels are
  still derived from the lecture, so treat a sheet as private.
- **Labels, sheets and lecture files are never committed.** Store them under
  `edusync/packages/c2/data/labels/real/`, which is git-ignored (`data/labels/.gitignore`; `*_label_sheet.csv` is
  ignored too). Only `labels_template.csv`, with synthetic example rows, is tracked.
- **Retention and withdrawal:** keep the files only for the length of the project, on an access-controlled drive. A
  lecturer can withdraw a lecture at any time by asking the project owner, who deletes its files and rows and
  retrains or re-splits if the lecture was already used. State the retention period in the consent record.

## 8. Workflow

Run from `edusync/` with the workspace environment active (paths below are relative to `packages/c2`).

1. **Export a sheet** for each lecture (the pipeline runs with its default setup, offline). Without `--graph` the
   concepts come from slide titles (the mock graph, PPTX only); with a C1 graph JSON they are C1's concepts.

       python scripts/export_label_sheet.py --file <lecture.pptx> --lecture-id L001 \
           --graph <c1_graph.json> --out data/labels/real/L001_sheet.csv

   Add `--rubric-version r2` after a rubric change. The command refuses to overwrite an existing file unless `--force`.
2. **Copy the sheet** once per rater and fill, per row: `difficulty_label`, `rater_id`, `student_level`, `confidence`
   (optional), `notes` (optional, no slide text), `labelled_at` (ISO date, `YYYY-MM-DD`). Leave the other columns.
3. **Validate** each filled file, and against its exported sheet so no concept id is invented or mistyped:

       python scripts/validate_labels.py --file data/labels/real/L001_R01.csv \
           --sheet data/labels/real/L001_sheet.csv

   Exit code 0 means ok (warnings allowed), 1 means findings, 2 means unreadable input. `--json` prints a
   machine-readable report. Rules `L-001` to `L-014` are listed in the script and below.
4. **Check agreement** on the lectures with two raters, on the concatenated file of both raters (one header, same
   columns):

       python scripts/validate_labels.py --file data/labels/real/all_labels.csv --agreement

5. **Adjudicate** disagreements (section 4), then re-validate.
6. At training time, the Phase 3 package writes the feature schema into the model folder (section 10).

### Validator rules

| Rule | Severity | Meaning |
|---|---|---|
| L-001 | error | A required column is missing |
| L-002 | error | `lecture_id` or `concept_id` is empty |
| L-003 | error | `difficulty_label` is not exactly `Low`, `Medium`, `High` or `CannotJudge` |
| L-004 | error | `rater_id` is empty on a labelled row |
| L-005 | error | `student_level` is not one of the five values |
| L-006 | error | `confidence` is not an integer 1 to 5 (empty is allowed) |
| L-007 | error | `labelled_at` is not an ISO date or datetime |
| L-008 | error | `rubric_version` is empty or differs between rows of one file |
| L-009 | error | `synthetic` is not `true` or `false` |
| L-010 | error | Duplicate `(lecture_id, concept_id, rater_id)` |
| L-011 | error | With `--sheet`: the concept is not in the exported sheet |
| L-012 | warning | Unlabelled rows remain (empty `difficulty_label`) |
| L-013 | warning | `synthetic=true` and `synthetic=false` rows are mixed |
| L-014 | warning | An id is not a short pseudonymous token |

## 9. CSV schema

The label file extends the synthetic `labels.csv` columns (`lecture_id`, `concept_id`, `concept_slug`,
`concept_label`, `difficulty_label`, `synthetic`) with the rater fields. One row per concept per rater.

| Column | Filled by | Value |
|---|---|---|
| `lecture_id` | exporter | Pseudonymous id such as `L001` |
| `concept_id` | exporter | The id C2 emits (C1 slug for a v1 graph, casefolded label for the mock) |
| `concept_slug` | exporter | Hyphen slug of the label |
| `concept_label` | exporter | The canonical concept label |
| `slide_indexes` | exporter | Slide numbers as C2 reports them, joined by `;` |
| `difficulty_label` | rater | `Low`, `Medium`, `High` or `CannotJudge` (case-sensitive) |
| `rater_id` | rater | Pseudonymous id such as `R01` |
| `student_level` | rater | `first_year`, `second_year`, `third_year`, `postgraduate`, `mixed` |
| `confidence` | rater | Integer 1 to 5, or empty |
| `notes` | rater | Free text, no slide text |
| `labelled_at` | rater | ISO date |
| `rubric_version` | exporter, kept by the rater | `r1` by default; one value per file |
| `synthetic` | exporter | `false` for real labels; `true` only for template or generated rows |

## 10. Connection to the model artifact and the Phase 3 checklist

The model folder must carry `feature_schema.json` so the loader can detect a registry change since training
([model artifact spec](c2-model-and-blueprint.md#4-artifact-spec-and-phase-3-hand-off)). Write it from the registry at
training time and check it afterwards:

    python scripts/write_feature_schema.py --out <model_dir>/feature_schema.json
    python scripts/write_feature_schema.py --check <model_dir>/feature_schema.json

The file holds `schema_version`, `names` (registry order) and `hash`; `--check` exits 1 and names the differing
field. A trained model is valid only for the schema it was trained on: any later change to the feature list needs
retraining.

Hand-off checklist (labels part): real labels collected and validated; agreement at or above the bar on the doubly
labelled lectures; sealed split recorded; `feature_schema.json` written from the same registry the features were
extracted with; the evaluation reports agreement next to model metrics; `ModelKind` moves from `TODO` to `TRAINED`
only once those hold.

## 11. Limits

- The rubric, the anchors and the 0.6 bar are untested proposals. The pilot decides whether they work.
- Synthetic labels are circular by construction. A model scored on them says nothing about accuracy and must never be
  reported as such; they exist only to test the pipeline.
- Labels reflect a few raters at a few institutions. They do not measure real student comprehension (that comes later,
  from live data in Phase 5) and may not transfer to other courses.
- Agreement from two raters on 30% of the lectures is a rough estimate with wide uncertainty at these sizes.
- The exporter lists the concepts C2 finds. Concepts that C2 misses, or that are wrongly extracted, are not labelled
  (use `CannotJudge` for wrong ones) and bias the data toward what the extractor sees.
