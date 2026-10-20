# C1 -> C2 graph contract

Written for the C1 developer. It states what C2's pre-lecture difficulty pipeline (see [c2-difficulty.md](c2-difficulty.md))
expects from C1's concept graph, how C2 uses each field, and what is still open. C2 never imports C1 code: the graph
reaches C2 as JSON.

## 1. How C2 receives the graph

- C2 accepts the response of C1's `POST /api/c1/lectures/graph` **unchanged**. A caller posts the lecture file to
  `POST /api/c2/difficulty` and puts that JSON, as C1 returned it, into the optional form field `c1_graph`.
- The schema C2 validates against is `C1GraphResponse` in `edusync/packages/c2/src/edusync_c2/difficulty/schemas.py`.
  Extra fields are allowed and kept, so C1 can add fields without breaking C2. Invalid JSON or a shape mismatch gives
  HTTP 422 with the failing field.
- The file posted to C2 must be the same lecture file C1 processed, because C2 locates each node on the slides by label.
- Without `c1_graph`, C2 builds a clearly labelled **mock** graph from slide titles and marks every value derived from it
  as `mocked`. The planned `C1ApiGraphProvider` (C2 calling C1 directly) is not implemented.

## 2. Top-level shape (both versions)

| Field | Type | Used by C2 |
|-------|------|------------|
| `lecture_id` | string or null | Echoed in C2's response |
| `filename`, `page_count`, `concept_count` | optional | Not used for computation |
| `concepts` | list of `{name, frequency}` | Fallback concept list when `graph.nodes` is empty (id = casefolded, whitespace-collapsed name) |
| `graph.nodes` | list | The concept list (preferred) |
| `graph.edges` | list | Relationships |
| `graph.note`, `message` | optional text | Not used |

## 3. v0: today's dummy sample

v0 is what C1's graph builder produces now (`edusync/packages/c1/src/edusync_c1/graph.py`, schemas in `schemas.py`):

- Node: `id`, `label`, `frequency`. The id is the casefolded concept name, so it is **per lecture** and not stable across
  lectures.
- Edge: `source`, `target`, `relation_type` = `co_mentioned` only, `evidence_count`, `evidence` = list of
  `{page, sentence}`.
- `lecture_id` is a new random id on every call (`uuid4`), so it cannot identify a lecture across calls.

With v0, C2 has **no prerequisite edges**: all five group B features are `unavailable` (note "C1 graph has no
prerequisite edges") and the response warns "C1 graph has no prerequisite edges: group B features unavailable". The
remaining features (language, explanation, density, `dens.concept_frequency`) still compute.

## 4. v1: the contract described by the C1 developer

> Unverified: the v1 fields below and the sample file `edusync/packages/c1/data/samples/sample_lecture_graph.json` were
> described by the C1 developer but are **not on GitHub yet**. The file does not exist in this repository at the time of
> writing. C2 has only tested v1 against its own synthetic graphs (`build_c1_graph(spec, "v1")` in
> `edusync_c2/difficulty/synthetic.py`). Please publish the sample so C2 can add a contract test against the real shape.

Nodes:

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Unique lowercase slug |
| `label` | string | Display name; C2 searches the slides for it |
| `description` | string | Short definition; goes into the text C2 would give a text encoder |
| `frequency` | int | Mentions in the lecture; becomes `dens.concept_frequency` |

Edges:

| Field | Type | Meaning |
|-------|------|---------|
| `source` | string | Node id of the **prerequisite** concept |
| `target` | string | Node id of the **dependent** concept (the one that needs the prerequisite) |
| `relation_type` | `prerequisite` or `co_mentioned` | Kind of relationship |
| `direction` | `source_is_prerequisite_of_target` | Fixed value; makes the direction explicit |
| `confidence` | number 0.0 - 1.0 | How sure C1 is about the edge (C2 rejects values outside the range) |
| `source_type` | `curriculum_defined` or `inferred` | Whether the edge comes from the curriculum or was inferred from the lecture |
| `evidence` | list of `{page, sentence}` | Where the edge was seen |

Example edge:

```json
{
  "source": "hash function",
  "target": "digital signature",
  "relation_type": "prerequisite",
  "direction": "source_is_prerequisite_of_target",
  "confidence": 0.82,
  "source_type": "inferred",
  "evidence_count": 1,
  "evidence": [{"page": 5, "sentence": "A digital signature is built on a hash function."}]
}
```

## 5. Exactly how C2 uses the graph

Implemented in `edusync_c2/difficulty/graph.py` (`GraphView`) and `features/prerequisites.py`.

Edge validity:

- Only edges with `relation_type == "prerequisite"` can feed the prerequisite features. `co_mentioned` edges (and any
  other relation type) are **never** prerequisites.
- A prerequisite edge is ignored, with a warning in the response, when: its `direction` is set to anything other than
  `source_is_prerequisite_of_target` (a missing `direction` is accepted); `source` or `target` is not a node id in
  `graph.nodes`; or `source == target` (self-loop).
- `confidence` and `source_type` are optional; missing values are skipped in the averages below.

Metrics per concept (the concept's id is the node id; "prerequisites of X" = valid edges with `target == X`):

| Feature | Definition |
|---------|------------|
| `prereq.direct_count` | Number of valid direct prerequisite edges into the concept |
| `prereq.chain_depth` | Longest chain of prerequisites below the concept, counted in edges (a concept with no prerequisites has depth 0) |
| `prereq.transitive_count` | Number of distinct concepts reachable by following prerequisite edges backwards (the concept itself excluded) |
| `prereq.mean_confidence` | Mean `confidence` over direct prerequisite edges that carry one; unavailable if there are no direct prerequisites or no confidences |
| `prereq.curriculum_defined_share` | Share of direct prerequisite edges whose `source_type == "curriculum_defined"`; unavailable if there are no direct prerequisites |

Cycles: the chain-depth search does not follow an edge back into a node already on the current path, so cycles cannot
loop forever; the metric then reflects the acyclic part. `PrerequisiteMetrics.has_cycle` records that a cycle was met,
but C2 does not currently turn it into a warning. Transitive count is a reachability set and is unaffected.

Other fields used: `dens.concept_frequency` = node `frequency`; the node `description` is added to the model text.

Not used so far: `evidence` content, `evidence_count`, `page_count`.

Status rules in the response: a graph supplied by the caller is `real` (C2 does not verify it) and group B is
`computed`; if no valid prerequisite edge exists, group B is `unavailable`.

## 6. What C2 needs that is not yet verified

- The v1 field names and values above, against a real C1 output (the sample file, see the note in section 4).
- That `id` values are lowercase slugs and that `source`/`target` always reference existing node ids.
- That `label` text appears on the slides in a form C2's whole-phrase, case-insensitive match can find. Nodes that are
  not found are reported as `not_found_in_materials` and their slide-based features are `unavailable`.
- How `confidence` is produced and whether it is comparable across edges and lectures.

## 7. Open questions for C1

1. **Stable ids across lectures.** v0 ids are casefolded labels per lecture; a stable id (and lecture id) is needed to
   compare a concept across lectures and to join predictions with later live data.
2. **Aliases.** Abbreviations such as TDE = Transparent Data Encryption should be one node (or carry an `aliases` list).
   C2's concept filter is a mock that keeps everything until this exists.
3. **Generic concepts.** Phrases such as "Best Practices" are not teachable concepts. Should C1 filter them, flag them, or
   leave it to C2?
4. **Pages per node.** Which pages mention each node (`pages: [int]`)? C2 currently re-derives this by text search, which
   can disagree with C1.
5. **Curriculum-level graph.** Is there a graph over the whole curriculum (not per lecture) from which
   `curriculum_defined` edges come? Prerequisites that are not in the current lecture cannot be nodes of a per-lecture
   graph.
6. **Direction and relation vocabulary.** Will any `direction` value other than `source_is_prerequisite_of_target`
   exist? C2 ignores them today.

## 8. Contract-test plan (Phase 2)

Goal: fail loudly if C1's real output drifts from what C2 reads.

1. Commit C1's real sample (`sample_lecture_graph.json`) as a fixture once it is on GitHub, and add a C2 test that
   validates it with `C1GraphResponse` and runs `DifficultyPipeline` with it.
2. Assert on the sample: every edge `source`/`target` is a node id; `relation_type` is `prerequisite` or
   `co_mentioned`; prerequisite edges have `direction == "source_is_prerequisite_of_target"`, `confidence` in [0, 1] and
   `source_type` in {`curriculum_defined`, `inferred`}; node ids are unique and lowercase.
3. Assert on C2's side that the sample yields no "ignored prerequisite edge" warnings and that group B features are
   `computed`.
4. Keep the v0 synthetic graph (`lecture_<k>_graph_v0.json`) as a regression check that a graph without prerequisite
   edges yields `unavailable` group B features and the matching warning.
5. When `C1ApiGraphProvider` is built, add a test with a fake C1 response and, in an integration environment, one
   against the running C1.
6. Version the contract: record the C1 version or a `schema_version` field in the fixture so a change is visible.
