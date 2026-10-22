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
  as `mocked`. C2 can also call C1 itself or read a graph file, both opt-in; see section 1a.

## 1a. C2 calling C1 itself (`c1_api` provider)

Optional and off by default (`EDUSYNC_C2_GRAPH_PROVIDER=c1_api`; settings in [c2-difficulty.md](c2-difficulty.md), section
"Graph providers"). When no `c1_graph` is posted, C2 sends the uploaded file to C1 instead of building a mock:

- **Request.** `POST {EDUSYNC_C2_C1_BASE_URL}/lectures/graph` (default base `http://127.0.0.1:8100/api/c1`), multipart
  form with one field `file` holding the upload unchanged (content type derived from `.pdf` / `.pptx`). Plain stdlib HTTP;
  C2 imports no C1 code.
- **Response.** The JSON object C1 returns, validated as `C1GraphResponse` (same rules as a posted graph; extra fields kept).
  Anything else is an error: C1 4xx gives HTTP 422 with C1's `detail` (cut to 200 characters), C1 5xx, unreachable, a
  non-JSON body or a shape mismatch gives 502, a timeout gives 504 (`EDUSYNC_C2_C1_TIMEOUT_S`, default 60).
- **`lecture_id`.** C1 returns a fresh uuid on every call. C2 drops it and reports `c1-<first 12 hex digits of
  sha256(upload)>`, so the same file always gives the same C2 response. C1's uuid is not visible in C2's output.
- **v0 from C1.** C1 emits v0 today (19 nodes and 20 `co_mentioned` edges were seen in a Docker check). C2 accepts it, marks
  the graph `real`, and group B is `unavailable` with the warning "C1 graph has no prerequisite edges: group B features
  unavailable". Nothing breaks; the prerequisite features simply wait for v1.
- **Privacy.** C2 logs only the status and a reason code of a failed call, never the file or its name.

### Swapping the sample graph (`file` provider)

`EDUSYNC_C2_GRAPH_PROVIDER=file` serves a graph from disk with no C1 running: by default the bundled synthetic v1 sample
`edusync/packages/c2/src/edusync_c2/difficulty/resources/sample_c1_graph_v1.json` (made by
`packages/c2/scripts/make_sample_graph.py`; not C1 data). To use another graph, for example C1's real v1 sample once it is
published, set `EDUSYNC_C2_GRAPH_FILE=<path>`. The file must be a `C1GraphResponse` (check it first with
`scripts/validate_c1_graph.py`, section 9). It is always labelled `mock` in `inputs[]`, because it is not C1 output for the
uploaded deck; concepts that are not in the deck come back as `not_found_in_materials`. Use a posted `c1_graph` (precedence:
posted graph, then configured provider, then slide-title mock) when the graph really belongs to the file.

A file that people wrote or corrected can be marked `"curated": true` (top level; `"note": "curated"` also works). It is then
labelled `curated` in `inputs[]`: features treat it as real, and the response warns that the graph is hand-made.
`packages/c2/scripts/make_curated_graph.py --deck <pptx|pdf> --out <file>` drafts one from the slide titles
(`"curated": false`, validated against this contract, refuses to overwrite without `--force`); its edges are
mention-based guesses (earlier concept mentioned where a later one is discussed) that a human must correct.

Still pending: C1's real v1 sample. Until it exists, the v1 side of the `c1_api` path has been tested only with C2's own
synthetic graphs behind a fake server.

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
and the pipeline turns it into a warning: a `graph_cycle` WARNING log line (count only), and for each concept whose
prerequisite chain meets a cycle the prediction warning `prerequisite cycle found below this concept; the back edge was ignored`
(`CONCEPT_CYCLE_WARNING` in `pipeline.py`; test `tests/difficulty/test_cycle_warning.py`). Transitive count is a reachability set and is unaffected.

Id convention (`difficulty/ids.py`): v0 graphs, the mock provider and the fallback concept list use **casefold ids** (`casefold_id`: casefolded label, whitespace collapsed, so `Loss  Function` becomes `loss function`). v1 graphs use **hyphen slugs** (`slug_id`: lowercase letters and digits joined by single hyphens, so `TCP/IP` becomes `tcp-ip`; rule C1-018). `providers.slugify` is a historical alias of `casefold_id`, despite its name. The two conventions never mix inside one graph (C1-021).

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
   *C2's current answer:* C2 does not require an alias field. It merges aliases itself in its optional heuristic concept
   filter (`EDUSYNC_C2_CONCEPT_FILTER=heuristic`, see "Concept filter" in [c2-difficulty.md](c2-difficulty.md); the default is
   still a mock that keeps everything). If C1 later emits `aliases`, a `description` or merge decisions, C2 can use them and
   shrink its own rules; C1-side merging would be more reliable than C2's string rules.
3. **Generic concepts.** Phrases such as "Best Practices" are not teachable concepts. Should C1 filter them, flag them, or
   leave it to C2?
   *C2's current answer:* no generic flag is required. C2's heuristic filter drops whole-label generic slide phrases
   (Overview, Best Practices, Thank You ...) itself. A `generic` flag from C1 would be used if present.
4. **Pages per node.** Which pages mention each node (`pages: [int]`)? C2 currently re-derives this by text search, which
   can disagree with C1.
5. **Curriculum-level graph.** Is there a graph over the whole curriculum (not per lecture) from which
   `curriculum_defined` edges come? Prerequisites that are not in the current lecture cannot be nodes of a per-lecture
   graph.
6. **Direction and relation vocabulary.** Will any `direction` value other than `source_is_prerequisite_of_target`
   exist? C2 ignores them today.

## 8. Contract-test plan (Phase 2)

Status after Phase 2: items 1 to 3 below are still open because C1's real v1 sample is not on GitHub; items 4 and 5 are done on synthetic data and fake servers; item 6 is open. The schema of C2's own features is frozen as `1.0` and does not depend on this contract's open fields; a change to what C2 reads from C1 that changes a feature's meaning would need a schema bump (see "Schema versioning policy" in [c2-difficulty.md](c2-difficulty.md)).

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
5. `C1ApiGraphProvider` is built: tests with a fake C1 server exist (`test_c1_client.py`, `test_c1_providers.py`,
   `test_graph_provider_selection.py`), and a Docker run against the running C1 returned a v0 graph. Still to add: a test
   against C1's real v1 output once it exists.
6. Version the contract: record the C1 version or a `schema_version` field in the fixture so a change is visible.

## 9. Self-check: validate a C1 graph file

C2 ships a small validator that checks a C1 graph JSON file against this contract (v0 and v1). It reads the file only;
it never calls C1 or changes anything. Run it before sending a sample or pushing a change.

From `edusync/`:

```
uv run --no-sync python packages/c2/scripts/validate_c1_graph.py --file graph.json
```

Options: `--expected auto|v0|v1` (default `auto`: the version is detected from the data; `v1` applies the v1
expectations such as slug ids and descriptions even to v0 data) and `--json` (machine-readable report).

On machines where Windows Smart App Control blocks the virtual-environment launcher (`uv run` then fails with
"os error 4551"), run the same script with the base Python 3.12 and put the project's `site-packages` on the path.

Exit codes: `0` ok (warnings allowed), `1` contract errors, `2` unreadable file or invalid JSON.

### Rules

The table is generated from the `RULES` constant in `edusync_c2/difficulty/contract_check.py`.

| Rule | Severity | Description |
| --- | --- | --- |
| C1-001 | error | The payload is a JSON object. |
| C1-002 | error | The payload has a `graph` object with `nodes` and `edges` lists. |
| C1-003 | error | Every node is an object with a string `id` and a string `label`. |
| C1-004 | error | Node ids are unique. |
| C1-005 | warning | `node_count` and `edge_count` match the list lengths. |
| C1-006 | error | Every edge is an object with a string `source` and a string `target`. |
| C1-007 | error | Every edge `source` and `target` is the id of a node. |
| C1-008 | error | No edge points from a node to itself. |
| C1-009 | error | No two edges share the same source, target and relation_type. |
| C1-010 | error | `relation_type` is `prerequisite` or `co_mentioned`. |
| C1-011 | error | A prerequisite edge `direction`, when present, is `source_is_prerequisite_of_target`. |
| C1-012 | error | A prerequisite edge `confidence`, when present, is a number from 0 to 1. |
| C1-013 | warning | A prerequisite edge has a `confidence`. |
| C1-014 | error | A prerequisite edge `source_type`, when present, is `curriculum_defined` or `inferred`. |
| C1-015 | warning | A prerequisite edge has a `source_type`. |
| C1-016 | error | Edge `evidence`, when present, is a list of objects with an integer `page` and a string `sentence`. |
| C1-017 | warning | The prerequisite edges contain no cycle. |
| C1-018 | error | v1: node ids are slugs (lowercase letters, digits and hyphens). |
| C1-019 | warning | v1: every node has a non-empty `description`. |
| C1-020 | warning | v0: node ids are casefolded. |
| C1-021 | warning | The graph does not mix v0 and v1 shapes (prerequisite edges without node descriptions, or slug ids with no v1 field). |

Version detection: any `prerequisite` edge or any node `description` means v1; only `co_mentioned` edges and no
descriptions means v0; a mix of the two shapes is reported as `mixed` (C1-021); an unparsable payload is `unknown`.

### Example output

Synthetic v0 file (co-mention edges only):

```
C1 graph check: v0.json
Detected version: v0   Result: ok
Nodes: 4   Edges: 5 (prerequisite 0, co_mentioned 5)
Summary: 0 error(s), 0 warning(s) - ok
```

Synthetic v1 file (prerequisite edges, descriptions, slug ids):

```
C1 graph check: v1.json
Detected version: v1   Result: ok
Nodes: 4   Edges: 8 (prerequisite 3, co_mentioned 5)
Summary: 0 error(s), 0 warning(s) - ok
```

The same v1 file, deliberately broken (one unknown relation, one confidence above 1, one empty description):

```
C1 graph check: broken.json
Detected version: v1   Result: FAILED
Nodes: 4   Edges: 8 (prerequisite 3, co_mentioned 4)
  ERROR C1-010 [graph.edges[0]]: relation_type 'similar' is not one of ('prerequisite', 'co_mentioned').
  ERROR C1-012 [graph.edges[7]]: confidence 1.7 is not a number in [0, 1].
  WARNING C1-019 [graph.nodes[0]]: Node has no description.
Summary: 2 error(s), 1 warning(s) - FAILED
```

Note: the synthetic generator (`difficulty/synthetic.py`) emits hyphenated slug ids for v1 graphs and casefolded ids for
v0 graphs, so its output passes. The hand-written `graph_v1` test fixture still uses ids with spaces (for example
`gradient descent`), which C1-018 reports as errors; the tests convert them to slugs first.

## 10. Message for the C1 developer

Paste-ready:

> Hi! I'm working on C2 (difficulty prediction), which reads the knowledge graph that C1 returns from
> `POST /api/c1/lectures/graph`. Here is what C2 needs from it, and a request.
>
> **Required (C2 cannot use the edge without these)**
> - Nodes: `id` (unique, lowercase slug such as `gradient-descent`), `label`, `frequency`.
> - Edges: `source`, `target` (both must be node ids), `relation_type` (`prerequisite` or `co_mentioned`).
> - For `prerequisite` edges: `source` is the concept that must be learned first and `target` is the concept that
>   depends on it, i.e. `direction: "source_is_prerequisite_of_target"`.
>
> **Nice to have (C2 works without them, but the features are weaker)**
> - Node `description`.
> - Prerequisite edges: `confidence` (0.0 to 1.0), `source_type` (`curriculum_defined` or `inferred`) and `evidence`
>   (`page` + `sentence`).
>
> **Request**
> 1. Please push your sample file `edusync/packages/c1/data/samples/sample_lecture_graph.json`, together with the new
>    v1 fields, to GitHub. C2 verified its side only against its own synthetic fixtures, because the real sample is not
>    on GitHub yet, so nothing is confirmed against your real output.
> 2. Before pushing, please run the validator on it:
>    `uv run --no-sync python packages/c2/scripts/validate_c1_graph.py --file <your file>` (from `edusync/`). It
>    exits 0 when the file is fine and prints the rule id for anything that is not.
>
> **Open questions** (details in `docs/edusync/c1-c2-graph-contract.md`)
> 1. Stable ids across lectures: can the same concept keep the same id (and can lectures have stable ids)?
> 2. Aliases such as TDE = Transparent Data Encryption: one node, or an `aliases` list? (C2 does not need either: its
>    heuristic filter merges them. An `aliases` list would still help.)
> 3. Generic concepts such as "Best Practices": will C1 filter or flag them, or should C2? (C2 drops them itself; a flag
>    would help.)
> 4. Pages per node: can each node carry `pages: [int]`?
> 5. Curriculum-level graph: do `curriculum_defined` edges come from a graph over the whole curriculum?
> 6. Will any `direction` value other than `source_is_prerequisite_of_target` exist?
>
> Thanks!

### Addendum: C1-side extraction problems found while building C2's concept filter

Paste-ready. Separate from the message above; it is about the concept labels C1 produces, not the graph shape.

> Hi again! While building C2's concept filter I ran C1's own extractor on 7 realistic slide texts (69 labels in total) to see
> what C2 will receive. Most labels are good (`SQL`, `ACID`, `TDE`, `Structured Query Language`, `Transparent Data
> Encryption`, `Convolutional Neural Network`). These problems come from the extraction itself, and C2 does not repair them:
>
> 1. **Newline collapse fuses a slide title with the first body line.** Newlines become spaces before phrase extraction, so a
>    title and the first line below it become one phrase: `Best Practices The Transparent`, `Overview Today`.
> 2. **Phrases that start or end with a stop word.** About 21 of the 69 labels do. Some are fragments of a longer phrase,
>    such as `controls step size`.
> 3. **Acronym frequencies are counted about twice.** In one test `SQL` appeared 3 times on the slides and got `frequency` 5.
>    C2 uses `frequency` as a feature (`dens.concept_frequency`) and takes the maximum when it merges aliases, so inflated
>    counts feed the model.
> 4. **No alias or description information.** `TDE` and `Transparent Data Encryption` come out as two unrelated nodes.
>
> **The ask**
> - Strip boundary stop words from phrases (or drop phrases that become empty or a single stop word).
> - Keep slide titles separate from body text (do not join them with the first body line before extraction).
> - Count each mention once (an acronym and the phrase it sits in should not both add to the same count).
> - Optional: provide `aliases` (for example `["TDE"]` on `Transparent Data Encryption`) and a short `description` per node.
>   C2 works without them.
>
> Thanks!

What C2 does with what it receives today: a whole-label generic phrase is dropped and obvious aliases are merged in the optional
heuristic filter; a fused title (`Best Practices The Transparent`) or a label with boundary stop words stays a concept, because
C2 cannot tell it from a real multi-word term (see "Known limits" in [c2-difficulty.md](c2-difficulty.md)).
