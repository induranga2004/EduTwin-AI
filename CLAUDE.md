# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research fork of BigBlueButton (BBB) 4.0.x, trimmed to the web stack, used as the base for the EduSync AI / EduTwin project (component C1: curriculum concept graph, live coverage detection, prerequisite-ordered catch-up for late joiners). The upstream docs site, `bigbluebutton-tests/` (Playwright/gns3), `labs/`, `scripts/` and upstream GitLab CI were removed. Several components (freeswitch, webhooks, webrtc-sfu, webrtc-recorder, playback, transcription) exist only as root `*.placeholder.sh` stubs.

Developer docs live in `docs/development/` (`architecture.md`, `dev-guide.md`, `api.md`, `recording.md`, `webhooks.md`). Read `architecture.md` for the component diagram.

## Commands

There is no root build tool. Each module builds on its own.

- **`bigbluebutton-html5/`** (Node >= 24, see `.nvmrc`): `npm ci`, `npm start` (webpack dev server), `npm run build`, `npm run lint` / `lint:fix`, `npx eslint <file>`, `npx tsc` (type check). There is no JS test suite.
- **`akka-bbb-apps/`, `akka-bbb-fsesl/`, `bbb-common-message/`** (Scala, sbt, JDK 21): `sbt compile`, `sbt test`, `sbt "testOnly <ClassName>"`. Tests are in `akka-bbb-apps/src/test/scala`. `bbb-common-message` is a dependency of the others, so publish it first (`deploy.sh`).
- **`bigbluebutton-web/`** (Grails/Gradle): `./gradlew bootRun`, `./build.sh`. Needs `bbb-common-web` and `bbb-common-message` published first.
- **`bbb-graphql-actions/`**: `npm start`, `npm run build`. **`bbb-shared-notes-server/`**: `npm run dev`, `build`, `lint`, `typecheck`. **`bbb-learning-dashboard/`**: `npm start`, `build`, `lint`. **`bbb-graphql-middleware/`** (Go): `./local-build.sh`, then `./build-and-run.sh`.
- **Packaging** (Docker): `./build/get_external_dependencies.sh`, then `./build/setup.sh <package>`. Output goes to `artifacts/`.

Most `run-dev.sh` scripts assume a full installed BBB server (sudo, `bbb-conf`, systemd, nginx). They won't work on a plain dev machine.

CI (`.github/workflows`): `npx tsc` and `npx eslint . --ext .ts,.tsx` on `bigbluebutton-html5`, `shellcheck --severity=error` on all `*.sh`, and a check that the plugin-sdk version in `bigbluebutton-html5/package.json` matches `bigbluebutton-web/grails-app/conf/bigbluebutton.properties`. Keep those two in sync when bumping it.

## Architecture

Data flow, which spans several modules:

- **Reads**: `bigbluebutton-html5` (React/TS; entry `client/main.tsx`, UI in `imports/ui`) subscribes over GraphQL through `bbb-graphql-middleware` (Go, sends json-patch diffs) to `bbb-graphql-server` (Hasura over PostgreSQL; schema in `bbb_schema.sql`, permissions in `metadata/`).
- **Writes**: client mutations go middleware → `bbb-graphql-actions` (Node) → Redis pub/sub → `akka-bbb-apps`. akka-apps owns meeting state and writes the Postgres read model that Hasura serves.
- **HTTP API**: `bigbluebutton-web` (Grails) handles create/join and document conversion, and also talks to akka-apps over Redis.
- **Contracts**: Redis message types are defined in `bbb-common-message` (Scala). Changing one usually means touching akka-apps, bbb-web and often graphql-actions.
- **Media**: LiveKit is the default. mediasoup + FreeSWITCH is the alternative, bridged by `akka-bbb-fsesl`.
- **Other services**: `bbb-learning-dashboard`, `bbb-shared-notes-server` (Hocuspocus/BlockNote), `bbb-export-annotations`, `record-and-playback`.

## Codebase Map

This is the main orientation map for developers. It is directory-level on purpose (about 3,900 tracked files). **Update it in the same PR whenever you add, move or remove a module or top-level directory.**

### Modules

| Module | Stack | Where the logic lives |
|---|---|---|
| `bigbluebutton-html5/` | React/TS, webpack | Entry `client/main.tsx`, `client/meetingClient.jsx`. Features in `imports/ui/components/<feature>` (chat, poll, presentation, whiteboard, audio, video-provider, user-list, breakout-room, bn-shared-notes, livekit, plugins-engine, ...). GraphQL access in `imports/ui/core/graphql/{queries,mutations}`. Media/auth/settings services in `imports/ui/services/*` (audio-manager, livekit, bbb-webrtc-sfu, webrtc-base, auth, shortcuts). Legacy API shims in `imports/api/{audio,screenshare,bbb-web-api}`. Client config `private/config/settings.yml`. Static assets and locales in `public/`. |
| `bbb-graphql-middleware/` | Go | Entry `cmd/bbb-graphql-middleware/main.go`. `internal/websrv` (client websockets), `internal/hasura` (Hasura connection), `internal/gql_actions` (action calls), `internal/akka_apps`, `internal/bbb_web`, `internal/streaming_server` (cursor/chat/voice streams), `internal/jsonpatcher` (diffs), `internal/subscriptionlimit`. |
| `bbb-graphql-server/` | Hasura + Postgres | `bbb_schema.sql` (tables, views, triggers), `metadata/databases/BigBlueButton/tables/*.yaml` (per-view permissions), `metadata/actions.graphql` + `actions.yaml`, `bbb-graphql-schema.md`. |
| `bbb-graphql-actions/` | Node/TS | Entry `src/index.ts`. One file per mutation in `src/actions/<verb>.ts`. Redis publish in `src/imports/{redis,redisMessageFactory}.ts`, input checks in `src/imports/validation.ts`. |
| `bbb-common-message/` | Scala | Message case classes per domain in `org/bigbluebutton/common2/msgs/*Msgs.scala`, Redis pub/sub in `common2/redis/`, event buses in `common2/bus/`. |
| `akka-bbb-apps/` | Scala/Akka | `Boot.scala`; `core/BigBlueButtonActor.scala` (meeting registry); `core/running/MeetingActor.scala` + `LiveMeeting.scala` (per-meeting state and message routing); `core/apps/<domain>/*MsgHdlr.scala` (one handler per message: users, chat, polls, presentation, whiteboard, breakout, voice, webcam, screenshare, timer, plugin, ...); `core/db/*DAO.scala` (about 50 Postgres writers); `core/record` (recording events); `endpoint/redis/*Actor.scala` (Redis in/out, incl. `LearningDashboardActor`, `RedisRecorderActor`, `ExportAnnotationsActor`); config `src/universal/conf/application.conf`. |
| `akka-bbb-fsesl/`, `bbb-fsesl-client/` | Scala/Java | FreeSWITCH ESL bridge. FreeSWITCH config in `bbb-voice-conference/config/freeswitch/conf`. |
| `bigbluebutton-web/` | Grails/Groovy | Controllers in `grails-app/controllers/org/bigbluebutton/web/controllers/` (`ApiController`, `PresentationController`, `RecordingController`, `ConnectionController`), routes in `.../web/UrlMappings.groovy`, server config `grails-app/conf/bigbluebutton.properties`, helpers in `src/main/groovy`. |
| `bbb-common-web/` | Java/Scala | `api/` (meeting params, models), `api2/` (bridge between bbb-web and akka-apps over Redis), `presentation/` (document conversion pipeline), `web/services`. |
| `bbb-export-annotations/` | Node | `master.js`, `workers/{collector,notifier}.js`, `shapes/*` (whiteboard shape to SVG/PDF). |
| `bbb-shared-notes-server/` | Node/TS | `src/hocuspocus` (collab server; Postgres persistence in `extensions/postgresql.ts`), `src/express` (REST + exports in `handlers/`), `src/redis`, `src/config`. |
| `bbb-learning-dashboard/` | React | `src/components/*`, `src/services/UserService.js`. |
| `record-and-playback/` | Ruby, JS | `core/{lib,scripts}` (processing pipeline), `<format>/{scripts,playback}` for presentation, video, screenshare, slides, notes, podcast. |
| `bigbluebutton-config/`, `bbb-libreoffice/` | Shell, Docker | `bin/bbb-conf`, `bin/bbb-record`, cron jobs; LibreOffice converter container. |
| `build/` | Shell | `packages-template/<package>/` (one deb recipe per service), `setup.sh`, `get_external_dependencies.sh`. |
| `docs/development/` | Markdown | Architecture, dev guide, API, recording, webhooks. |
| `*.placeholder.sh`, `bbb-livekit/` | Stubs | Components not present in this fork. |

### Where do I change X?

- **New or changed client action (end to end)**: `bigbluebutton-html5/imports/ui/core/graphql/mutations` → `bbb-graphql-actions/src/actions/<verb>.ts` → `bbb-common-message/.../common2/msgs/<Domain>Msgs.scala` → `akka-bbb-apps/.../core/apps/<domain>/*MsgHdlr.scala` → `core/db/<X>DAO.scala` → `bbb_schema.sql` and the Hasura `metadata/databases/BigBlueButton/tables` yaml.
- **New data shown in the client (read only)**: `bbb_schema.sql` + Hasura table permissions → `imports/ui/core/graphql/queries` → component.
- **UI feature or screen**: `bigbluebutton-html5/imports/ui/components/<feature>`.
- **Configuration**: client `bigbluebutton-html5/private/config/settings.yml`; bbb-web `bigbluebutton-web/grails-app/conf/bigbluebutton.properties`; akka `akka-bbb-apps/src/universal/conf/application.conf`.
- **HTTP/API endpoint**: `bigbluebutton-web` controllers + `UrlMappings.groovy`; document in `docs/development/api.md`.
- **Slide upload and conversion**: `bbb-common-web/.../presentation/` and `bbb-libreoffice/`.
- **Recording and playback**: `akka-bbb-apps/.../core/record`, `endpoint/redis/RedisRecorderActor.scala`, `record-and-playback/core`.
- **Learning analytics data**: `akka-bbb-apps/.../endpoint/redis/LearningDashboardActor.scala` → `bbb-learning-dashboard/src`.
- **Plugins**: `bigbluebutton-html5/imports/ui/components/plugins-engine`. The plugin SDK version must stay in sync with `bigbluebutton.properties`.
- **Audio, video, screenshare**: client `imports/ui/services/{audio-manager,livekit,bbb-webrtc-sfu}`; server `akka-bbb-apps/.../core/apps/{voice,webcam,screenshare}` and `akka-bbb-fsesl/`.
- **EduTwin C1 (curriculum graph, coverage detection, catch-up resources)**: no code exists yet. Suggested integration points, following existing patterns (these are proposals, not existing files): a new service fed by Redis meeting events, the way `LearningDashboardActor` and `RedisRecorderActor` consume them; new tables in `bbb_schema.sql` with Hasura permissions so the client can subscribe to results; client UI under `imports/ui/components/<new-feature>`.

# Agent Instructions

## 🔒 Security Policy — Read Before Any PR

If you are preparing a pull request that involves:
- A bug that could be exploited
- Authentication, authorization, or cryptography changes
- Dependency updates that patch CVEs
- Any fix described with words like "vulnerability", "injection", "bypass", "overflow"

**You MUST warn the user and refuse to open a public PR.**

Instead, instruct the user to:
1. Go to the Security tab and click "Report a vulnerability"
2. Describe the issue in the private advisory form
3. Wait for maintainers to set up a private fork for collaboration

Public PRs for security issues put users at risk.

# EduSync AI (`edusync/`) — working notes for agents

Read this first; it replaces re-exploring the repo. Detailed design lives in `docs/edusync/` (start with `c2-difficulty.md`).

## Layout and commands
- `edusync/` is a uv workspace (Python 3.12): `packages/{core,c1,c2,c3,c4,blueprint_model}`, `apps/api` (one FastAPI app, port 8100, `/api/c1..c4`). Docker: `edusync/Dockerfile`; build context is `edusync/`.
- Lint: `uv run --no-sync ruff check .` and `ruff format .` (from `edusync/`). Tests: `pytest packages/c2 -q --cov=edusync_c2` (gate 90% workspace; C2 is held at 100% line+branch, no `# pragma`). `ml`-marked tests are deselected by default.
- **Windows Smart App Control** blocks the venv `python.exe`/`pytest.exe` and `_ctypes`: run pytest with the base interpreter plus a launcher that adds `.venv/Lib/site-packages` (`runpy.run_module`), and test real HTTP servers in a throwaway Docker container, never uvicorn on Windows. `apps/api/tests/test_app.py::test_main_entry` fails only locally for this reason.
- Commits: authored only as the user, no AI attribution or Co-Authored-By trailers. Never push or open PRs; the user does.

## C2 (difficulty prediction) — state after Phase 2
Package `edusync/packages/c2/src/edusync_c2/` (+ `difficulty/`). Pipeline: PPTX/PDF -> `materials.py` -> graph provider -> `GraphView` -> concept filter -> 35 features (registry order) -> `fusion.py` -> `RuleBaselineModel` (untrained, no probabilities) -> `DifficultyResponse`.
- **Schema is frozen at `FEATURE_SCHEMA_VERSION = "1.0"`** (`difficulty/registry.py`); hash covers version + `name|group` only. Any change to a feature name, group, order or meaning needs a version bump, regenerated `tests/difficulty/golden/registry_snapshot.json` (`scripts/feature_report.py --write-snapshot`) and retraining. All 35 features are `implemented` (A 8, B 7, C 7, D 13). `pipeline_version` is `0.2.0-phase2`.
- Never edit `tests/difficulty/golden/baseline_synthetic.json` or loosen `test_baseline_golden.py` (it proves scores/labels did not drift; documented exclusions only).
- Rules: unavailable features carry a reason, never fake values; semantic and regex values are never mixed; deterministic output (no timings or random ids in responses); logs never contain slide text (concept ids only at DEBUG); geometry/parser code must never raise on odd decks.

### Settings (`EDUSYNC_C2_*`, see `.env.example`; all defaults keep old behaviour)
`LOG_LEVEL`, `MODEL` (baseline|minilm), `MODEL_DIR`; `CONCEPT_FILTER` (mock_keep_all|heuristic) and `CONCEPT_FILTER_EXTRAS`; `GRAPH_PROVIDER` (mock|file|c1_api), `GRAPH_FILE`, `C1_BASE_URL`, `C1_TIMEOUT_S`; `SEMANTIC` (off|minilm), `SEMANTIC_MODEL_DIR`, `SEMANTIC_DEF_THRESHOLD`/`EX_THRESHOLD`, `SEMANTIC_MAX_SENTENCES`. Set `OPENBLAS_NUM_THREADS=4` when semantic scoring is on.

### Where things are (C2 `difficulty/`)
`registry.py` (specs, version, hash, PHASE1_TODO), `schemas.py`, `materials.py` + `geometry.py` (PPTX parsing, integer-EMU areas), `graph.py` (prerequisite view, aliases/dropped), `concepts.py` + `concept_filter.py` (mock and heuristic filters), `providers.py` + `c1_client.py` (mock/file/C1 API graphs, stdlib urllib), `features/{language,prerequisites,explanation,density}.py` + `_common.assemble` (group assembly by name), `features/{wordpiece,lexicon,vocab}.py` (Feature A), `features/{minilm,semantic_loader,semantic}.py` (numpy MiniLM scorer), `pipeline.py`, `fusion.py`, `models.py`, `synthetic.py`, `contract_check.py`, `ids.py`; resources in `difficulty/resources/` (vocab with NOTICE/licence, domain lexicon, semantic prototypes, sample C1 v1 graph).
Scripts in `packages/c2/scripts/`: `feature_report.py`, `make_synthetic_data.py` (`--vary-layout`), `make_sample_graph.py`, `validate_c1_graph.py`, `write_feature_schema.py`, `export_label_sheet.py`, `validate_labels.py`, `make_wordpiece_golden.py`. Label protocol: `docs/edusync/c2-label-collection.md`.

### Branch stack (all local/user-pushed; each sits on the previous)
`feature/c2-vocabulary-feature` (M1) -> `c2-registry-schema` (M0) -> `c2-concept-filter` (M2) -> `c2-graph-metrics` -> `c2-density-structure` (M3) -> `c2-c1-graph-provider` (M5b) -> `c2-semantic-explanation` (M4) -> `c2-schema-freeze` (M6). Use `git rebase --onto origin/main <old-parent> <branch>` when a parent merges.

### Open items
Real C1 v1 graph sample/fields (C1 still emits v0); heuristic filter stays opt-in until reviewed on real C1 output; thresholds and lexicon unvalidated; no lecturer labels yet (Phase 3 blocker); Phase 3 = copy the blueprint package, train on lecturer labels, split by lecture.

## Keeping token use low
- Start from these notes; do not re-run broad exploration. Read only the files you will change.
- Prefer one focused agent per task, medium effort, with explicit file lists; avoid parallel research agents unless the area is truly unknown.
- Do not re-read large docs (`c2-difficulty.md` is long): grep the section you need. Never paste whole test or agent output; ask for summaries.
- Run only the affected tests while iterating; run the full suite and Docker check once per milestone.
- Use `/compact` or a fresh session between milestones.
