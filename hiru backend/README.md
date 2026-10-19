# EduSync AI — C1 Backend

This component provides the backend for:

- Curriculum concept graph construction
- Live curriculum coverage detection
- Prerequisite-ordered catch-up resources for late-arriving students

Research component: C1 — Curriculum Alignment and Lecture Navigation Intelligence.
Related GitHub issue: #1

---

# Research Fork — BigBlueButton 4.0 (Web Stack Base)

This repository is a trimmed copy of BigBlueButton v4.0.x used as the base for a university research project (AI/ML-based monitoring, in progress).

**Removed from upstream** (not needed to build or run the web stack): the Docusaurus docs site (`docs/` — developer docs kept in [`docs/development/`](docs/development/)), Playwright/gns3 tests (`bigbluebutton-tests/`), `bbb-graphql-client-test/`, `labs/`, `scripts/`, upstream CI (`.github/`, `.gitlab-ci.yml`), `publiccode.yml`, `transifex.yml`.

**Kept:** HTML5 client, bbb-web, akka-apps/fsesl, GraphQL server (Postgres/Hasura), middleware, actions, audio/video stack, recording, learning dashboard, shared notes, export annotations, and `build/` packaging.

Developer setup: see [`docs/development/dev-guide.md`](docs/development/dev-guide.md).

---

BigBlueButton
=============
BigBlueButton is an open-source virtual classroom designed to help teachers teach and learners learn.  

BigBlueButton supports real-time sharing of audio, video, slides (with whiteboard annotations), chat, and the screen.  Instructors can engage remote students with polling, emojis, multi-user whiteboards, shared notes, and breakout rooms.  During the session, BigBlueButton generates analytics that are visible to moderators in the Learning Analytics Dashboard.

Presenters can record and playback content for later sharing with others.

We designed BigBlueButton for online learning, it can be used for many other applications as well.  The educational use cases for BigBlueButton are

  * Online tutoring (one-to-one)
  * Flipped classrooms (recording content ahead of your session)
  * Group collaboration (many-to-many)
  * Online classes (one-to-many)

The latest version is BigBlueButton 4.0.  You can install BigBlueButton it on Ubuntu 24.04 using [bbb-install.sh](https://github.com/bigbluebutton/bbb-install) within 30 minutes (or your money back 😉).

For full technical documentation of BigBlueButton -- including architecture, features, API, and GreenLight (the default front-end) -- see [https://docs.bigbluebutton.org/](https://docs.bigbluebutton.org/).

BigBlueButton and the BigBlueButton Logo are trademarks of [BigBlueButton Inc](https://bigbluebutton.org).


---

## EduSync AI backend

Python backend (FastAPI, uv workspace in [`edusync/`](edusync)) for the EduSync AI components C1-C4.

```bash
cp .env.example .env
docker compose up
curl http://localhost:8100/health    # API docs: http://localhost:8100/docs
```

With BBB (sidecar mode) the API is at `https://bbb.test/edusync/` (docs: `/edusync/docs`).
Python-only (no BBB): set `COMPOSE_PROFILES=edusync` in `.env`. Without Docker: `uv --directory edusync sync`.

Windows (PowerShell only, Docker Desktop with the WSL2 backend; no WSL terminal needed, the repo may stay on `E:`):

```powershell
.\scripts\dev.ps1 setup         # .env, mkcert certs, hosts entry (asks for admin), resource checks
.\scripts\dev.ps1 first-run     # starts BBB + EduSync, stores BBB secrets in .env
.\scripts\dev.ps1 up -PythonOnly   # fallback: Python API only
```

Docker Desktop must be running. Add `-DryRun` to preview. BBB needs more memory than the default WSL cap: set
`memory=10GB` / `swap=4GB` under `[wsl2]` in `%UserProfile%\.wslconfig`, run `wsl --shutdown` and restart Docker
Desktop; consider moving Docker's disk image off `C:` (Settings > Resources > Advanced). Running BBB in Docker on
Windows is UNVERIFIED; details in [Docker docs](docs/edusync/docker.md).

Docs: [architecture](docs/edusync/architecture.md), [development](docs/edusync/development.md), [Docker](docs/edusync/docker.md), [run on a new machine](docs/edusync/new-machine.md).
