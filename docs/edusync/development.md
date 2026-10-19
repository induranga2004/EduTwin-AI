# EduSync AI - Development Guide

## Setup

Prerequisites: [`uv`](https://docs.astral.sh/uv/), optionally [`just`](https://github.com/casey/just), git.
Use `uv` only; never bare `pip`. Python is pinned to **3.12** (`requires-python = ">=3.12,<3.13"`). A local
Python 3.14 is NOT supported; `uv` fetches 3.12 itself.

```bash
uv --directory edusync sync     # venv + all workspace packages + dev tools
pre-commit install              # optional; config at repo root, restricted to edusync files
```

## Layout

```
edusync/
  pyproject.toml        # uv workspace root + ruff/pytest/coverage config
  uv.lock
  Dockerfile
  apps/api/             # edusync_api: FastAPI app (asgi.py, main.py, routes.py, __main__.py)
  packages/core/        # edusync_core: models, settings, Component contract, BBB adapters, fakes
  packages/c1..c4/      # edusync_c1..c4: one package per component (src/ and tests/)
  fixtures/             # small committed test fixtures
  data/                 # local data/models (not committed)
```

## Commands

```bash
uv --directory edusync run ruff check .
uv --directory edusync run ruff format .
uv --directory edusync run pytest --cov
just check              # lint + format check + tests
```
Other `just` recipes: `sync`, `test`, `lint`, `fmt`, `up`, `down`, `logs`, `ps`, `salt`, `html5`.

Run the API locally:

```bash
uv --directory edusync run python -m edusync_api      # uvicorn on 127.0.0.1:8100 (EDUSYNC_HOST overrides host)
```
Then open <http://localhost:8100/docs>.

## Adding an endpoint

1. Edit the router in `edusync/packages/cN/src/edusync_cN/`.
2. Keep handlers thin; put logic in plain functions/classes.
3. Use `edusync_core.models` types; add a shared model there only if two or more components need it.
4. Add tests in `packages/cN/tests/` (FastAPI `TestClient` / `httpx`).

## Adding a component

Create `edusync/packages/cN` with its own `pyproject.toml`; export
`COMPONENT = Component(name=..., title=..., description=..., router=...)` from `edusync_cN`; add it to
`[tool.uv.sources]` and ruff `known-first-party` in `edusync/pyproject.toml`; register it in `edusync_api`
(see how c1 is wired).

## Fakes for testing

`edusync_core.testing` is intended to hold in-memory fakes of the BBB adapters (Hasura client, Redis event bus,
BBB API client) so unit tests never need a running BBB. UNVERIFIED: the module did not exist when this was
written; check `packages/core/src/edusync_core/` for the current names. Unit tests must not use the network,
Redis, or real models.

## Testing and coverage policy

- New logic requires tests.
- Coverage gate `fail_under = 80`, branch coverage (`edusync/pyproject.toml`).
- Tests needing real ML weights: `@pytest.mark.ml`, deselected by default (`-m 'not ml'`).
- `asyncio_mode = "auto"`: plain `async def test_...` works.

## Conventions

- Branches: `feat/`, `fix/`, `docs/`, `ci/`. Conventional commits, e.g. `feat(c1): add coverage endpoint`.
- Small PRs, squash merge.
- Never commit `.env`, data files or model weights.
- Ruff: line length 100, rules E,F,I,UP,B,SIM,C4,PT,RUF,ASYNC.
- CI: `.github/workflows/python-ci.yml` with jobs `lint`, `test`, `docker`.
- Security problems: do not open a public PR; use the repository Security tab -> "Report a vulnerability".

## C1 prototype migration notes

The earlier standalone C1 prototype served `/api/lectures/*`. In the unified app these routes live under the
component prefix: **`/api/lectures/*` becomes `/api/c1/lectures/*`**. Update clients, scripts and notebooks.
Configuration now comes from `CoreSettings` (`EDUSYNC_` environment prefix).
