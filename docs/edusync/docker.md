# EduSync AI - Docker

Files: `docker-compose.yml`, `docker-compose.sidecar.yml`, `.env.example`, `edusync/Dockerfile`, `justfile`
(all at repo root except the Dockerfile). Anything about running BBB in Docker here is **UNVERIFIED** until the
Phase-0 spike below is done.

## Profiles

`COMPOSE_PROFILES` in `.env`:

| Value | Runs |
|-------|------|
| `bbb,edusync` | BBB dev container and the EduSync API (default in `.env.example`) |
| `edusync` | Python API only (fallback when BBB cannot run) |

BBB uses the prebuilt dev image `imdt/bigbluebutton:4.0.x-develop` (override with `BBB_IMAGE`; the tag is an
assumption): one privileged systemd container, as in <https://github.com/bigbluebutton/docker-dev>. The repo is
mounted at `/home/bigbluebutton/src`; `bigbluebutton-html5/run-dev.sh` runs webpack-dev-server with hot reload
behind nginx (`just html5`).

## First run

```bash
cp .env.example .env          # never commit .env
docker compose up             # or: just up
curl http://localhost:8100/health
```
API docs: <http://localhost:8100/docs>. Get the BBB shared secret with `just salt` and put it in
`EDUSYNC_BBB_SECRET`.

## Windows setup

- Docker Desktop with the WSL2 backend. **Clone the repo inside WSL2** (e.g. `~/src`), not on `E:\` / `/mnt/*`:
  bind mounts across the boundary are slow (node_modules already uses a named volume).
- Browsers require HTTPS for mic/camera except on `localhost`. Create a cert for `bbb.test` with
  [mkcert](https://github.com/FiloSottile/mkcert) into `./.dev/certs`, and add `127.0.0.1 bbb.test` to
  `C:\Windows\System32\drivers\etc\hosts`. Browse to `https://bbb.test`.
- UDP media: only a small range (default 16384-16484, plus 7882/udp) is published; whether audio/video works
  through Docker Desktop is UNVERIFIED. Test from the same machine first.

## Sidecar override

BBB services bind to `127.0.0.1` inside the BBB container (Hasura 8085, Redis 6379), so a separate container
cannot reach them. The sidecar override puts EduSync in the BBB container's network namespace
(`network_mode: service:bbb`) and sets `EDUSYNC_HASURA_URL` / `EDUSYNC_REDIS_URL` to `127.0.0.1`. Port 8100 is
then published by the `bbb` service. Requires Docker Compose >= 2.24 (`!reset` tag).

```bash
docker compose -f docker-compose.yml -f docker-compose.sidecar.yml up
# or set COMPOSE_FILE=docker-compose.yml:docker-compose.sidecar.yml in .env
```
Enable only after the spike passes. Without it, `.env.example` points at `host.docker.internal`, which only
works if BBB's services are reachable on the host (they bind 127.0.0.1 inside BBB, so expect this not to work).

## Phase-0 spike checklist

Record results here (edit this table in a PR).

| # | Check | Result |
|---|-------|--------|
| 1 | Pull `imdt/bigbluebutton:4.0.x-develop`; note image size | TBD |
| 2 | Boots with privileged, cgroup host, NET_ADMIN | TBD |
| 3 | systemd healthy: `bbb-conf --status` | TBD |
| 4 | `https://bbb.test` loads | TBD |
| 5 | Create and join a meeting | TBD |
| 6 | Mic permission prompt works | TBD |
| 7 | Audio/video over UDP works | TBD |
| 8 | `run-dev.sh` hot reload from a WSL-side edit | TBD |
| 9 | `curl 127.0.0.1:8085` from the sidecar reaches Hasura | TBD |

If boot (2-3) or HTTPS (4) fail, ship the Python-only default (`COMPOSE_PROFILES=edusync`) and develop BBB
integration against fakes or a separate Linux host.

## Troubleshooting

- BBB container exits or systemd does not start: check privileged mode, `cgroup: host`, cgroup v2 in WSL2, and
  RAM (raise limits in `%UserProfile%\.wslconfig`).
- Mic/camera blocked: you are on HTTP with a non-localhost host; use the mkcert HTTPS setup.
- Sidecar cannot reach Hasura: confirm the override is applied (`docker compose config`) and run
  `bbb-conf --status` in the BBB container. Dev admin secret defaults to `bigbluebutton`.
- Slow file watching/hot reload: repo is on the Windows drive; move it into WSL2.
- `uv` or Python version errors: only Python 3.12 is supported; let `uv` fetch it.
- API health check failing: see `docker compose logs edusync`.

## Known risks

- systemd-in-Docker BBB on Docker Desktop/Windows is unproven for this team.
- WebRTC UDP through Docker Desktop NAT may not work.
- BBB integration details in [architecture.md](architecture.md) come from code reading, not a live run.
- The `4.0.x-develop` tag moves; pin a digest once the spike succeeds.
