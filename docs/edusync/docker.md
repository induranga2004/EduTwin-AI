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

Linux/macOS/WSL shell:

```bash
cp .env.example .env          # never commit .env
docker compose up             # or: just up
curl http://localhost:8100/health
```
API docs: <http://localhost:8100/docs>. Get the BBB shared secret with `just salt` and put it in
`EDUSYNC_BBB_SECRET`. On Windows use the PowerShell flow below, which does this for you.

## Windows setup (PowerShell only)

Docker Desktop (WSL2 backend) provides the hidden Linux VM, so no WSL/Ubuntu terminal is needed. The repo can stay
on the Windows drive (e.g. `E:`). Everything is driven by `scripts\dev.ps1` (same tasks as the `justfile`):

```powershell
winget install Docker.DockerDesktop     # once; start Docker Desktop and wait for "Engine running"
.\scripts\dev.ps1 setup                  # .env, mkcert + certs, hosts entry, resource checks
.\scripts\dev.ps1 first-run              # up, wait for BBB, store secrets in .env, restart edusync
.\scripts\dev.ps1 status                 # containers + /health ("bbb": "configured")
```

Use `-DryRun` on any command to see what it would do without changing anything, and `-Yes` to skip prompts.
Other commands: `up`, `secret`, `logs [service]`, `down`, `html5` (HMR dev server). Docker Desktop must be running
for all of them.

What `setup` does: checks `docker` and the daemon (offers to start Docker Desktop), copies `.env.example` to `.env`
if missing, sets `COMPOSE_PATH_SEPARATOR=:` and the sidecar `COMPOSE_FILE` line, installs mkcert via winget if
needed, creates `.dev\certsullchain.pem` and `privkey.pem` for `BBB_HOSTNAME` (default `bbb.test`), and adds
`127.0.0.1 bbb.test` to `C:\Windows\System32\drivers\etc\hosts`. **Editing the hosts file needs administrator**:
the script relaunches only that step elevated and you accept the UAC prompt. If `mkcert` is not found right after
installing, open a new PowerShell window and re-run.

`first-run` polls `bbb-conf --secret` inside the container (BBB can take several minutes to boot), writes
`EDUSYNC_BBB_SECRET`, `EDUSYNC_BBB_URL` and `EDUSYNC_HASURA_ADMIN_SECRET` to `.env` and recreates `edusync`.
This parsing against a live container is UNVERIFIED. Browse to `https://bbb.test`.

### Resources (important)

- **WSL2 memory.** The default WSL cap is about half of RAM (about 6.9 GB on a 13.8 GB machine), too low for BBB.
  Create `%UserProfile%\.wslconfig`:

  ```ini
  [wsl2]
  memory=10GB
  swap=4GB
  ```

  Then run `wsl --shutdown` and restart Docker Desktop. `setup` warns if the value is missing or below 8 GB but
  never writes the file.
- **Disk.** Docker's disk image lives on `C:` by default. With little free space there, move it via Docker Desktop
  > Settings > Resources > Advanced > Disk image location (`setup` warns if `C:` has under 40 GB free).
- **Bind-mount speed.** Bind mounts from `E:` are slower than from inside WSL2; `node_modules` already uses a named
  volume. If hot reload is too slow, clone inside WSL2 instead.
- Browsers require HTTPS for mic/camera except on `localhost`, hence the mkcert certificate and hosts entry.
- UDP media: only a small range (default 16384-16484, plus 7882/udp) is published; whether audio/video works
  through Docker Desktop is UNVERIFIED. Test from the same machine first.

### Python-only fallback

If BBB cannot run, `.\scripts\dev.ps1 up -PythonOnly` starts only the API (profile `edusync`, no sidecar file;
`.env` is not modified). `status` then shows `"bbb": "unconfigured"`.

## Sidecar override

BBB services bind to `127.0.0.1` inside the BBB container (Hasura 8085, Redis 6379), so a separate container
cannot reach them. The sidecar override puts EduSync in the BBB container's network namespace
(`network_mode: service:bbb`) and sets `EDUSYNC_HASURA_URL` / `EDUSYNC_REDIS_URL` to `127.0.0.1`. EduSync is then
served through BBB's nginx at `https://<BBB_HOSTNAME>/edusync/` (docs: `/edusync/docs`) via
`docker/bbb/edusync.nginx`, because the BBB image's `ufw` firewall drops port 8100 (verified: the app answered
inside the container but not from the host). Requires Docker Compose >= 2.24 (`!reset` tag).

```bash
docker compose -f docker-compose.yml -f docker-compose.sidecar.yml up
# or set COMPOSE_FILE=docker-compose.yml:docker-compose.sidecar.yml in .env
# (with COMPOSE_PATH_SEPARATOR=:, needed on Windows; `scripts\dev.ps1 setup` writes both)
```
Enable only after the spike passes (`dev.ps1 setup` enables it for BBB mode). Without it, `.env.example` points at `host.docker.internal`, which only
works if BBB's services are reachable on the host (they bind 127.0.0.1 inside BBB, so expect this not to work).

## Phase-0 spike checklist

Record results here (edit this table in a PR). On Windows run the steps via `scripts\dev.ps1` (`setup`, `first-run`, `status`, `html5`).

| # | Check | Result |
|---|-------|--------|
| 1 | Pull `imdt/bigbluebutton:4.0.x-develop`; note image size | TBD |
| 2 | Boots with privileged, cgroup host, NET_ADMIN | TBD |
| 3 | systemd healthy: `bbb-conf --status` | TBD |
| 4 | `https://bbb.test` loads | TBD |
| 5 | Create and join a meeting | TBD |
| 6 | Mic permission prompt works | TBD |
| 7 | Audio/video over UDP works | TBD |
| 8 | `run-dev.sh` hot reload from an edit on the host (`dev.ps1 html5`) | TBD |
| 9 | `curl 127.0.0.1:8085` from the sidecar reaches Hasura | TBD |

If boot (2-3) or HTTPS (4) fail, ship the Python-only default (`COMPOSE_PROFILES=edusync`) and develop BBB
integration against fakes or a separate Linux host.

## Troubleshooting

- BBB container exits or systemd does not start: check privileged mode, `cgroup: host`, cgroup v2 in WSL2, and
  RAM (raise limits in `%UserProfile%\.wslconfig`, then `wsl --shutdown`).
- `dev.ps1` says Docker is not reachable: start Docker Desktop and wait for "Engine running".
- Mic/camera blocked: you are on HTTP with a non-localhost host; use the mkcert HTTPS setup.
- Sidecar cannot reach Hasura: confirm the override is applied (`docker compose config`) and run
  `bbb-conf --status` in the BBB container. Dev admin secret defaults to `bigbluebutton`.
- Slow file watching/hot reload: repo is on the Windows drive; clone it inside WSL2 if it is too slow.
- `uv` or Python version errors: only Python 3.12 is supported; let `uv` fetch it.
- API health check failing: see `docker compose logs edusync`.

## Known risks

- systemd-in-Docker BBB on Docker Desktop/Windows is unproven for this team.
- WebRTC UDP through Docker Desktop NAT may not work.
- BBB integration details in [architecture.md](architecture.md) come from code reading, not a live run.
- The `4.0.x-develop` tag moves; pin a digest once the spike succeeds.
