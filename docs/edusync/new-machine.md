# Run EduSync + BigBlueButton on a new machine

Step-by-step guide to bring up the whole stack (BigBlueButton 4.0 dev server + the EduSync API) on a fresh
PC with Docker. The Windows route was run end to end on a Windows 11 PC; the Linux notes at the bottom are
**untested**. Background and design: [docker.md](docker.md), [architecture.md](architecture.md).

What you end up with:

| URL | What |
|---|---|
| `https://bbb.test` | BigBlueButton 4.0 (HTML5 client, API under `/bigbluebutton/api`) |
| `https://bbb.test/edusync/docs` | EduSync API (C1-C4), served through BigBlueButton's nginx |

## 0. Check the PC

- Windows 10 (22H2) or 11, 64-bit, with administrator rights.
- **16 GB RAM** recommended. BigBlueButton needs about 8 GB for itself, so 8 GB machines will struggle.
- About **30 GB free disk** on the drive Docker uses.
- Virtualization on: Task Manager > Performance > CPU shows "Virtualization: Enabled". If not, enable it in
  the BIOS.
- Ports **80 and 443 free** (stop IIS, Skype, XAMPP or any local web server).

## 1. Install the tools (PowerShell)

```powershell
winget install Git.Git
winget install Docker.DockerDesktop
```

Restart if asked. Then **open Docker Desktop once**, accept the terms and wait until it says "Engine
running". Docker Desktop brings its own Linux VM (WSL2 backend): no Ubuntu and no WSL terminal needed.

## 2. Give Docker enough memory

Create `C:\Users\<name>\.wslconfig`:

```
[wsl2]
memory=10GB
swap=4GB
```

Use `10GB` for a 16 GB PC and `12GB` for 32 GB. Then run `wsl --shutdown` and restart Docker Desktop.

If `C:` has under 40 GB free, also move Docker's disk image: Docker Desktop > Settings > Resources >
Advanced > **Disk image location**.

## 3. Get the code

```powershell
cd C:\dev        # any normal folder, not OneDrive
git clone https://github.com/induranga2004/EduTwin-AI.git
cd EduTwin-AI
```

The repository is public, so cloning needs no login. `main` already contains everything described here.

## 4. Set up and start everything

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\dev.ps1 setup -DryRun    # optional: preview what it will do, changes nothing
.\scripts\dev.ps1 setup -Yes
.\scripts\dev.ps1 first-run
```

During `setup`, two pop-ups appear; click **Yes** on both:

- a Windows "install root certificate" dialog (mkcert trusts its own HTTPS certificate),
- an administrator (UAC) prompt, to add `127.0.0.1 bbb.test` to the hosts file.

If it says `mkcert` is not found right after installing it, close PowerShell, open a new window and run
`setup` again.

`first-run` downloads the 4.2 GB BigBlueButton image, builds EduSync, waits for BigBlueButton to boot and
writes the secrets (`EDUSYNC_BBB_SECRET`, `EDUSYNC_BBB_URL`, `EDUSYNC_HASURA_ADMIN_SECRET`) into `.env`.
Expect **10-30 minutes** the first time.

## 5. Check that it works

```powershell
.\scripts\dev.ps1 status
```

Open `https://bbb.test` (no certificate warning expected) and `https://bbb.test/edusync/docs`.
`https://bbb.test/edusync/health` should report `"bbb": "configured"`.

## 6. Join a test meeting

BigBlueButton has no landing page, so create a meeting through its API:

```powershell
docker compose exec bbb bbb-conf --secret
```

Open the **"Link to the API-Mate"** line it prints (it contains the shared secret, so do not share it),
click **create**, then **join** as moderator. Mic and camera work because the page is served over HTTPS.
Audio/video over UDP through Docker Desktop is **unverified**.

## 7. Every day after that

Open Docker Desktop, then:

```powershell
docker compose up -d      # start
docker compose stop       # pause (keeps state)
```

| Task | Command |
|---|---|
| Status and `/health` | `.\scripts\dev.ps1 status` |
| Follow logs | `.\scripts\dev.ps1 logs bbb` (or `edusync`) |
| HTML5 client with hot reload | `.\scripts\dev.ps1 html5` |
| Rebuild EduSync after Python changes | `docker compose up -d --build edusync` |
| Python API only, no BigBlueButton | `.\scripts\dev.ps1 up -PythonOnly`, then <http://localhost:8100/docs> |
| Wipe everything and start over | `docker compose down -v` |

The full `setup` is only needed again on a new machine or after deleting `.env` or `.dev/`.

## Troubleshooting

| Problem | Fix |
|---|---|
| "running scripts is disabled" | Run the `Set-ExecutionPolicy` line in the same window. |
| "docker daemon not reachable" | Docker Desktop is not running; start it and wait for "Engine running". |
| Port 80 or 443 already in use | Stop the other web server, then run `first-run` again. |
| `curl.exe` shows SSL error 35 or code `000` | Add `--ssl-no-revoke` (Windows curl cannot check revocation for mkcert's CA). Browsers and the scripts are unaffected. |
| Certificate warning in the browser | Run `setup` again, click Yes on the certificate dialog, restart the browser. |
| `https://bbb.test` does not load | Check the hosts file has `127.0.0.1 bbb.test`; BigBlueButton can take a few minutes to boot. |
| EduSync works but `bbb` is `unconfigured` | Run `.\scripts\dev.ps1 first-run` (safe to repeat; refreshes the secrets). |
| BigBlueButton never becomes ready | `docker compose logs --tail 80 bbb`, then fall back to `up -PythonOnly`. |

Notes:

- Every PC creates its own certificate, `.env` and secrets. **Never copy `.env` or `.dev/` between machines**
  and never commit them (both are git-ignored).
- Port 8100 is not reachable from the host in this mode: the BigBlueButton image's `ufw` firewall drops it.
  EduSync is reached through `https://bbb.test/edusync/` instead (see `docker/bbb/edusync.nginx`).

## Linux (untested)

Install Docker Engine, `mkcert` and `git`, then:

```bash
git clone https://github.com/induranga2004/EduTwin-AI.git && cd EduTwin-AI
cp .env.example .env
# in .env set:
#   COMPOSE_PATH_SEPARATOR=:
#   COMPOSE_FILE=docker-compose.yml:docker-compose.sidecar.yml
mkcert -install
mkdir -p .dev/certs
mkcert -cert-file .dev/certs/fullchain.pem -key-file .dev/certs/privkey.pem bbb.test
echo "127.0.0.1 bbb.test" | sudo tee -a /etc/hosts
docker compose up -d --build
docker compose exec bbb bbb-conf --secret      # copy the Secret into EDUSYNC_BBB_SECRET in .env
docker compose up -d --force-recreate edusync
```

Audio and video are most likely to work on Linux. macOS and Apple Silicon are not tested, and the
BigBlueButton image may not run there.
