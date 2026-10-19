<#
.SYNOPSIS
  Windows (PowerShell only) helper for running BigBlueButton + EduSync AI in Docker Desktop.
.DESCRIPTION
  Mirrors the justfile tasks and automates first-run setup (mkcert, hosts file, .env secrets).
  Run `.\scripts\dev.ps1 help` for usage. Works in Windows PowerShell 5.1 and PowerShell 7.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'up', 'first-run', 'secret', 'status', 'logs', 'down', 'html5', 'meeting')]
    [string]$Command = 'help',
    [switch]$PythonOnly,
    [switch]$Lite,
    [switch]$NoOpen,
    [string]$Name = 'Lecturer',
    [ValidateSet('MODERATOR', 'VIEWER')]
    [string]$Role = 'MODERATOR',
    [switch]$DryRun,
    [switch]$Yes,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$script:EnvFile = Join-Path $script:RepoRoot '.env'
$script:EnvExample = Join-Path $script:RepoRoot '.env.example'
$script:HostsFile = Join-Path $env:SystemRoot 'System32\drivers\etc\hosts'
# Docker Desktop installs per machine (Program Files) or per user (LocalAppData); first hit wins.
$script:DockerDesktopExe = @(
    (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe')
) | Where-Object { Test-Path $_ } | Select-Object -First 1
$script:ComposeFileValue = 'docker-compose.yml:docker-compose.sidecar.yml'
# -Lite adds the low-memory override (about 6 GB machines): no media, notes or recording.
if ($Lite) { $script:ComposeFileValue += ':docker-compose.lite.yml' }

# ---------------------------------------------------------------- output helpers

function Write-Info([string]$Text) { Write-Host $Text }
function Write-Warn([string]$Text) { Write-Host "WARNING: $Text" -ForegroundColor Yellow }
function Write-Ok([string]$Text) { Write-Host $Text -ForegroundColor Green }

function Write-Step([string]$Text) {
    if ($DryRun) { Write-Host "[dry-run] $Text" -ForegroundColor Cyan }
    else { Write-Host "==> $Text" -ForegroundColor Cyan }
}

# Run $Action unless -DryRun (then only print the description).
function Invoke-Step([string]$Description, [scriptblock]$Action) {
    Write-Step $Description
    if (-not $DryRun) { & $Action }
}

function Confirm-Action([string]$Question) {
    if ($Yes -or $DryRun) { return $true }
    $answer = Read-Host "$Question [y/N]"
    return ($answer -match '^(y|yes)$')
}

# ---------------------------------------------------------------- native command helpers

# Capture output and exit code of a native command without letting stderr throw.
function Invoke-Native([string]$File, [string[]]$Arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $File @Arguments 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
    return [pscustomobject]@{ ExitCode = $code; Output = @($output) }
}

# Run a native command with output going straight to the console. Returns the exit code.
function Invoke-Live([string]$File, [string[]]$Arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Out-Host keeps native stdout out of the function's return value (only the exit code is returned).
        & $File @Arguments | Out-Host
        return [int]$LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
}

function Get-ComposePrefix {
    if ($PythonOnly) { return @('compose', '-f', 'docker-compose.yml') }
    return @('compose')
}

function Invoke-Compose([string[]]$Arguments) {
    return Invoke-Native 'docker' (@(Get-ComposePrefix) + $Arguments)
}

function Invoke-ComposeLive([string[]]$Arguments) {
    return Invoke-Live 'docker' (@(Get-ComposePrefix) + $Arguments)
}

# In -PythonOnly mode force the edusync profile and ignore COMPOSE_FILE from .env; restore afterwards.
function Invoke-InMode([scriptblock]$Block) {
    if (-not $PythonOnly) { & $Block; return }
    $oldProfiles = [Environment]::GetEnvironmentVariable('COMPOSE_PROFILES', 'Process')
    $oldFile = [Environment]::GetEnvironmentVariable('COMPOSE_FILE', 'Process')
    try {
        [Environment]::SetEnvironmentVariable('COMPOSE_PROFILES', 'edusync', 'Process')
        [Environment]::SetEnvironmentVariable('COMPOSE_FILE', $null, 'Process')
        & $Block
    }
    finally {
        [Environment]::SetEnvironmentVariable('COMPOSE_PROFILES', $oldProfiles, 'Process')
        [Environment]::SetEnvironmentVariable('COMPOSE_FILE', $oldFile, 'Process')
    }
}

function Test-DockerInstalled { return [bool](Get-Command docker -ErrorAction SilentlyContinue) }

function Test-DockerDaemon {
    if (-not (Test-DockerInstalled)) { return $false }
    return ((Invoke-Native 'docker' @('info')).ExitCode -eq 0)
}

# ---------------------------------------------------------------- .env helpers

function Get-EnvValue([string]$Name) {
    foreach ($file in @($script:EnvFile, $script:EnvExample)) {
        if (-not (Test-Path $file)) { continue }
        foreach ($line in [IO.File]::ReadAllLines($file)) {
            if ($line -match "^\s*$([regex]::Escape($Name))=(.*)$") {
                return ($Matches[1] -replace '\s+#.*$', '').Trim()
            }
        }
    }
    return $null
}

# Edit an active line, else uncomment a commented one, else append. Logs only the name.
function Set-EnvValue([string]$Name, [string]$Value) {
    Write-Step "Set $Name in .env"
    if ($DryRun) { return }
    $lines = New-Object System.Collections.Generic.List[string]
    if (Test-Path $script:EnvFile) { $lines.AddRange([string[]][IO.File]::ReadAllLines($script:EnvFile)) }
    $escaped = [regex]::Escape($Name)
    $index = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$escaped=") { $index = $i; break }
    }
    if ($index -lt 0) {
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match "^\s*#\s*$escaped=") { $index = $i; break }
        }
    }
    if ($index -ge 0) { $lines[$index] = "$Name=$Value" } else { $lines.Add("$Name=$Value") }
    [IO.File]::WriteAllLines($script:EnvFile, $lines.ToArray(), (New-Object Text.UTF8Encoding $false))
}

function Get-Hostname {
    $name = Get-EnvValue 'BBB_HOSTNAME'
    if ([string]::IsNullOrWhiteSpace($name)) { return 'bbb.test' }
    # The name ends up in an elevated command and a certificate, so accept hostnames only.
    if ($name -notmatch '^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$') { throw "BBB_HOSTNAME '$name' is not a valid hostname." }
    return $name
}

# Python-only: published port on localhost. With BBB: through BBB's nginx (docker/bbb/edusync.nginx).
function Get-EdusyncBaseUrl {
    if ($PythonOnly) { return "http://localhost:$(Get-EdusyncPort)" }
    return "https://$(Get-Hostname)/edusync"
}

function Get-EdusyncPort {
    $port = Get-EnvValue 'EDUSYNC_PORT'
    if ([string]::IsNullOrWhiteSpace($port)) { return '8100' }
    return $port
}

# ---------------------------------------------------------------- setup steps

function Assert-DockerInstalled {
    if (Test-DockerInstalled) { return }
    $msg = "docker was not found on PATH. Install Docker Desktop (WSL2 backend), then reopen PowerShell:`n" +
        "  winget install Docker.DockerDesktop`n  https://www.docker.com/products/docker-desktop/"
    if ($DryRun) { Write-Warn $msg; return }
    throw $msg
}

function Wait-DockerDaemon {
    if (Test-DockerDaemon) { return }
    if (-not (Test-DockerInstalled)) { return }
    Write-Warn 'The Docker daemon is not reachable (Docker Desktop is not running).'
    if (-not $script:DockerDesktopExe) {
        if ($DryRun) { return }
        throw "Start Docker Desktop manually, wait until it says 'Engine running', then re-run this command."
    }
    if (-not (Confirm-Action 'Start Docker Desktop now?')) {
        throw 'Docker Desktop must be running. Start it and re-run this command.'
    }
    Invoke-Step 'Start Docker Desktop' { Start-Process -FilePath $script:DockerDesktopExe }
    if ($DryRun) { return }
    Write-Host 'Waiting for the Docker engine (up to 3 minutes)' -NoNewline
    $deadline = (Get-Date).AddMinutes(3)
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerDaemon) { Write-Host ''; Write-Ok 'Docker engine is running.'; return }
        Write-Host '.' -NoNewline
        Start-Sleep -Seconds 5
    }
    Write-Host ''
    throw 'Docker did not become ready in time. Open Docker Desktop, fix any prompts, and re-run.'
}

function Initialize-EnvFile {
    if (Test-Path $script:EnvFile) { Write-Info '.env already exists (left untouched).'; return }
    Invoke-Step 'Copy .env.example to .env' { Copy-Item $script:EnvExample $script:EnvFile }
}

function Initialize-ComposeSettings {
    Set-EnvValue 'COMPOSE_PATH_SEPARATOR' ':'
    Set-EnvValue 'COMPOSE_FILE' $script:ComposeFileValue
}

function Install-Mkcert {
    if (Get-Command mkcert -ErrorAction SilentlyContinue) { return }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        if ($DryRun) { Write-Warn 'mkcert and winget not found; install mkcert manually.'; return }
        throw 'mkcert not found and winget is unavailable. Install mkcert from https://github.com/FiloSottile/mkcert'
    }
    if (-not (Confirm-Action 'mkcert is missing. Install it with winget?')) {
        throw 'mkcert is required for HTTPS (mic/camera). Install it and re-run, or use -PythonOnly.'
    }
    Invoke-Step 'winget install --id FiloSottile.mkcert -e' {
        [void](Invoke-Live 'winget' @('install', '--id', 'FiloSottile.mkcert', '-e'))
    }
    if ($DryRun) { return }
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
    if (-not (Get-Command mkcert -ErrorAction SilentlyContinue)) {
        throw 'mkcert was installed but is not on PATH yet. Close this PowerShell window, open a new one and re-run setup.'
    }
}

function New-DevCertificates {
    $hostName = Get-Hostname
    $certDir = Join-Path $script:RepoRoot '.dev\certs'
    $full = Join-Path $certDir 'fullchain.pem'
    $key = Join-Path $certDir 'privkey.pem'
    Invoke-Step 'mkcert -install (trust the local CA; Windows may show a prompt)' {
        $r = Invoke-Native 'mkcert' @('-install')
        if ($r.ExitCode -ne 0) { throw "mkcert -install failed: $($r.Output -join ' ')" }
    }
    Invoke-Step 'Create .dev\certs' { [void](New-Item -ItemType Directory -Force -Path $certDir) }
    if ((Test-Path $full) -and (Test-Path $key)) { Write-Info 'Certificates already exist (kept).'; return }
    Invoke-Step "mkcert certificate for $hostName into .dev\certs" {
        $r = Invoke-Native 'mkcert' @('-cert-file', $full, '-key-file', $key, $hostName)
        if ($r.ExitCode -ne 0) { throw "mkcert failed: $($r.Output -join ' ')" }
    }
}

function Test-Elevated {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return (New-Object Security.Principal.WindowsPrincipal $id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-HostsEntry([string]$HostName) {
    if (-not (Test-Path $script:HostsFile)) { return $false }
    $pattern = '^\s*127\.0\.0\.1\s+(.*\s)?' + [regex]::Escape($HostName) + '(\s|$)'
    foreach ($line in [IO.File]::ReadAllLines($script:HostsFile)) {
        if ($line -match $pattern) { return $true }
    }
    return $false
}

function Add-HostsEntry {
    $hostName = Get-Hostname
    if (Test-HostsEntry $hostName) { Write-Info "hosts file already maps $hostName."; return }
    if (-not (Confirm-Action "Add '127.0.0.1 $hostName' to $script:HostsFile (needs administrator)?")) {
        Write-Warn "Skipped. Add the line manually from an elevated editor, or https://$hostName will not resolve."
        return
    }
    $append = "[IO.File]::AppendAllText('$script:HostsFile', [Environment]::NewLine + '127.0.0.1 $hostName' + [Environment]::NewLine)"
    if (Test-Elevated) {
        Invoke-Step "Append 127.0.0.1 $hostName to hosts" { Invoke-Expression $append }
        return
    }
    Invoke-Step "Append 127.0.0.1 $hostName to hosts (elevated PowerShell, accept the UAC prompt)" {
        $argText = "-NoProfile -Command `"$append`""
        Start-Process -FilePath 'powershell' -Verb RunAs -ArgumentList $argText -Wait
        if (-not (Test-HostsEntry $hostName)) { Write-Warn 'The hosts entry was not added (UAC declined?).' }
    }
}

function Get-WslMemoryGb {
    $path = Join-Path $env:USERPROFILE '.wslconfig'
    if (-not (Test-Path $path)) { return $null }
    foreach ($line in [IO.File]::ReadAllLines($path)) {
        if ($line -match '^\s*memory\s*=\s*([0-9.]+)\s*([A-Za-z]*)') {
            $n = [double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture)
            switch -Regex ($Matches[2].ToUpper()) {
                '^(MB|M)$' { return $n / 1024 }
                '^(KB|K)$' { return $n / 1048576 }
                '^(GB|G|)$' { return $n }
                default { return $n / 1073741824 }
            }
        }
    }
    return $null
}

function Test-ResourceLimits {
    $gb = Get-WslMemoryGb
    $needGb = 8
    if ($Lite) { $needGb = 3 }
    if ($Lite -and ($null -eq $gb -or $gb -lt $needGb)) {
        Write-Warn 'WSL2 memory is capped low. Lite mode needs about 3.5 GB for Docker; on a 6 GB PC use:'
        Write-Info "    [wsl2]`n    memory=3584MB`n    swap=4GB"
        Write-Info 'Then run `wsl --shutdown` and restart Docker Desktop.'
    }
    elseif (-not $Lite -and ($null -eq $gb -or $gb -lt $needGb)) {
        Write-Warn 'WSL2 memory is capped low (default is about half of RAM); BigBlueButton needs 8 GB or more.'
        Write-Info "Create $(Join-Path $env:USERPROFILE '.wslconfig') with:"
        Write-Info "    [wsl2]`n    memory=10GB`n    swap=4GB"
        Write-Info 'Then run `wsl --shutdown` and restart Docker Desktop.'
    }
    $freeGb = [math]::Round(([IO.DriveInfo]'C').AvailableFreeSpace / 1GB, 1)
    if ($freeGb -lt 40) {
        Write-Warn "C: has only $freeGb GB free. Move Docker's disk image to another drive:"
        Write-Info '    Docker Desktop > Settings > Resources > Advanced > Disk image location'
    }
}

function Invoke-Setup {
    Assert-DockerInstalled
    Wait-DockerDaemon
    Initialize-EnvFile
    if ($PythonOnly) {
        Write-Info '-PythonOnly: .env is not modified; use `.\scripts\dev.ps1 up -PythonOnly` to run the Python API only.'
    }
    else {
        Initialize-ComposeSettings
        Install-Mkcert
        New-DevCertificates
        Add-HostsEntry
    }
    Test-ResourceLimits
    Write-Ok 'Setup finished. Next steps:'
    if ($PythonOnly) { Write-Info '  .\scripts\dev.ps1 up -PythonOnly' }
    else {
        Write-Info '  .\scripts\dev.ps1 first-run   (starts BBB + EduSync, waits for BBB, stores secrets in .env)'
        Write-Info '  .\scripts\dev.ps1 status'
    }
}

# ---------------------------------------------------------------- run-time tasks

function Write-Urls {
    if ($PythonOnly) { Write-Info "EduSync API docs: $(Get-EdusyncBaseUrl)/docs"; return }
    Write-Info "BigBlueButton:    https://$(Get-Hostname)"
    Write-Info "EduSync API docs: $(Get-EdusyncBaseUrl)/docs"
}

function Invoke-Up {
    Invoke-InMode {
        Write-Step 'docker compose up -d --build'
        if (-not $DryRun) {
            if ((Invoke-ComposeLive @('up', '-d', '--build')) -ne 0) { throw 'docker compose up failed.' }
        }
    }
    Write-Urls
}

function Get-BbbSecret {
    $r = Invoke-Compose @('exec', '-T', 'bbb', 'bbb-conf', '--secret')
    if ($r.ExitCode -ne 0) { return $null }
    $url = $null; $secret = $null
    foreach ($line in $r.Output) {
        if ($line -match '^\s*URL:\s*(\S+)') { $url = $Matches[1] }
        if ($line -match '^\s*Secret:\s*(\S+)') { $secret = $Matches[1] }
    }
    if (-not $secret) { return $null }
    return [pscustomobject]@{ Url = $url; Secret = $secret }
}

function Get-HasuraSecret {
    $cmd = "grep '^HASURA_GRAPHQL_ADMIN_SECRET=' /etc/default/bbb-graphql-server-admin-pass | cut -d= -f2-"
    $r = Invoke-Compose @('exec', '-T', 'bbb', 'sh', '-c', $cmd)
    if ($r.ExitCode -eq 0) {
        $value = (($r.Output -join '') -replace '\s', '')
        if ($value) { return $value }
    }
    return 'bigbluebutton'
}

function Wait-BbbSecret {
    Write-Host 'Waiting for BigBlueButton to finish booting (up to 15 minutes)' -NoNewline
    $deadline = (Get-Date).AddMinutes(15)
    while ((Get-Date) -lt $deadline) {
        $found = Get-BbbSecret
        if ($found) { Write-Host ''; return $found }
        Write-Host '.' -NoNewline
        Start-Sleep -Seconds 10
    }
    Write-Host ''
    Write-Warn 'BigBlueButton did not become ready in time.'
    Write-Info '  Inspect:   docker compose logs --tail 50 bbb'
    Write-Info '  Fallback:  .\scripts\dev.ps1 up -PythonOnly   (Python API only)'
    throw 'bbb-conf --secret never succeeded.'
}

function Get-Health {
    try {
        return Invoke-RestMethod -Uri "$(Get-EdusyncBaseUrl)/health" -TimeoutSec 5 -UseBasicParsing
    }
    catch { return $null }
}

function Get-BbbState($Health) {
    if ($null -ne $Health -and $Health.PSObject.Properties['bbb']) { return [string]$Health.bbb }
    return 'unknown'
}

function Wait-Health {
    Write-Host 'Waiting for EduSync /health' -NoNewline
    $deadline = (Get-Date).AddMinutes(3)
    while ((Get-Date) -lt $deadline) {
        $h = Get-Health
        if ($null -ne $h) { Write-Host ''; return $h }
        Write-Host '.' -NoNewline
        Start-Sleep -Seconds 3
    }
    Write-Host ''
    return $null
}

function Invoke-FirstRun {
    Invoke-Up
    if ($PythonOnly) {
        Write-Info 'Python only: no BBB secrets are fetched. /health should report "bbb": "unconfigured".'
        return
    }
    if ($DryRun) {
        Write-Step 'Poll bbb-conf --secret until BBB is ready'
        Set-EnvValue 'EDUSYNC_BBB_SECRET' '<from bbb-conf>'
        Set-EnvValue 'EDUSYNC_BBB_URL' '<from bbb-conf>'
        Set-EnvValue 'EDUSYNC_HASURA_ADMIN_SECRET' '<from the bbb container>'
        Write-Step 'docker compose up -d --force-recreate edusync'
        Write-Step "Poll $(Get-EdusyncBaseUrl)/health"
        return
    }
    $found = Wait-BbbSecret
    $base = if ($found.Url) { $found.Url.TrimEnd('/') } else { "https://$(Get-Hostname)/bigbluebutton" }
    Set-EnvValue 'EDUSYNC_BBB_SECRET' $found.Secret
    Set-EnvValue 'EDUSYNC_BBB_URL' "$base/api"
    Set-EnvValue 'EDUSYNC_HASURA_ADMIN_SECRET' (Get-HasuraSecret)
    Write-Step 'docker compose up -d --force-recreate edusync'
    if ((Invoke-ComposeLive @('up', '-d', '--force-recreate', 'edusync')) -ne 0) { throw 'Could not recreate edusync.' }
    $health = Wait-Health
    if ($null -eq $health) { Write-Warn 'EduSync did not answer; see: docker compose logs edusync' }
    else { Write-Ok "EduSync is up; bbb: $(Get-BbbState $health)" }
    Write-Urls
}

function Invoke-Secret {
    $found = Get-BbbSecret
    if (-not $found) {
        throw 'Could not read the secret. Is the bbb container running and fully booted? Try: .\scripts\dev.ps1 status'
    }
    Write-Info "URL:    $($found.Url)"
    Write-Info "Secret: $($found.Secret)"
}

function Invoke-Status {
    if (-not (Test-DockerInstalled)) { Write-Warn 'docker is not installed or not on PATH. Run: .\scripts\dev.ps1 setup'; return }
    if (-not (Test-DockerDaemon)) {
        Write-Warn 'Docker Desktop is not running, so there is nothing to show yet. Start it (or run `setup`), then retry.'
        return
    }
    [void](Invoke-ComposeLive @('ps'))
    $health = Get-Health
    if ($null -eq $health) { Write-Info "EduSync /health: not reachable on port $(Get-EdusyncPort) (not started yet?)." }
    else { Write-Info "EduSync /health: ok, bbb: $(Get-BbbState $health)" }
}

function Invoke-Logs {
    $extra = @()
    if ($Rest) { $extra = @($Rest) }
    [void](Invoke-ComposeLive (@('logs', '-f', '--tail', '100') + $extra))
}

function Invoke-Down { [void](Invoke-ComposeLive @('down')) }

function Get-Sha1Hex([string]$Text) {
    $hash = [Security.Cryptography.SHA1]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($Text))
    return (($hash | ForEach-Object { $_.ToString('x2') }) -join '')
}

# BBB API URL for $Call; the checksum is sha1(call + query + shared secret).
function Get-BbbApiUrl([string]$Call, [string]$Query, [string]$Secret) {
    $sum = Get-Sha1Hex ($Call + $Query + $Secret)
    return "https://$(Get-Hostname)/bigbluebutton/api/${Call}?$Query&checksum=$sum"
}

# BigBlueButton has no login page: create a meeting through its API and open the join link.
function Invoke-Meeting {
    if ($PythonOnly) { throw 'meeting needs BigBlueButton; do not use -PythonOnly.' }
    $secret = Get-EnvValue 'EDUSYNC_BBB_SECRET'
    if ([string]::IsNullOrWhiteSpace($secret)) {
        throw 'EDUSYNC_BBB_SECRET is empty in .env. Run: .\scripts\dev.ps1 first-run'
    }
    $createQuery = 'allowStartStopRecording=false&attendeePW=ap&autoStartRecording=false' +
        '&meetingID=edusync-demo&moderatorPW=mp&name=EduSync+Demo&record=false'
    $password = if ($Role -eq 'MODERATOR') { 'mp' } else { 'ap' }
    $joinQuery = "fullName=$([uri]::EscapeDataString($Name))&meetingID=edusync-demo&password=$password&redirect=true"
    Write-Step "Create meeting 'EduSync Demo' (id edusync-demo) and join as $Name ($Role)"
    if ($DryRun) { return }
    try { $created = Invoke-RestMethod -Uri (Get-BbbApiUrl 'create' $createQuery $secret) -TimeoutSec 30 }
    catch { throw "BigBlueButton API not reachable ($($_.Exception.Message)). Is it running? Try: .\scripts\dev.ps1 status" }
    $code = [string]$created.response.returncode
    if ($code -ne 'SUCCESS') { throw "create failed: $($created.response.messageKey) $($created.response.message)" }
    $joinUrl = Get-BbbApiUrl 'join' $joinQuery $secret
    Write-Ok "Meeting ready. Join link (valid while the meeting runs, roughly 5 minutes if nobody joins):"
    Write-Info $joinUrl
    if (-not $NoOpen) { Start-Process $joinUrl }
}

function Invoke-Html5 {
    $inner = 'cd ~/src/bigbluebutton-html5 && ./run-dev.sh'
    [void](Invoke-ComposeLive @('exec', '-u', 'bigbluebutton', 'bbb', 'bash', '-lc', $inner))
}

function Show-Help {
    @'
EduSync AI / BigBlueButton Docker helper (PowerShell only)

Usage:  .\scripts\dev.ps1 <command> [-PythonOnly] [-DryRun] [-Yes] [service...]

Commands:
  setup       Check Docker, create .env, mkcert certs, hosts entry, resource warnings
  up          docker compose up -d --build
  first-run   up, wait for BBB, store BBB/Hasura secrets in .env, restart edusync
  secret      Print the BBB URL and shared secret (bbb-conf --secret)
  status      docker compose ps and the EduSync /health result
  logs        Follow logs; optional service name, e.g.  logs bbb
  down        docker compose down
  html5       Start the BBB HTML5 dev server (run-dev.sh) in the bbb container
  meeting     Create a demo meeting and open the join link in your browser (BBB has no login page)

Options:
  -PythonOnly Run only the Python API (profile edusync), no BBB
  -Lite       With `setup`: low-memory BBB for ~6 GB PCs (no audio/video, notes or recording)
  -Name <n>   With `meeting`: your display name (default Lecturer)
  -Role <r>   With `meeting`: MODERATOR (default) or VIEWER
  -NoOpen     With `meeting`: only print the join link
  -DryRun     Print what would happen without changing anything
  -Yes        Skip confirmation prompts

Examples:
  .\scripts\dev.ps1 setup
  .\scripts\dev.ps1 first-run
  .\scripts\dev.ps1 up -PythonOnly
  .\scripts\dev.ps1 setup -Lite        (then first-run; re-run `setup` without -Lite to switch back)
  .\scripts\dev.ps1 meeting             (join as moderator; add -Name Alice -Role VIEWER for a second user)
  .\scripts\dev.ps1 status
  .\scripts\dev.ps1 setup -DryRun
'@ | Write-Host
}

# ---------------------------------------------------------------- main

Push-Location $script:RepoRoot
try {
    switch ($Command) {
        'help' { Show-Help }
        'setup' { Invoke-Setup }
        'up' { Invoke-Up }
        'first-run' { Invoke-FirstRun }
        'secret' { Invoke-Secret }
        'status' { Invoke-Status }
        'logs' { Invoke-Logs }
        'down' { Invoke-Down }
        'html5' { Invoke-Html5 }
        'meeting' { Invoke-Meeting }
    }
}
catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally { Pop-Location }
