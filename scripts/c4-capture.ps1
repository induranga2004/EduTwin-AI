<#
.SYNOPSIS
  Capture a BigBlueButton lecture (annotations + slides) into C4. Windows PowerShell.
.DESCRIPTION
  Run this BEFORE you start teaching and leave it running. It
    1. finds the running meeting,
    2. copies the slide images out of the BBB container and stores them in C4 with the slide text,
    3. every few seconds stores the annotations of the slide you are showing.
  BBB only exposes annotations of the CURRENT slide, so wait about 3 seconds after writing before
  changing slide. Press Ctrl+C after the last slide, BEFORE ending the meeting.
  Re-running is safe: events already stored are ignored.
.EXAMPLE
  .\scripts\c4-capture.ps1            # slides once, then capture until Ctrl+C
  .\scripts\c4-capture.ps1 -Once      # slides and one capture, then exit
#>
[CmdletBinding()]
param(
    [switch]$Once,
    [int]$IntervalSeconds = 2,
    [switch]$SkipSlides
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

function Invoke-Native([string[]]$CommandLine, [switch]$Quiet) {
    # Windows PowerShell 5.1 turns native stderr (docker progress text) into errors when
    # ErrorActionPreference is Stop, so rely on the exit code instead.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($Quiet) { $out = & $CommandLine[0] $CommandLine[1..($CommandLine.Length - 1)] 2>$null }
        else { $out = & $CommandLine[0] $CommandLine[1..($CommandLine.Length - 1)] }
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
    if ($code -ne 0) {
        # show what the command printed before it failed: that is where the reason is
        $out | ForEach-Object { Write-Host ("    " + $_) }
        throw "Command failed ($code): $($CommandLine -join ' ')"
    }
    return $out
}

Write-Host '==> Copy the capture helper into the edusync container'
Invoke-Native @('docker', 'compose', 'cp', 'scripts/c4_capture.py', 'edusync:/tmp/c4_capture.py') -Quiet | Out-Null

Write-Host '==> Find the running meeting'
$infoJson = Invoke-Native @('docker', 'compose', 'exec', '-T', 'edusync', 'python', '/tmp/c4_capture.py', '--info') -Quiet
$info = ($infoJson | Out-String) | ConvertFrom-Json
Write-Host ("    meeting {0} ({1}), presentation {2}, {3} slides" -f $info.meetingId, $info.extId, $info.presentationId, $info.pages)

$hostName = 'bbb.test'
$hostLine = Select-String -Path '.env' -Pattern '^BBB_HOSTNAME=' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($hostLine) { $hostName = ($hostLine.Line -split '=', 2)[1].Trim() }
$reportUrl = "https://$hostName/edusync/api/c4/lectures/$($info.meetingId)/report"
Write-Host "    report (open it any time, it updates as data arrives): $reportUrl"

if (-not $SkipSlides) {
    Write-Host '==> Copy slide images out of BigBlueButton and store them in C4'
    $m = $info.meetingId
    $tmp = Join-Path $env:TEMP ("c4slides_" + $m)
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    New-Item -ItemType Directory -Path $tmp | Out-Null
    $src = "bbb:/var/bigbluebutton/$m/$m/$($info.presentationId)/svgs"
    Invoke-Native @('docker', 'compose', 'cp', $src, $tmp) -Quiet | Out-Null
    # A fresh folder per run: files copied in by docker cp belong to root, so an old folder cannot be deleted.
    $dest = '/tmp/c4slides_' + (Get-Date -Format 'yyyyMMddHHmmss')
    Invoke-Native @('docker', 'compose', 'cp', (Join-Path $tmp 'svgs'), "edusync:$dest") -Quiet | Out-Null
    Invoke-Native @('docker', 'compose', 'exec', '-T', 'edusync', 'python', '/tmp/c4_capture.py', '--slides', $dest) | ForEach-Object { Write-Host ("    " + $_) }
    Remove-Item -Recurse -Force $tmp

    # Best effort: word positions let the report name what an underline or circle points at.
    Write-Host '==> Store slide word positions (best effort)'
    try { & (Join-Path $PSScriptRoot 'c4-words.ps1') -Lecture $m }
    catch { Write-Host "    skipped: $($_.Exception.Message)" }
}

Write-Host '==> Capture annotations (Ctrl+C when the lecture is finished, before ending the meeting)'
try {
    while ($true) {
        $line = Invoke-Native @('docker', 'compose', 'exec', '-T', 'edusync', 'python', '/tmp/c4_capture.py', '--once')
        Write-Host ("{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), ($line | Out-String).Trim())
        if ($Once) { break }
        Start-Sleep -Seconds $IntervalSeconds
    }
}
finally {
    Write-Host 'Capture stopped. Data is stored in C4; you can now end the meeting.'
    Write-Host "Report: $reportUrl"
}
