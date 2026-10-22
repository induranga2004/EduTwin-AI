<#
.SYNOPSIS
  Store the position of every word on your slides in C4, so the report can name what an underline
  or circle points at. Windows PowerShell.
.DESCRIPTION
  Reads the lecture's PDF from the BigBlueButton container with `pdftotext -bbox` (word positions)
  and uploads them to C4. It runs automatically at the start of c4-capture.ps1; run it by hand for a
  lecture that is already captured. The PDF stays in BBB after the meeting ends.
.EXAMPLE
  .\scripts\c4-words.ps1                      # newest lecture stored in C4
  .\scripts\c4-words.ps1 -Lecture <meetingId>
#>
[CmdletBinding()]
param(
    [string]$Lecture
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

function Invoke-Native([string[]]$CommandLine, [switch]$Quiet) {
    # Windows PowerShell 5.1 turns native stderr into errors when ErrorActionPreference is Stop,
    # so rely on the exit code and show what the command printed when it fails.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($Quiet) { $out = & $CommandLine[0] $CommandLine[1..($CommandLine.Length - 1)] 2>$null }
        else { $out = & $CommandLine[0] $CommandLine[1..($CommandLine.Length - 1)] }
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
    if ($code -ne 0) {
        $out | ForEach-Object { Write-Host ("    " + $_) }
        throw "Command failed ($code): $($CommandLine -join ' ')"
    }
    return $out
}

function Get-BbbHost {
    $name = 'bbb.test'
    $line = Select-String -Path '.env' -Pattern '^BBB_HOSTNAME=' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($line) { $name = ($line.Line -split '=', 2)[1].Trim() }
    return $name
}

if (-not $Lecture) {
    # Windows PowerShell 5.1 hands a JSON array back as ONE object; ForEach-Object unrolls it.
    $lectures = @(Invoke-RestMethod -Uri ("https://{0}/edusync/api/c4/lectures" -f (Get-BbbHost)) -TimeoutSec 30 | ForEach-Object { $_ })
    if ($lectures.Count -eq 0) { throw 'No lecture is stored in C4 yet. Run scripts\c4-capture.ps1 during the lecture.' }
    $Lecture = $lectures[0].lecture_id
}
Write-Host "==> Word positions for lecture $Lecture"

$listing = Invoke-Native @('docker', 'compose', 'exec', '-T', 'bbb', 'bash', '-lc',
    "ls /var/bigbluebutton/$Lecture/$Lecture/*/*.pdf 2>/dev/null | head -1") -Quiet
$pdf = ($listing | Select-Object -First 1)
if (-not $pdf) { throw "No slide PDF was found for this lecture inside BigBlueButton (/var/bigbluebutton/$Lecture)." }
$pdf = ([string]$pdf).Trim()

$remote = "/tmp/c4words_$Lecture.html"
Invoke-Native @('docker', 'compose', 'exec', '-T', 'bbb', 'bash', '-lc', "pdftotext -bbox '$pdf' '$remote'") -Quiet | Out-Null
$tmp = Join-Path $env:TEMP ("c4words_" + $Lecture)
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Path $tmp | Out-Null
Invoke-Native @('docker', 'compose', 'cp', "bbb:$remote", (Join-Path $tmp 'words.html')) -Quiet | Out-Null
Invoke-Native @('docker', 'compose', 'cp', 'scripts/c4_words.py', 'edusync:/tmp/c4_words.py') -Quiet | Out-Null
Invoke-Native @('docker', 'compose', 'cp', (Join-Path $tmp 'words.html'), 'edusync:/tmp/c4words.html') -Quiet | Out-Null
Remove-Item -Recurse -Force $tmp
Invoke-Native @('docker', 'compose', 'exec', '-T', 'edusync', 'python', '/tmp/c4_words.py', '--lecture', $Lecture, '--xml', '/tmp/c4words.html') |
    ForEach-Object { Write-Host ("    " + $_) }
