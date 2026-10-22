<#
.SYNOPSIS
  Transcribe a lecture recording and attach the speech to the C4 lecture report. Windows PowerShell.
.DESCRIPTION
  Copies your recording (screen recording or audio file) into the edusync container, transcribes it
  with Whisper (runs locally, no data leaves your computer), and stores the timestamped words in C4.
  To line the voice up with the drawings, say "sync" at the very start of the lecture while you draw
  one small mark, then use -Sync. Or give the time yourself with -Offset.
  The first run downloads the speech model and needs internet: run  -Check  once beforehand.
.EXAMPLE
  .\scripts\c4-speech.ps1 -Check                                   # download and test the model
  .\scripts\c4-speech.ps1 -Recording D:\Videos\lecture.mp4 -Sync   # align with the word "sync"
  .\scripts\c4-speech.ps1 -Recording D:\Videos\lecture.mp4 -Offset 125.5
  .\scripts\c4-speech.ps1 -Recording "C:\Users\me\Videos\Captures\App 2026-10-22 18-51-22.mp4" -AutoOffset
      (uses the start time in the file name and the meeting start time: no spoken cue needed)
  .\scripts\c4-speech.ps1 -Recording lecture.mp4 -Sync -Model small -Lecture <meetingId>
  .\scripts\c4-speech.ps1 -Recording lecture.mp4 -Sync -Vocabulary 'throughput, latency'
  Optional: -Hint gives Whisper a word list from your slides, -Vocabulary adds your own terms.
  They are off by default because they did not improve accuracy in tests and can merge sentences.
#>
[CmdletBinding()]
param(
    [string]$Recording,
    [string]$Lecture,
    [switch]$Sync,
    [string]$SyncWord,
    [double]$Offset,
    [switch]$AutoOffset,
    [switch]$FromBbb,
    [string]$Model = 'small',
    [string]$Language = 'en',
    [string]$Vocabulary,
    [switch]$Hint,
    [switch]$Check
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

if (-not $Check -and -not $Recording -and -not $FromBbb) {
    throw 'Give -FromBbb (use the audio BigBlueButton recorded), -Recording <file>, or -Check to test the speech model.'
}
if ($FromBbb -and ($Recording -or $Sync -or $SyncWord -or $AutoOffset -or $PSBoundParameters.ContainsKey('Offset'))) {
    throw '-FromBbb needs no recording file and no offset: BBB audio is aligned by its own start time.'
}

function Get-BbbHost {
    $name = 'bbb.test'
    $line = Select-String -Path '.env' -Pattern '^BBB_HOSTNAME=' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($line) { $name = ($line.Line -split '=', 2)[1].Trim() }
    return $name
}
if ($Sync -and $SyncWord) { throw 'Use either -Sync or -SyncWord, not both.' }
if ($AutoOffset -and ($Sync -or $SyncWord -or $PSBoundParameters.ContainsKey('Offset'))) {
    throw 'Use only one of -AutoOffset, -Sync, -SyncWord or -Offset.'
}
if ($Sync) { $SyncWord = 'sync' }
$hasOffset = $PSBoundParameters.ContainsKey('Offset')
if ($hasOffset -and $SyncWord) { throw 'Use either -Offset or -Sync, not both.' }

Write-Host '==> Copy the speech helper into the edusync container'
Invoke-Native @('docker', 'compose', 'cp', 'scripts/c4_transcribe.py', 'edusync:/tmp/c4_transcribe.py') -Quiet | Out-Null

$py = @('docker', 'compose', 'exec', '-T', 'edusync', 'python', '/tmp/c4_transcribe.py')
if ($Check) {
    Write-Host '==> Load the speech model (first run downloads it; needs internet)'
    $args2 = $py + @('--check')
    if ($Model) { $args2 += @('--model', $Model) }
    Invoke-Native $args2 | ForEach-Object { Write-Host ("    " + $_) }
    return
}

if ($FromBbb) {
    if (-not $Lecture) {
        Write-Host '==> Find the newest lecture stored in C4'
        $lectures = @(Invoke-RestMethod -Uri ("https://{0}/edusync/api/c4/lectures" -f (Get-BbbHost)) -TimeoutSec 30)
        if ($lectures.Count -eq 0) { throw 'No lecture is stored in C4 yet. Run scripts\c4-capture.ps1 during the lecture.' }
        $Lecture = $lectures[0].lecture_id
    }
    Write-Host "    lecture $Lecture"
    Write-Host '==> Look for the microphone recording BigBlueButton saved'
    $rawDir = "/var/bigbluebutton/recording/raw/$Lecture/audio"
    $listing = Invoke-Native @('docker', 'compose', 'exec', '-T', 'bbb', 'bash', '-lc', "ls $rawDir 2>/dev/null") -Quiet
    $names = @($listing | Where-Object { $_ -match '\.(webm|opus|ogg)$' })
    if ($names.Count -eq 0) {
        throw ('BBB saved no audio for this meeting. Make sure the meeting was started with -Record, was ended ' +
            '(scripts\end-meeting.cmd), and your microphone was UNMUTED while you spoke.')
    }
    Write-Host "    found $($names.Count) audio file(s)"
    $tmp = Join-Path $env:TEMP ('c4bbb_' + $Lecture)
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    New-Item -ItemType Directory -Path $tmp | Out-Null
    $destDir = '/tmp/c4bbb_' + (Get-Date -Format 'yyyyMMddHHmmss')
    foreach ($n in $names) {
        Invoke-Native @('docker', 'compose', 'cp', "bbb:$rawDir/$n", (Join-Path $tmp $n)) -Quiet | Out-Null
    }
    Invoke-Native @('docker', 'compose', 'cp', $tmp, "edusync:$destDir") -Quiet | Out-Null
    Remove-Item -Recurse -Force $tmp
    Write-Host '==> Transcribe and store (this can take a few minutes)'
    $args2 = $py + @('--bbb-audio') + @($names | ForEach-Object { "$destDir/$_" }) + @('--language', $Language, '--lecture', $Lecture)
    if ($Model) { $args2 += @('--model', $Model) }
    $lines = Invoke-Native $args2
    $lines | ForEach-Object { Write-Host ("    " + $_) }
    $pathLine = $lines | Where-Object { $_ -like 'report-path:*' } | Select-Object -Last 1
    if ($pathLine) { Write-Host ("Report: https://{0}{1}" -f (Get-BbbHost), ($pathLine -replace '^report-path:\s*', '')) }
    return
}

$file = (Resolve-Path $Recording).Path
$startEpoch = $null
if ($AutoOffset) {
    # Screen recorders (Windows Game Bar, OBS) put the start time in the file name: YYYY-MM-DD HH-MM-SS
    $m = [regex]::Match((Split-Path $file -Leaf), '(\d{4})-(\d{2})-(\d{2}) (\d{2})-(\d{2})-(\d{2})')
    if (-not $m.Success) { throw 'The file name has no start time (YYYY-MM-DD HH-MM-SS). Use -Sync or -Offset instead.' }
    $local = [datetime]::ParseExact($m.Value, 'yyyy-MM-dd HH-mm-ss', [Globalization.CultureInfo]::InvariantCulture)
    $stamp = [DateTimeOffset]::new($local, [TimeZoneInfo]::Local.GetUtcOffset($local))
    $startEpoch = $stamp.ToUnixTimeMilliseconds() / 1000.0
    Write-Host "    recording start taken from the file name: $($m.Value) (this computer's time zone)"
}
$ext = [System.IO.Path]::GetExtension($file)
$dest = '/tmp/c4rec_' + (Get-Date -Format 'yyyyMMddHHmmss') + $ext
$sizeMb = [math]::Round((Get-Item $file).Length / 1MB, 1)
Write-Host "==> Copy the recording ($sizeMb MB) into the container"
Invoke-Native @('docker', 'compose', 'cp', $file, "edusync:$dest") -Quiet | Out-Null

Write-Host '==> Transcribe and store (this can take a few minutes)'
$args2 = $py + @('--file', $dest, '--language', $Language, '--source', (Split-Path $file -Leaf))
if ($Lecture) { $args2 += @('--lecture', $Lecture) }
if ($Model) { $args2 += @('--model', $Model) }
if ($Vocabulary) { $args2 += @('--vocabulary', $Vocabulary) }
if ($Hint) { $args2 += @('--hint') }
if ($startEpoch) { $args2 += @('--recording-start-epoch', ([string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0}', $startEpoch))) }
elseif ($hasOffset) { $args2 += @('--offset', [string]$Offset) }
elseif ($SyncWord) { $args2 += @('--sync-word', $SyncWord) }
$lines = Invoke-Native $args2
$lines | ForEach-Object { Write-Host ("    " + $_) }

$hostName = 'bbb.test'
$hostLine = Select-String -Path '.env' -Pattern '^BBB_HOSTNAME=' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($hostLine) { $hostName = ($hostLine.Line -split '=', 2)[1].Trim() }
$pathLine = $lines | Where-Object { $_ -like 'report-path:*' } | Select-Object -Last 1
if ($pathLine) { Write-Host ("Report: https://{0}{1}" -f $hostName, ($pathLine -replace '^report-path:\s*', '')) }
