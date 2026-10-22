@echo off
rem Double-click to end the demo BigBlueButton meeting (BBB then saves the recording).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev.ps1" end
echo.
pause
