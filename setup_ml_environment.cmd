@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_ml_environment.ps1" -ProjectRoot "%CD%" %*
exit /b %errorlevel%
