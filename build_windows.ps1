$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    py -3.12 -m venv .venv
}

& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt pyinstaller

& $python -m PyInstaller `
    --clean `
    --noconfirm `
    --onefile `
    --windowed `
    --name PointCloudMLStudio `
    --add-data 'assets;assets' `
    --add-data 'hpo_core;hpo_core' `
    --add-data 'hpo_pointnet.py;.' `
    --add-data 'predict_scene.py;.' `
    --add-data 'run_all.py;.' `
    --add-data 'run_all.cmd;.' `
    --add-data 'run_all.ps1;.' `
    --add-data 'run_all.sh;.' `
    --add-data 'summary_report.py;.' `
    --add-data 'mlstudio_engine_checks.py;.' `
    --add-data 'setup_ml_environment.ps1;.' `
    --add-data 'setup_ml_environment.cmd;.' `
    --collect-all laspy `
    --collect-all lazrs `
    main.py

Write-Host ''
Write-Host 'Windows executable is ready:' (Join-Path $PSScriptRoot 'dist\PointCloudMLStudio.exe')
