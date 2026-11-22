# All tasks on Windows (PowerShell): creates .venv311 (Python 3.11) with dependencies if missing, then runs run_all.py.
# Example: powershell -ExecutionPolicy Bypass -File .\run_all.ps1 --n-trials 1 --epochs 1
# ASCII only: Windows PowerShell 5.1 misreads UTF-8 scripts without BOM.
# No $ErrorActionPreference='Stop': in 5.1 it turns any stderr of a native command into an exception.
trap { Write-Host "ERROR: $_"; Read-Host 'Press Enter to close'; exit 1 }
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv311\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'

function Invoke-Checked {
    & $args[0] @($args | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { throw "Failed ($LASTEXITCODE): $($args -join ' ')" }
}

if (-not (Test-Path $py)) {
    # pyg-lib has no wheels for Python 3.14, so exactly 3.11; any 3.11 the py launcher knows (python.org, uv/Astral).
    $base = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $line = py -0p 2>$null | Where-Object { $_ -match '3\.11' } | Select-Object -First 1
        if ($line -match '([A-Za-z]:\\.*python\.exe)') { $base = $Matches[1] }
    }
    if (-not $base) {
        Write-Host 'Python 3.11 not found. Install it from python.org with "py launcher" enabled and rerun.'
        Read-Host 'Press Enter to close'
        exit 2
    }
    Write-Host "=== Creating .venv311 from $base"
    Invoke-Checked $base -m venv .venv311
}

& $py -c 'import torch, pyg_lib, torch_geometric, torch_pointcloud, optuna' 2>$null
if ($LASTEXITCODE -ne 0) {
    # torch and pyg-lib must match: CUDA build if an NVIDIA driver is present, else CPU.
    $build = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { 'cu128' } else { 'cpu' }
    Write-Host "=== Installing dependencies (torch 2.8.0+$build)"
    Invoke-Checked $py -m pip install --upgrade pip
    Invoke-Checked $py -m pip install torch==2.8.0 --index-url "https://download.pytorch.org/whl/$build"
    Invoke-Checked $py -m pip install pyg-lib -f "https://data.pyg.org/whl/torch-2.8.0+$build.html"
    Invoke-Checked $py -m pip install torch-geometric torch-pointcloud==0.0.16 optuna numpy
}

& $py run_all.py @args
$code = $LASTEXITCODE
# Keeps a window started by double-click / "Run with PowerShell" open so the output can be read.
Read-Host "=== Finished with code $code. Press Enter to close"
exit $code
