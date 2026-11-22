param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$VenvRoot = Join-Path $ProjectRoot '.venv311'
$Python = Join-Path $VenvRoot 'Scripts\python.exe'

function Invoke-Checked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Command)
    & $Command[0] @($Command | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $($Command -join ' ')"
    }
}

Write-Output "=== ML Studio environment: $ProjectRoot"
if (-not (Test-Path $Python)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Python Launcher (py.exe) not found. Install Python 3.11 from python.org with the launcher enabled.'
    }
    $Version = & py -3.11 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null
    if ($LASTEXITCODE -ne 0 -or $Version.Trim() -ne '3.11') {
        throw 'Python 3.11 not found. Install it from python.org and rerun setup.'
    }
    Write-Output '=== Creating .venv311'
    Invoke-Checked py -3.11 -m venv $VenvRoot
}

$Build = 'cpu'
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi | Out-Null
    if ($LASTEXITCODE -eq 0) { $Build = 'cu128' }
}

Write-Output "=== Installing PyTorch 2.8.0 ($Build)"
Invoke-Checked $Python -m pip install --upgrade pip
$PipMode = @()
if ($Force) { $PipMode += '--force-reinstall' }
Invoke-Checked $Python -m pip install @PipMode 'torch==2.8.0' --index-url "https://download.pytorch.org/whl/$Build"
Invoke-Checked $Python -m pip install @PipMode pyg-lib -f "https://data.pyg.org/whl/torch-2.8.0+$Build.html"
Invoke-Checked $Python -m pip install @PipMode 'torch-geometric' 'torch-pointcloud==0.0.16' 'optuna' 'numpy' 'scipy'

Write-Output '=== Verifying packages'
Invoke-Checked $Python -c "import torch,pyg_lib,torch_geometric,torch_pointcloud,optuna,numpy,scipy; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('optuna',optuna.__version__,'numpy',numpy.__version__)"

$Status = @{
    python = $Python
    build = $Build
    installed_at = (Get-Date).ToString('o')
}
$Status | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $ProjectRoot '.mlstudio-environment.json')
Write-Output '=== ML Studio environment is ready'
