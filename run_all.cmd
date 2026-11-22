@echo off
rem All tasks on Windows: creates .venv311 (Python 3.11) with dependencies if missing, then runs run_all.py.
rem Example: run_all.cmd --n-trials 1 --epochs 1
rem ASCII only: cmd.exe misparses UTF-8 batch files.
setlocal
cd /d "%~dp0"
set "PY=.venv311\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"

if exist "%PY%" goto check
rem pyg-lib has no wheels for Python 3.14, so exactly 3.11.
rem Any 3.11 the py launcher knows (python.org, uv/Astral, ...); lines look like "-V:Tag[ *] path".
set "BASE="
for /f "tokens=1,2,3" %%a in ('py -0p 2^>nul ^| findstr /c:"3.11"') do if "%%c"=="" (set "BASE=%%b") else (set "BASE=%%c")
if not defined BASE (
    echo Python 3.11 not found. Install it from python.org with "py launcher" enabled and rerun.
    pause
    exit /b 2
)
echo === Creating .venv311 from %BASE%
"%BASE%" -m venv .venv311 || goto fail

:check
"%PY%" -c "import torch, pyg_lib, torch_geometric, torch_pointcloud, optuna" >nul 2>&1
if not errorlevel 1 goto run
rem torch and pyg-lib must match: CUDA build if an NVIDIA driver is present, else CPU.
set "BUILD=cpu"
where nvidia-smi >nul 2>&1 && set "BUILD=cu128"
echo === Installing dependencies (torch 2.8.0+%BUILD%)
"%PY%" -m pip install --upgrade pip || goto fail
"%PY%" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/%BUILD% || goto fail
"%PY%" -m pip install pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+%BUILD%.html || goto fail
"%PY%" -m pip install torch-geometric torch-pointcloud==0.0.16 optuna numpy || goto fail

:run
"%PY%" run_all.py %*
set "CODE=%errorlevel%"
echo === Finished with code %CODE%
pause
exit /b %CODE%

rem Keeps a double-clicked window open so the error can be read.
:fail
echo === Setup failed
pause
exit /b 1
