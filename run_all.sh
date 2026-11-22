#!/usr/bin/env bash
# Все задачи (см. run_all.py). Пример: bash run_all.sh --n-trials 1 --epochs 1
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv311/Scripts/python.exe ]; then PY=.venv311/Scripts/python.exe
elif [ -x .venv311/bin/python ]; then PY=.venv311/bin/python
else PY=${PYTHON:-python3}; fi
exec "$PY" run_all.py "$@"
