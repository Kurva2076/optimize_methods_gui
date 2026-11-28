"""Проверка и подготовка отдельного Python 3.11 для ML-ядра."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from studio_context import StudioContext
from studio_processes import ManagedProcess


@dataclass(frozen=True, slots=True)
class EnvironmentStatus:
    python: str | None
    python_version: str | None
    torch: str | None
    cuda: bool | None
    numpy: str | None
    optuna: str | None
    torch_pointcloud: str | None
    ready: bool
    error: str | None = None


def environment_python(project_root: Path) -> Path:
    if os.name == 'nt':
        return project_root / '.venv311' / 'Scripts' / 'python.exe'
    return project_root / '.venv311' / 'bin' / 'python'


def hidden_process_creation_flags() -> int:
    """Не создавать консоль для коротких диагностических процессов Windows."""
    return (
        getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        if os.name == 'nt'
        else 0
    )


def inspect_environment(project_root: str | Path, timeout: float = 12.0) -> EnvironmentStatus:
    root = Path(project_root).expanduser().resolve()
    python = environment_python(root)
    if not python.is_file():
        return EnvironmentStatus(None, None, None, None, None, None, None, False, '.venv311 не найдена')
    script = (
        'import json,sys\n'
        'out={"python":sys.version.split()[0]}\n'
        'for name in ("numpy","torch","optuna","torch_pointcloud"):\n'
        ' try:\n'
        '  module=__import__(name); out[name]=getattr(module,"__version__","installed")\n'
        ' except Exception as error: out[name]=None; out[name+"_error"]=str(error)\n'
        'try:\n import torch; out["cuda"]=bool(torch.cuda.is_available())\n'
        'except Exception: out["cuda"]=None\n'
        'print(json.dumps(out))'
    )
    try:
        completed = subprocess.run(
            [str(python), '-c', script],
            cwd=root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            check=False,
            creationflags=hidden_process_creation_flags(),
        )
        data = json.loads(completed.stdout.splitlines()[-1]) if completed.stdout else {}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return EnvironmentStatus(str(python), None, None, None, None, None, None, False, str(error))
    ready = completed.returncode == 0 and all(
        data.get(name) for name in ('numpy', 'torch', 'optuna', 'torch_pointcloud')
    )
    return EnvironmentStatus(
        str(python),
        data.get('python'),
        data.get('torch'),
        data.get('cuda'),
        data.get('numpy'),
        data.get('optuna'),
        data.get('torch_pointcloud'),
        ready,
        None if ready else (completed.stderr.strip() or 'не все ML-пакеты установлены'),
    )


def bundled_file(name: str) -> Path:
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
    return base / name


def environment_setup_command(
    project_root: str | Path,
    *,
    force: bool = False,
) -> tuple[str, list[str]]:
    """Команда PowerShell без shell=True; подходит для QProcess."""
    if os.name != 'nt':
        raise RuntimeError('автоустановка ML-окружения сейчас поддерживается на Windows')
    script = bundled_file('setup_ml_environment.ps1')
    if not script.is_file():
        raise FileNotFoundError(f'не найден встроенный скрипт: {script}')
    program = os.environ.get('SystemRoot', r'C:\Windows') + r'\System32\WindowsPowerShell\v1.0\powershell.exe'
    arguments = [
        '-NoProfile',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        str(script),
        '-ProjectRoot',
        str(Path(project_root).expanduser().resolve()),
    ]
    if force:
        arguments.append('-Force')
    return program, arguments


class EnvironmentInstaller(QObject):
    """Запуск установки в общем ProcessRegistry, чтобы лог был виден в Live."""

    started = Signal(object)
    finished = Signal(bool, str)

    def __init__(self, context: StudioContext, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self.process: ManagedProcess | None = None

    def start(self, *, force: bool = False) -> ManagedProcess:
        root = self.context.project_root
        if root is None:
            raise RuntimeError('сначала откройте проект')
        program, arguments = environment_setup_command(root, force=force)
        process = self.context.processes.create(
            'Установка ML-окружения',
            program,
            arguments,
            root,
        )
        process.finished.connect(self._finished)
        process.start()
        self.process = process
        self.started.emit(process)
        return process

    def _finished(self, code: int) -> None:
        root = self.context.project_root
        status = inspect_environment(root) if root is not None else None
        ready = code == 0 and status is not None and status.ready
        message = 'Окружение готово' if ready else f'Установка завершилась с кодом {code}'
        self.finished.emit(ready, message)
