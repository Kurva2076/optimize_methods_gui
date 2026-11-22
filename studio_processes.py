"""Запуск, безопасная остановка и разбор живых логов ML Studio."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

MAX_MEMORY_LOG_CHARS = 1_000_000
MAX_STRUCTURED_EVENTS = 20_000

F1_PATTERN = re.compile(
    r'F1\s*[=:]\s*(?P<f1>0(?:\.\d+)?|1(?:\.0+)?)',
    re.IGNORECASE,
)
EPOCH_PATTERN = re.compile(
    r'trial\s*=\s*(?P<trial>\d+)\s+'
    r'epoch\s*=\s*(?P<epoch>\d+).*?'
    r'F1\s*=\s*(?P<f1>\d+(?:\.\d+)?).*?'
    r'baseline\s*=\s*(?P<baseline>\d+(?:\.\d+)?)'
    r'(?:.*?(?:время|elapsed)\s*=\s*(?P<elapsed>[^\s]+))?',
    re.IGNORECASE,
)
TRIAL_PATTERN = re.compile(
    r'trial\s*=\s*(?P<trial>\d+)\s+'
    r'(?P<state>COMPLETE|PRUNED|FAIL(?:ED)?)'
    r'(?:\s+(?:за|in)\s+(?P<elapsed>[^\s]+))?',
    re.IGNORECASE,
)
SCENE_PATTERN = re.compile(
    r'(?:===\s*)?сцена\s+(?P<current>\d+)\s*/\s*(?P<total>\d+)\s*:\s*(?P<name>.+)',
    re.IGNORECASE,
)
PATCH_PATTERN = re.compile(
    r'патч\s+(?P<current>\d+)\s*/\s*(?P<total>\d+)'
    r'(?:\s+(?P<seconds>\d+(?:\.\d+)?)s)?',
    re.IGNORECASE,
)
PREDICTION_METRICS_PATTERN = re.compile(
    r'(?:===\s*)?F1\s*=\s*(?P<f1>\d+(?:\.\d+)?).*?'
    r'precision\s*=\s*(?P<precision>\d+(?:\.\d+)?).*?'
    r'recall\s*=\s*(?P<recall>\d+(?:\.\d+)?).*?'
    r'baseline\s*=\s*(?P<baseline>\d+(?:\.\d+)?)',
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EpochEvent:
    trial: int
    epoch: int
    f1: float
    baseline: float
    elapsed_seconds: float | None
    raw: str


@dataclass(frozen=True, slots=True)
class TrialEvent:
    trial: int
    state: str
    elapsed_seconds: float | None
    best_f1: float | None
    raw: str


@dataclass(frozen=True, slots=True)
class PredictionEvent:
    scene_index: int | None
    scene_total: int | None
    scene_name: str | None
    patch_index: int | None
    patch_total: int | None
    elapsed_seconds: float | None
    f1: float | None
    precision: float | None
    recall: float | None
    baseline: float | None
    raw: str


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    name: str
    status: str
    seconds: int
    last_f1: float | None
    command: tuple[str, ...]
    cwd: Path
    status_key: str = 'pending'
    last_baseline: float | None = None
    epoch_count: int = 0
    trial_count: int = 0
    prediction_count: int = 0
    log_path: Path | None = None
    kind: str = 'training'


class ManagedProcess(QObject):
    output_received = Signal(str)
    changed = Signal()
    finished = Signal(int)
    epoch_recorded = Signal(object)
    trial_recorded = Signal(object)
    prediction_recorded = Signal(object)

    def __init__(
        self,
        name: str,
        program: str,
        arguments: list[str],
        cwd: Path,
        parent: QObject | None = None,
        *,
        max_log_chars: int = MAX_MEMORY_LOG_CHARS,
    ) -> None:
        super().__init__(parent)
        self.name = name
        self.program = program
        self.arguments = list(arguments)
        self.cwd = Path(cwd)
        self.status = 'Ожидает запуска'
        self.status_key = 'pending'
        self.seconds = 0
        self.last_f1: float | None = None
        self.last_baseline: float | None = None
        self.log: list[str] = []
        self.epochs: list[EpochEvent] = []
        self.trials: list[TrialEvent] = []
        self.predictions: list[PredictionEvent] = []
        self.kind = 'prediction' if any('predict_scene.py' in arg for arg in self.arguments) else 'training'
        self.log_path: Path | None = None
        self._log_stream: TextIO | None = None
        self._log_chars = 0
        self._max_log_chars = max(1, int(max_log_chars))
        self._parse_buffer = ''
        self._stop_requested = False
        self._tree_pid = 0
        self._scene_index: int | None = None
        self._scene_total: int | None = None
        self._scene_name: str | None = None
        self._last_trial_f1: dict[int, float] = {}

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.cwd))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONIOENCODING', 'utf-8')
        environment.insert('PYTHONUNBUFFERED', '1')
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

    @property
    def command(self) -> tuple[str, ...]:
        return (self.program, *self.arguments)

    def start(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        self._reset_run_state()
        self._open_log()
        self.status = 'Запускается'
        self.status_key = 'starting'
        self.changed.emit()
        launch_program, launch_arguments = self._launch_command()
        self.process.start(launch_program, launch_arguments)

    def stop(self) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self._stop_requested = True
        self.status = 'Останавливается'
        self.status_key = 'stopping'
        self.changed.emit()
        self._terminate_tree()
        QTimer.singleShot(2500, self._kill_tree_if_running)

    def restart(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self._stop_requested = True
            self._kill_tree_if_running()
            self.process.waitForFinished(1500)
        self.start()

    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def snapshot(self) -> ProcessSnapshot:
        return ProcessSnapshot(
            name=self.name,
            status=self.status,
            seconds=self.seconds,
            last_f1=self.last_f1,
            command=self.command,
            cwd=self.cwd,
            status_key=self.status_key,
            last_baseline=self.last_baseline,
            epoch_count=len(self.epochs),
            trial_count=len(self.trials),
            prediction_count=len(self.predictions),
            log_path=self.log_path,
            kind=self.kind,
        )

    def log_text(self) -> str:
        """Ограниченный хвост лога для живого интерфейса."""
        return ''.join(self.log)

    def full_log_text(self) -> str:
        """Полный лог с диска; память GUI при этом не растёт."""
        if self._log_stream is not None:
            self._log_stream.flush()
        if self.log_path is not None and self.log_path.is_file():
            try:
                return self.log_path.read_text(encoding='utf-8', errors='replace')
            except OSError:
                pass
        return self.log_text()

    def ingest_output(self, text: str) -> None:
        """Записать и разобрать порцию stdout/stderr."""
        if not text:
            return
        self._write_full_log(text)
        self._append_memory_log(text)
        self._parse_buffer += text
        lines = self._parse_buffer.splitlines(keepends=True)
        self._parse_buffer = ''
        for line in lines:
            if line.endswith(('\n', '\r')):
                self._parse_line(line.rstrip('\r\n'))
            else:
                self._parse_buffer = line
        for match in F1_PATTERN.finditer(text):
            self.last_f1 = float(match.group('f1'))
        self.output_received.emit(text)
        self.changed.emit()

    def _reset_run_state(self) -> None:
        self.seconds = 0
        self.last_f1 = None
        self.last_baseline = None
        self.log.clear()
        self.epochs.clear()
        self.trials.clear()
        self.predictions.clear()
        self._last_trial_f1.clear()
        self._log_chars = 0
        self._parse_buffer = ''
        self._stop_requested = False
        self._tree_pid = 0
        self._scene_index = None
        self._scene_total = None
        self._scene_name = None

    def _launch_command(self) -> tuple[str, list[str]]:
        if os.name != 'nt':
            setsid = shutil.which('setsid')
            if setsid:
                return setsid, [self.program, *self.arguments]
        return self.program, list(self.arguments)

    def _started(self) -> None:
        self._tree_pid = int(self.process.processId())
        self.status = 'Выполняется'
        self.status_key = 'running'
        self.timer.start()
        self.changed.emit()

    def _tick(self) -> None:
        self.seconds += 1
        self.changed.emit()

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        self.ingest_output(text)

    def _parse_line(self, line: str) -> None:
        epoch = EPOCH_PATTERN.search(line)
        if epoch:
            event = EpochEvent(
                trial=int(epoch.group('trial')),
                epoch=int(epoch.group('epoch')),
                f1=float(epoch.group('f1')),
                baseline=float(epoch.group('baseline')),
                elapsed_seconds=_parse_duration(epoch.group('elapsed')),
                raw=line,
            )
            self.last_f1 = event.f1
            self.last_baseline = event.baseline
            self._last_trial_f1[event.trial] = event.f1
            _bounded_append(self.epochs, event)
            self.epoch_recorded.emit(event)
            return

        trial = TRIAL_PATTERN.search(line)
        if trial:
            number = int(trial.group('trial'))
            event = TrialEvent(
                trial=number,
                state=trial.group('state').upper(),
                elapsed_seconds=_parse_duration(trial.group('elapsed')),
                best_f1=self._last_trial_f1.get(number),
                raw=line,
            )
            _bounded_append(self.trials, event)
            self.trial_recorded.emit(event)
            return

        scene = SCENE_PATTERN.search(line)
        if scene:
            self._scene_index = int(scene.group('current'))
            self._scene_total = int(scene.group('total'))
            self._scene_name = scene.group('name').strip()
            self._record_prediction(raw=line)
            return

        patch = PATCH_PATTERN.search(line)
        if patch:
            self._record_prediction(
                patch_index=int(patch.group('current')),
                patch_total=int(patch.group('total')),
                elapsed_seconds=float(patch.group('seconds')) if patch.group('seconds') else None,
                raw=line,
            )
            return

        metrics = PREDICTION_METRICS_PATTERN.search(line)
        if metrics:
            f1 = float(metrics.group('f1'))
            baseline = float(metrics.group('baseline'))
            self.last_f1 = f1
            self.last_baseline = baseline
            self._record_prediction(
                f1=f1,
                precision=float(metrics.group('precision')),
                recall=float(metrics.group('recall')),
                baseline=baseline,
                raw=line,
            )

    def _record_prediction(
        self,
        *,
        patch_index: int | None = None,
        patch_total: int | None = None,
        elapsed_seconds: float | None = None,
        f1: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        baseline: float | None = None,
        raw: str,
    ) -> None:
        event = PredictionEvent(
            scene_index=self._scene_index,
            scene_total=self._scene_total,
            scene_name=self._scene_name,
            patch_index=patch_index,
            patch_total=patch_total,
            elapsed_seconds=elapsed_seconds,
            f1=f1,
            precision=precision,
            recall=recall,
            baseline=baseline,
            raw=raw,
        )
        _bounded_append(self.predictions, event)
        self.prediction_recorded.emit(event)

    def _append_memory_log(self, text: str) -> None:
        self.log.append(text)
        self._log_chars += len(text)
        while len(self.log) > 1 and self._log_chars > self._max_log_chars:
            self._log_chars -= len(self.log.pop(0))
        if self._log_chars > self._max_log_chars:
            self.log[0] = self.log[0][-self._max_log_chars :]
            self._log_chars = len(self.log[0])

    def _open_log(self) -> None:
        self._close_log()
        try:
            directory = self.cwd / '.studio_logs'
            directory.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r'[^A-Za-zА-Яа-яЁё0-9_.-]+', '_', self.name).strip('_.')
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            self.log_path = directory / f'{stamp}-{safe_name or "process"}.log'
            self._log_stream = self.log_path.open('w', encoding='utf-8', buffering=1)
            self._log_stream.write('$ ' + subprocess.list2cmdline(self.command) + '\n')
        except OSError:
            self.log_path = None
            self._log_stream = None

    def _write_full_log(self, text: str) -> None:
        if self._log_stream is None:
            return
        try:
            self._log_stream.write(text)
            self._log_stream.flush()
        except OSError:
            self._close_log()

    def _close_log(self) -> None:
        if self._log_stream is not None:
            try:
                self._log_stream.flush()
                self._log_stream.close()
            except OSError:
                pass
        self._log_stream = None

    def _flush_parser(self) -> None:
        if self._parse_buffer:
            self._parse_line(self._parse_buffer)
            self._parse_buffer = ''

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self.timer.stop()
        self._flush_parser()
        self._close_log()
        if self._stop_requested:
            self.status = 'Остановлен'
            self.status_key = 'stopped'
        elif exit_code == 0:
            self.status = 'Завершён'
            self.status_key = 'complete'
        else:
            self.status = f'Ошибка ({exit_code})'
            self.status_key = 'error'
        if not self._stop_requested:
            self._tree_pid = 0
        self.changed.emit()
        self.finished.emit(exit_code)

    def _error(self, _error: QProcess.ProcessError) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.timer.stop()
            self._close_log()
            if self._stop_requested:
                self.status = 'Остановлен'
                self.status_key = 'stopped'
            else:
                self.status = f'Ошибка запуска: {self.process.errorString()}'
                self.status_key = 'error'
            self.changed.emit()

    def _terminate_tree(self) -> None:
        pid = self._tree_pid or int(self.process.processId())
        if pid <= 0:
            self.process.terminate()
            return
        if os.name == 'nt':
            self._taskkill(pid, force=False)
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            self.process.terminate()

    def _kill_tree_if_running(self) -> None:
        pid = self._tree_pid or int(self.process.processId())
        if os.name == 'nt' and pid > 0:
            if not self._taskkill(pid, force=True):
                self.process.kill()
            self._tree_pid = 0
            return
        if pid > 0:
            try:
                os.killpg(pid, signal.SIGKILL)
                self._tree_pid = 0
                return
            except (OSError, ProcessLookupError):
                pass
        self.process.kill()
        self._tree_pid = 0

    @staticmethod
    def _taskkill(pid: int, *, force: bool) -> bool:
        command = ['taskkill', '/PID', str(pid), '/T']
        if force:
            command.append('/F')
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=5,
                check=False,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0


class ProcessRegistry(QObject):
    changed = Signal()
    process_added = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.processes: list[ManagedProcess] = []

    def create(
        self,
        name: str,
        program: str,
        arguments: list[str],
        cwd: Path,
    ) -> ManagedProcess:
        process = ManagedProcess(name, program, arguments, cwd, self)
        process.changed.connect(self.changed)
        process.finished.connect(lambda _code: self.changed.emit())
        self.processes.append(process)
        self.process_added.emit(process)
        self.changed.emit()
        return process

    def running(self) -> list[ManagedProcess]:
        return [process for process in self.processes if process.is_running()]

    def stop_all(self) -> None:
        for process in self.running():
            process.stop()


def _parse_duration(value: str | None) -> float | None:
    if not value:
        return None
    normalized = value.strip().casefold().replace(',', '.')
    if normalized.endswith('s'):
        try:
            return float(normalized[:-1])
        except ValueError:
            return None
    days = 0
    if 'day' in normalized or 'ден' in normalized:
        match = re.match(r'(?P<days>\d+)\s+[^,]+,?\s*(?P<time>.*)', normalized)
        if match:
            days = int(match.group('days'))
            normalized = match.group('time')
    parts = normalized.split(':')
    if len(parts) not in {2, 3}:
        try:
            return float(normalized)
        except ValueError:
            return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        hours = 0.0
        minutes, seconds = numbers
    else:
        hours, minutes, seconds = numbers
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _bounded_append(items: list, item) -> None:
    items.append(item)
    if len(items) > MAX_STRUCTURED_EVENTS:
        del items[: len(items) - MAX_STRUCTURED_EVENTS]
