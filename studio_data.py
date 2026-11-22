"""Файловая модель ML Studio.

Модуль намеренно не хранит собственную базу данных: источником правды остаются
каталоги проекта, как и в приложении команды тестирования.
"""

from __future__ import annotations

import csv
import json
import shutil
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_MARKER = 'hpo_pointnet.py'


@dataclass(frozen=True, slots=True)
class SceneInfo:
    path: Path
    points: int
    columns: int
    bytes_on_disk: int


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    path: Path
    scenes: tuple[SceneInfo, ...]

    @property
    def points(self) -> int:
        return sum(scene.points for scene in self.scenes)

    @property
    def bytes_on_disk(self) -> int:
        return sum(scene.bytes_on_disk for scene in self.scenes)


@dataclass(frozen=True, slots=True)
class RunInfo:
    path: Path
    name: str
    target: str = ''
    features: str = ''
    best_f1: float | None = None
    baseline_f1: float | None = None
    status: str = 'Не запускался'
    trials: tuple[dict[str, str], ...] = field(default_factory=tuple)
    best: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactInfo:
    path: Path
    category: str
    run_name: str = ''


def is_project(path: str | Path) -> bool:
    root = Path(path).expanduser()
    return root.is_dir() and (root / PROJECT_MARKER).is_file()


def human_bytes(size: int) -> str:
    value = float(size)
    for suffix in ('Б', 'КБ', 'МБ', 'ГБ', 'ТБ'):
        if value < 1024.0 or suffix == 'ТБ':
            precision = 0 if suffix == 'Б' else 1
            return f'{value:.{precision}f} {suffix}'.replace('.', ',')
        value /= 1024.0
    return f'{size} Б'


def human_count(value: int) -> str:
    if value >= 1_000_000:
        return f'{value / 1_000_000:.1f} млн'.replace('.', ',')
    if value >= 1_000:
        return f'{value / 1_000:.1f} тыс.'.replace('.', ',')
    return str(value)


def npy_shape(path: str | Path) -> tuple[int, ...]:
    """Прочитать форму массива без загрузки точек в оперативную память."""
    array = np.load(Path(path), mmap_mode='r', allow_pickle=False)
    try:
        return tuple(int(value) for value in array.shape)
    finally:
        mmap = getattr(array, '_mmap', None)
        if mmap is not None:
            mmap.close()


def scan_datasets(root: str | Path) -> list[DatasetInfo]:
    project = Path(root).expanduser().resolve()
    result: list[DatasetInfo] = []
    candidates = sorted(
        path for path in project.glob('exports*') if path.is_dir()
    )
    for directory in candidates:
        scenes: list[SceneInfo] = []
        for path in sorted(directory.glob('scene_*.npy')):
            try:
                shape = npy_shape(path)
                if len(shape) != 2 or shape[0] <= 0 or shape[1] < 3:
                    continue
                scenes.append(
                    SceneInfo(
                        path=path,
                        points=shape[0],
                        columns=shape[1],
                        bytes_on_disk=path.stat().st_size,
                    )
                )
            except (OSError, ValueError):
                continue
        if scenes:
            result.append(DatasetInfo(directory, tuple(scenes)))
    return result


def evenly_spaced_scenes(
    scenes: Iterable[SceneInfo],
    count: int,
) -> tuple[SceneInfo, ...]:
    """Выбрать сцены равномерно по всему датасету."""
    values = tuple(scenes)
    if not values:
        return ()
    if count <= 0:
        raise ValueError('Число сцен должно быть больше нуля.')
    if count >= len(values):
        return values
    if count == 1:
        return (values[0],)
    indexes = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return tuple(values[index] for index in indexes)


def create_mini_dataset(
    dataset: DatasetInfo,
    count: int,
    *,
    progress: Callable[[int, int, Path], bool | None] | None = None,
) -> Path:
    """Атомарно создать ``<dataset>_mini`` без перезаписи.

    ``progress(done, total, path)`` может вернуть ``False`` для отмены.
    Сначала файлы пишутся во временный каталог, поэтому ошибка не
    оставит неполную мини-копию с конечным именем.
    """
    selected = evenly_spaced_scenes(dataset.scenes, count)
    target = dataset.path.with_name(f'{dataset.path.name}_mini')
    if target.exists():
        raise FileExistsError(f'Каталог уже существует: {target}')

    temporary = target.with_name(f'.{target.name}.tmp-{uuid.uuid4().hex}')
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        total = len(selected)
        for done, scene in enumerate(selected, 1):
            shutil.copy2(scene.path, temporary / scene.path.name)
            if progress is not None and progress(done, total, scene.path) is False:
                raise InterruptedError('Создание мини-копии отменено.')
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _first_number(value: Any, keys: Iterable[str]) -> float | None:
    wanted = {key.casefold() for key in keys}
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.casefold() in wanted and isinstance(nested, (int, float)):
                return float(nested)
        for nested in value.values():
            found = _first_number(nested, wanted)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _first_number(nested, wanted)
            if found is not None:
                return found
    return None


def _first_text(value: Any, keys: Iterable[str]) -> str:
    wanted = {key.casefold() for key in keys}
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.casefold() in wanted and isinstance(nested, str):
                return nested
        for nested in value.values():
            found = _first_text(nested, wanted)
            if found:
                return found
    return ''


def read_trials(path: Path) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        return ()
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return tuple(dict(row) for row in csv.DictReader(stream))
    except (OSError, csv.Error):
        return ()


def scan_runs(root: str | Path, runs_directory: str | Path | None = None) -> list[RunInfo]:
    project = Path(root).expanduser().resolve()
    runs_root = (
        Path(runs_directory).expanduser().resolve()
        if runs_directory is not None
        else project / 'runs'
    )
    if not runs_root.is_dir():
        return []

    result: list[RunInfo] = []
    for path in sorted(entry for entry in runs_root.iterdir() if entry.is_dir()):
        best = _read_json(path / 'best.json')
        trials = read_trials(path / 'trials.csv')
        if not best and not trials and not (path / 'study.db').exists():
            continue

        target = _first_text(best, ('target',))
        features = _first_text(best, ('features', 'feature_set'))
        best_f1 = _first_number(best, ('f1', 'best_f1', 'value'))
        baseline = _first_number(best, ('baseline_f1', 'baseline'))

        states = [row.get('state', '').upper() for row in trials]
        if any(state in {'RUNNING', 'WAITING'} for state in states):
            status = 'Выполняется'
        elif trials and any(state == 'COMPLETE' for state in states):
            status = 'Завершён'
        elif trials and all(state in {'FAIL', 'PRUNED'} for state in states):
            status = 'Ошибка'
        elif best:
            status = 'Завершён'
        else:
            status = 'Остановлен'

        result.append(
            RunInfo(
                path=path,
                name=path.name,
                target=target,
                features=features,
                best_f1=best_f1,
                baseline_f1=baseline,
                status=status,
                trials=trials,
                best=best,
            )
        )
    return result


def scan_artifacts(root: str | Path) -> list[ArtifactInfo]:
    project = Path(root).expanduser().resolve()
    artifacts: list[ArtifactInfo] = []
    runs_root = project / 'runs'
    if runs_root.is_dir():
        for path in sorted(runs_root.rglob('*.md')):
            artifacts.append(ArtifactInfo(path, 'Документы', _run_name(path, runs_root)))
        for path in sorted(runs_root.rglob('*.pt')):
            artifacts.append(ArtifactInfo(path, 'Модели', _run_name(path, runs_root)))
        for path in sorted(runs_root.rglob('*.log')):
            artifacts.append(ArtifactInfo(path, 'Логи', _run_name(path, runs_root)))
    models_root = project / 'models'
    if models_root.is_dir():
        for path in sorted(models_root.glob('*.pt')):
            artifacts.append(ArtifactInfo(path, 'Модели', 'Импортированные'))
    return artifacts


def _run_name(path: Path, runs_root: Path) -> str:
    try:
        relative = path.relative_to(runs_root)
    except ValueError:
        return ''
    return relative.parts[0] if len(relative.parts) > 1 else 'Сводный отчёт'


def read_text_preview(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='replace')[:limit]
    except OSError as error:
        return f'Не удалось прочитать файл: {error}'

