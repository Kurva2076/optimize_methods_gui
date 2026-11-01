"""Загрузка результатов обучения PointNet++ из файлов ML-команды.

Модуль намеренно использует только стандартную библиотеку: экран результатов
не должен требовать PyTorch, CUDA или импорт кода обучения.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HISTORY_COLUMNS = (
    'epoch',
    'train_loss',
    'val_loss',
    'val_miou',
    'val_iou_ground',
    'val_iou_non_ground',
    'val_precision_ground',
    'val_precision_non_ground',
    'val_recall_ground',
    'val_recall_non_ground',
    'val_f1_ground',
    'val_f1_non_ground',
    'val_macro_precision',
    'val_macro_recall',
    'val_macro_f1',
    'lr',
)


class ResultsLoadError(ValueError):
    """Папка результатов не соответствует контракту ML-команды."""


@dataclass(frozen=True)
class HistoryRow:
    epoch: int
    train_loss: float
    val_loss: float
    val_miou: float
    val_iou_ground: float
    val_iou_non_ground: float
    val_precision_ground: float
    val_precision_non_ground: float
    val_recall_ground: float
    val_recall_non_ground: float
    val_f1_ground: float
    val_f1_non_ground: float
    val_macro_precision: float
    val_macro_recall: float
    val_macro_f1: float
    lr: float


@dataclass(frozen=True)
class ClassMetric:
    class_id: str
    class_name: str
    support: int
    iou: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class SceneMetric:
    scene_id: str
    num_points: int
    miou: float
    iou_ground: float
    iou_non_ground: float
    macro_f1: float


@dataclass(frozen=True)
class ExperimentResults:
    run_dir: Path
    results_dir: Path
    run_id: str
    split: str
    history: tuple[HistoryRow, ...]
    class_metrics: tuple[ClassMetric, ...]
    scene_metrics: tuple[SceneMetric, ...]
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    class_names: tuple[str, str]
    num_points: int
    miou: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    best_epoch: int
    best_val_miou: float


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
    except OSError as exc:
        raise ResultsLoadError(f'Не удалось прочитать файл {path.name}: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise ResultsLoadError(f'Файл {path.name} содержит некорректный JSON.') from exc
    if not isinstance(payload, dict):
        raise ResultsLoadError(f'В корне файла {path.name} должен быть JSON-объект.')
    return payload


def _finite_float(value: Any, field: str, source: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultsLoadError(f'Поле «{field}» в {source} должно быть числом.') from exc
    if not math.isfinite(result):
        raise ResultsLoadError(f'Поле «{field}» в {source} не должно быть NaN или Inf.')
    return result


def _nonnegative_int(value: Any, field: str, source: str) -> int:
    number = _finite_float(value, field, source)
    if number < 0 or number != int(number):
        raise ResultsLoadError(f'Поле «{field}» в {source} должно быть целым неотрицательным числом.')
    return int(number)


def _read_history(path: Path) -> tuple[HistoryRow, ...]:
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            missing = [column for column in HISTORY_COLUMNS if column not in fieldnames]
            if missing:
                raise ResultsLoadError(
                    f'В history.csv отсутствуют колонки: {", ".join(missing)}.'
                )
            rows = list(reader)
    except OSError as exc:
        raise ResultsLoadError(f'Не удалось прочитать history.csv: {exc}') from exc

    if not rows:
        raise ResultsLoadError('history.csv не содержит завершённых эпох.')

    history: list[HistoryRow] = []
    previous_epoch = 0
    for row_number, row in enumerate(rows, start=2):
        source = f'history.csv, строка {row_number}'
        epoch = _nonnegative_int(row.get('epoch'), 'epoch', source)
        if epoch < 1 or epoch <= previous_epoch:
            raise ResultsLoadError('Эпохи в history.csv должны идти по возрастанию, начиная с 1.')
        previous_epoch = epoch
        values = {
            name: _finite_float(row.get(name), name, source)
            for name in HISTORY_COLUMNS
            if name != 'epoch'
        }
        history.append(HistoryRow(epoch=epoch, **values))
    return tuple(history)


def _resolve_directories(selected: Path) -> tuple[Path, Path]:
    selected = selected.expanduser().resolve()
    if not selected.is_dir():
        raise ResultsLoadError('Выбранный путь не является папкой.')

    run_candidates = [selected]
    if selected.name.lower() in {'results', 'test-results', 'test_results'}:
        run_candidates.extend([selected.parent, selected.parent.parent])

    run_dir = next((path for path in run_candidates if (path / 'history.csv').is_file()), None)
    if run_dir is None:
        raise ResultsLoadError(
            'Не найден history.csv. Выберите папку конкретного запуска runs/<run_id> '
            'или вложенную в неё папку results.'
        )

    results_candidates = [
        selected,
        run_dir / 'results',
        run_dir / 'test-results',
        run_dir / 'test_results',
        run_dir,
    ]
    results_dir = next(
        (path for path in results_candidates if (path / 'metrics.json').is_file()),
        None,
    )
    if results_dir is None:
        raise ResultsLoadError(
            'Не найден metrics.json. Сначала выполните оценку модели и выберите папку '
            'запуска, содержащую results/metrics.json.'
        )
    return run_dir, results_dir


def _read_class_names(run_dir: Path) -> tuple[str, str]:
    config_path = run_dir / 'config.json'
    if not config_path.is_file():
        return ('Ground', 'Non-ground')
    config = _read_json(config_path)
    names = config.get('evaluation', {}).get('class_names')
    if (
        isinstance(names, list)
        and len(names) == 2
        and all(isinstance(name, str) and name.strip() for name in names)
    ):
        return names[0].strip(), names[1].strip()
    return ('Ground', 'Non-ground')


def _read_class_metrics(
    results_dir: Path,
    metrics: dict[str, Any],
    class_names: tuple[str, str],
) -> tuple[ClassMetric, ...]:
    csv_path = results_dir / 'metrics.csv'
    if csv_path.is_file():
        try:
            with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
        except OSError as exc:
            raise ResultsLoadError(f'Не удалось прочитать metrics.csv: {exc}') from exc
        parsed: list[ClassMetric] = []
        for row_number, row in enumerate(rows, start=2):
            if str(row.get('class_id', '')).lower() == 'macro':
                continue
            source = f'metrics.csv, строка {row_number}'
            parsed.append(
                ClassMetric(
                    class_id=str(row.get('class_id', len(parsed))),
                    class_name=str(row.get('class_name') or class_names[min(len(parsed), 1)]),
                    support=_nonnegative_int(row.get('support'), 'support', source),
                    iou=_finite_float(row.get('iou'), 'iou', source),
                    precision=_finite_float(row.get('precision'), 'precision', source),
                    recall=_finite_float(row.get('recall'), 'recall', source),
                    f1=_finite_float(row.get('f1'), 'f1', source),
                )
            )
        if len(parsed) == 2:
            return tuple(parsed)

    arrays = {
        field: metrics.get(field)
        for field in ('support', 'iou', 'precision', 'recall', 'f1')
    }
    if not all(isinstance(values, list) and len(values) == 2 for values in arrays.values()):
        raise ResultsLoadError('В metrics.json отсутствуют метрики для двух классов.')
    return tuple(
        ClassMetric(
            class_id=str(index),
            class_name=class_names[index],
            support=_nonnegative_int(arrays['support'][index], 'support', 'metrics.json'),
            iou=_finite_float(arrays['iou'][index], 'iou', 'metrics.json'),
            precision=_finite_float(arrays['precision'][index], 'precision', 'metrics.json'),
            recall=_finite_float(arrays['recall'][index], 'recall', 'metrics.json'),
            f1=_finite_float(arrays['f1'][index], 'f1', 'metrics.json'),
        )
        for index in range(2)
    )


def _read_scene_metrics(
    results_dir: Path,
    metrics: dict[str, Any],
) -> tuple[SceneMetric, ...]:
    rows: list[dict[str, Any]] = []
    csv_path = results_dir / 'per_scene_metrics.csv'
    if csv_path.is_file():
        try:
            with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
        except OSError as exc:
            raise ResultsLoadError(f'Не удалось прочитать per_scene_metrics.csv: {exc}') from exc
    elif isinstance(metrics.get('per_scene'), list):
        rows = [row for row in metrics['per_scene'] if isinstance(row, dict)]

    parsed: list[SceneMetric] = []
    for row_number, row in enumerate(rows, start=2):
        source = f'per_scene_metrics.csv, строка {row_number}'
        scene_id = str(row.get('scene_id', '')).strip()
        if not scene_id:
            raise ResultsLoadError(f'В {source} не указан scene_id.')
        parsed.append(
            SceneMetric(
                scene_id=scene_id,
                num_points=_nonnegative_int(row.get('num_points'), 'num_points', source),
                miou=_finite_float(row.get('miou'), 'miou', source),
                iou_ground=_finite_float(row.get('iou_ground'), 'iou_ground', source),
                iou_non_ground=_finite_float(
                    row.get('iou_non_ground'), 'iou_non_ground', source
                ),
                macro_f1=_finite_float(row.get('macro_f1'), 'macro_f1', source),
            )
        )
    return tuple(parsed)


def _read_confusion_matrix(metrics: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    matrix = metrics.get('confusion_matrix')
    if not (
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
    ):
        raise ResultsLoadError('Поле confusion_matrix в metrics.json должно иметь размер 2×2.')
    values = tuple(
        tuple(
            _nonnegative_int(value, 'confusion_matrix', 'metrics.json')
            for value in row
        )
        for row in matrix
    )
    if sum(sum(row) for row in values) == 0:
        raise ResultsLoadError('Матрица ошибок в metrics.json не должна быть пустой.')
    return values  # type: ignore[return-value]


def load_experiment_results(directory: str | Path) -> ExperimentResults:
    """Загрузить один завершённый запуск по контракту ``lidar_pointnet2``.

    Можно передать как корень ``runs/<run_id>``, так и его папку ``results``.
    """
    run_dir, results_dir = _resolve_directories(Path(directory))
    history = _read_history(run_dir / 'history.csv')
    metrics = _read_json(results_dir / 'metrics.json')
    class_names = _read_class_names(run_dir)
    class_metrics = _read_class_metrics(results_dir, metrics, class_names)
    scene_metrics = _read_scene_metrics(results_dir, metrics)
    matrix = _read_confusion_matrix(metrics)

    num_points = _nonnegative_int(metrics.get('num_points'), 'num_points', 'metrics.json')
    matrix_points = sum(sum(row) for row in matrix)
    if num_points != matrix_points:
        raise ResultsLoadError(
            'Число точек в metrics.json не совпадает с суммой элементов матрицы ошибок.'
        )

    best_val_miou = _finite_float(
        metrics.get('best_val_miou', max(row.val_miou for row in history)),
        'best_val_miou',
        'metrics.json',
    )
    best_epoch_value = metrics.get('best_epoch')
    if best_epoch_value is None:
        best_epoch = max(history, key=lambda row: row.val_miou).epoch
    else:
        best_epoch = _nonnegative_int(best_epoch_value, 'best_epoch', 'metrics.json')
    if best_epoch not in {row.epoch for row in history}:
        raise ResultsLoadError('Лучшая эпоха отсутствует в history.csv.')

    return ExperimentResults(
        run_dir=run_dir,
        results_dir=results_dir,
        run_id=str(metrics.get('run_id') or run_dir.name),
        split=str(metrics.get('split') or 'test'),
        history=history,
        class_metrics=class_metrics,
        scene_metrics=scene_metrics,
        confusion_matrix=matrix,
        class_names=class_names,
        num_points=num_points,
        miou=_finite_float(metrics.get('miou'), 'miou', 'metrics.json'),
        macro_precision=_finite_float(
            metrics.get('macro_precision'), 'macro_precision', 'metrics.json'
        ),
        macro_recall=_finite_float(
            metrics.get('macro_recall'), 'macro_recall', 'metrics.json'
        ),
        macro_f1=_finite_float(metrics.get('macro_f1'), 'macro_f1', 'metrics.json'),
        best_epoch=best_epoch,
        best_val_miou=best_val_miou,
    )
