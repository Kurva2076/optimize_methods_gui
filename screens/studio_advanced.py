"""Расширенные экраны аналитики ML Studio.

Модуль не хранит метрики в собственной базе: все числа берутся из
``best.json``, ``trials.csv``, ``metrics.csv``, stdout процессов и
``scenes__*.json``. Это позволяет подменить базовые страницы без изменения
форматов вычислительного ядра.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QFile, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QWidget,
)

from point_cloud_canvas import PointCloudCanvas
from point_cloud_data import PointCloudData, load_point_cloud
from screens.studio import StudioPage, panel, title_label
from studio_data import (
    ArtifactInfo,
    DatasetInfo,
    RunInfo,
    human_bytes,
    read_text_preview,
    scan_artifacts,
    scan_datasets,
    scan_runs,
)
from studio_dialogs import LaunchDialog, TestModelDialog, write_json_exclusive
from studio_processes import ManagedProcess
from studio_widgets import open_in_file_manager

TARGET_CLASS_IDS = {
    'ground': (2,),
    'vegetation': (3, 4, 5),
    'noise': (7, 18),
    'building': (6,),
    'water': (9,),
}

LIVE_METRIC_PATTERN = re.compile(
    r'(?:trial\s*[=:]\s*(?P<trial>\d+).*?)?'
    r'(?:epoch\s*[=:]\s*(?P<epoch>\d+).*?)?'
    r'F1\s*[=:]\s*(?P<f1>\d+(?:\.\d+)?)',
    re.IGNORECASE,
)
BASELINE_PATTERN = re.compile(
    r'baseline\s*[=:]\s*(?P<baseline>\d+(?:\.\d+)?)',
    re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: Any, *keys: str) -> float | None:
    wanted = {key.casefold() for key in keys}
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.casefold() in wanted and isinstance(nested, (int, float)):
                return float(nested)
        for nested in value.values():
            found = _number(nested, *keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _number(nested, *keys)
            if found is not None:
                return found
    return None


def _best_params(run: RunInfo) -> dict[str, Any]:
    params = run.best.get('params') if isinstance(run.best, dict) else None
    if isinstance(params, dict):
        return params
    best_trial = run.best.get('best_trial') if isinstance(run.best, dict) else None
    if isinstance(best_trial, dict) and isinstance(best_trial.get('params'), dict):
        return best_trial['params']
    return {}


def _trial_score(row: dict[str, str]) -> float:
    try:
        return float(row.get('f1', row.get('value', '-inf')))
    except (TypeError, ValueError):
        return -math.inf


def _best_trial(run: RunInfo) -> dict[str, str] | None:
    complete = [
        row
        for row in run.trials
        if row.get('state', 'COMPLETE').upper() == 'COMPLETE'
        and math.isfinite(_trial_score(row))
    ]
    return max(complete, key=_trial_score, default=None)


def _metrics_path(run: RunInfo) -> Path | None:
    row = _best_trial(run)
    candidates: list[Path] = []
    if row:
        checkpoint = row.get('checkpoint', '')
        if checkpoint:
            path = Path(checkpoint)
            candidates.append((run.path / path).resolve() if not path.is_absolute() else path)
        number = row.get('number', '')
        if number.isdigit():
            candidates.extend(
                (
                    run.path / f'trial_{int(number):04d}' / 'metrics.csv',
                    run.path / f'trial_{int(number)}' / 'metrics.csv',
                )
            )
    checkpoint = run.best.get('checkpoint') if isinstance(run.best, dict) else None
    if isinstance(checkpoint, str) and checkpoint:
        path = Path(checkpoint)
        checkpoint_path = (run.path / path).resolve() if not path.is_absolute() else path
        candidates.append(checkpoint_path.parent / 'metrics.csv')
    candidates.extend(sorted(run.path.glob('trial_*/metrics.csv')))
    for candidate in candidates:
        if candidate.name != 'metrics.csv':
            candidate = candidate.parent / 'metrics.csv'
        if candidate.is_file():
            return candidate
    return None


def _metric_series(path: Path | None) -> list[float]:
    if path is None:
        return []
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return []
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get('f1', '')))
        except (TypeError, ValueError):
            continue
    return values


def parse_live_metrics(text: str) -> list[dict[str, float | int | None]]:
    """Разобрать только реальные строки метрик из stdout."""
    rows: list[dict[str, float | int | None]] = []
    for line in text.splitlines():
        match = LIVE_METRIC_PATTERN.search(line)
        if not match:
            continue
        baseline_match = BASELINE_PATTERN.search(line)
        rows.append(
            {
                'trial': int(match.group('trial')) if match.group('trial') else None,
                'epoch': int(match.group('epoch')) if match.group('epoch') else None,
                'f1': float(match.group('f1')),
                'baseline': (
                    float(baseline_match.group('baseline')) if baseline_match else None
                ),
            }
        )
    return rows


class SeriesChart(QWidget):
    """Лёгкий Qt-график без внешнего движка и выдуманных точек."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.values: list[float] = []
        self.baseline: float | None = None
        self.setMinimumHeight(150)

    def set_values(self, values: list[float], baseline: float | None = None) -> None:
        self.values = [value for value in values if math.isfinite(value)]
        self.baseline = baseline if baseline is None or math.isfinite(baseline) else None
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = QApplication.instance().property('studioTheme') == 'dark'
        background = QColor('#0F192A' if dark else '#F8FAFD')
        grid = QColor('#2C3E59' if dark else '#DCE4EF')
        muted = QColor('#9BAAC0' if dark else '#697386')
        accent = QColor('#73A2FF' if dark else '#356DF3')
        painter.fillRect(self.rect(), background)
        area = QRectF(38, 14, max(10, self.width() - 54), max(10, self.height() - 42))
        painter.setPen(QPen(grid, 1))
        for step in range(5):
            y = area.top() + area.height() * step / 4
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
        if not self.values:
            painter.setPen(muted)
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, 'Нет metrics.csv / Live-метрик')
            return
        minimum = min([0.0, *self.values, *([self.baseline] if self.baseline is not None else [])])
        maximum = max([1.0, *self.values, *([self.baseline] if self.baseline is not None else [])])
        span = max(1e-9, maximum - minimum)

        def point(index: int, value: float) -> QPointF:
            x = area.left() + area.width() * index / max(1, len(self.values) - 1)
            y = area.bottom() - area.height() * (value - minimum) / span
            return QPointF(x, y)

        if self.baseline is not None:
            y = area.bottom() - area.height() * (self.baseline - minimum) / span
            painter.setPen(QPen(QColor('#D08B00'), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
        path = QPainterPath(point(0, self.values[0]))
        for index, value in enumerate(self.values[1:], 1):
            path.lineTo(point(index, value))
        painter.setPen(QPen(accent, 2.2))
        painter.drawPath(path)
        painter.setBrush(accent)
        for index, value in enumerate(self.values):
            painter.drawEllipse(point(index, value), 3, 3)


class AdvancedRunsPage(StudioPage):
    """Прогоны, сравнение, финальный запус и безопасное удаление."""

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.runs: list[RunInfo] = []
        self._delete_path: Path | None = None
        refresh = QPushButton('Перечитать runs/')
        refresh.setObjectName('secondaryButton')
        refresh.clicked.connect(self.refresh)
        launch = QPushButton('Новый запуск')
        launch.setObjectName('primaryButton')
        launch.clicked.connect(lambda: LaunchDialog(self.context, parent=self).exec())
        self.add_header(
            'Прогоны',
            'Метрики из best.json, trials.csv и metrics.csv без демо-значений.',
            [refresh, launch],
        )
        content = QHBoxLayout()
        content.setSpacing(12)
        left, left_layout = panel()
        left.setMinimumWidth(280)
        left.setMaximumWidth(350)
        left_layout.addWidget(title_label('Прогоны'))
        hint = QLabel('Ctrl/⌘ + щелчок — сравнить')
        hint.setObjectName('mutedText')
        left_layout.addWidget(hint)
        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.run_list.itemSelectionChanged.connect(self.show_selection)
        left_layout.addWidget(self.run_list, 1)
        content.addWidget(left)

        center, center_layout = panel()
        self.heading = title_label('Прогон не выбран')
        self.summary = QLabel('')
        self.summary.setObjectName('runSummary')
        self.summary.setWordWrap(True)
        center_layout.addWidget(self.heading)
        center_layout.addWidget(self.summary)
        self.chart = SeriesChart()
        center_layout.addWidget(self.chart)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        center_layout.addWidget(self.table, 1)
        content.addWidget(center, 1)

        right, right_layout = panel()
        right.setMinimumWidth(290)
        right.setMaximumWidth(370)
        right_layout.addWidget(title_label('Действия'))
        self.params = QPlainTextEdit()
        self.params.setObjectName('jsonPreview')
        self.params.setReadOnly(True)
        right_layout.addWidget(self.params, 1)
        self.note = QPlainTextEdit()
        self.note.setObjectName('commandPreview')
        self.note.setPlaceholderText('Заметка к прогону…')
        self.note.setMaximumHeight(100)
        right_layout.addWidget(self.note)
        save_note = QPushButton('Сохранить заметку')
        save_note.setObjectName('secondaryButton')
        save_note.clicked.connect(self.save_note)
        self.continue_button = QPushButton('Продолжить в форме')
        self.continue_button.setObjectName('secondaryButton')
        self.continue_button.clicked.connect(self.continue_run)
        self.export_button = QPushButton('Параметры в JSON')
        self.export_button.setObjectName('secondaryButton')
        self.export_button.clicked.connect(self.export_params)
        self.final_button = QPushButton('Финальный прогон')
        self.final_button.setObjectName('primaryButton')
        self.final_button.clicked.connect(self.final_run)
        self.folder_button = QPushButton('Открыть папку')
        self.folder_button.setObjectName('secondaryButton')
        self.folder_button.clicked.connect(self.open_folder)
        self.delete_button = QPushButton('Удалить в корзину')
        self.delete_button.setObjectName('dangerButton')
        self.delete_button.clicked.connect(self.delete_run)
        for button in (
            save_note,
            self.continue_button,
            self.export_button,
            self.final_button,
            self.folder_button,
            self.delete_button,
        ):
            right_layout.addWidget(button)
        content.addWidget(right)
        self.root.addLayout(content, 1)
        self.refresh()

    def selected_runs(self) -> list[RunInfo]:
        return [
            self.runs[index.row()]
            for index in self.run_list.selectedIndexes()
            if 0 <= index.row() < len(self.runs)
        ]

    def current_run(self) -> RunInfo | None:
        selected = self.selected_runs()
        return selected[0] if len(selected) == 1 else None

    def refresh(self) -> None:
        selected_names = {run.name for run in self.selected_runs()}
        self.runs = scan_runs(self.context.project_root) if self.context.project_root else []
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for run in self.runs:
            f1 = f'F1 {run.best_f1:.4f}' if run.best_f1 is not None else 'F1 —'
            complete = sum(row.get('state', '').upper() == 'COMPLETE' for row in run.trials)
            item = QListWidgetItem(f'{run.name}\n{run.status} · {f1} · trials {complete}/{len(run.trials)}')
            item.setToolTip(str(run.path))
            self.run_list.addItem(item)
            if run.name in selected_names:
                item.setSelected(True)
        self.run_list.blockSignals(False)
        if self.runs and not self.run_list.selectedItems():
            self.run_list.setCurrentRow(0)
        self.show_selection()

    def show_selection(self) -> None:
        runs = self.selected_runs()
        enabled = len(runs) == 1
        for button in (
            self.continue_button,
            self.export_button,
            self.folder_button,
            self.delete_button,
        ):
            button.setEnabled(enabled)
        self.final_button.setEnabled(enabled and bool(_best_params(runs[0])) if runs else False)
        if not runs:
            self.heading.setText('Прогоны не найдены')
            self.summary.setText('Каталог runs/ пуст или ещё не создан.')
            self.table.setRowCount(0)
            self.chart.set_values([])
            self.params.clear()
            self.note.clear()
            return
        if len(runs) > 1:
            self.show_comparison(runs)
            return
        self.show_run(runs[0])

    def show_run(self, run: RunInfo) -> None:
        self.heading.setText(run.name)
        f1 = f'{run.best_f1:.4f}' if run.best_f1 is not None else '—'
        baseline = f'{run.baseline_f1:.4f}' if run.baseline_f1 is not None else '—'
        gap_value = (
            run.best_f1 - run.baseline_f1
            if run.best_f1 is not None and run.baseline_f1 is not None
            else None
        )
        gap = f'{gap_value:+.4f}' if gap_value is not None else '—'
        precision = _number(run.best, 'precision')
        recall = _number(run.best, 'recall')
        self.summary.setText(
            f'Класс: {run.target or "—"} · Признаки: {run.features or "—"} · '
            f'F1: {f1} · Precision: {precision:.4f} · Recall: {recall:.4f} · '
            f'Baseline: {baseline} · Отрыв: {gap}'
            if precision is not None and recall is not None
            else (
                f'Класс: {run.target or "—"} · Признаки: {run.features or "—"} · '
                f'F1: {f1} · Baseline: {baseline} · Отрыв: {gap}'
            )
        )
        columns = [
            key
            for key in ('number', 'state', 'f1', 'precision', 'recall', 'seconds')
            if any(key in row for row in run.trials)
        ]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(run.trials))
        for row_index, values in enumerate(sorted(run.trials, key=_trial_score, reverse=True)):
            for column, key in enumerate(columns):
                self.table.setItem(row_index, column, QTableWidgetItem(values.get(key, '')))
        self.chart.set_values(_metric_series(_metrics_path(run)), run.baseline_f1)
        params = _best_params(run)
        self.params.setPlainText(json.dumps(params, ensure_ascii=False, indent=2) if params else 'Нет params в best.json')
        studio = _read_json(run.path / 'studio.json')
        self.note.setPlainText(str(studio.get('note', '')))

    def show_comparison(self, runs: list[RunInfo]) -> None:
        self.heading.setText(f'Сравнение · {len(runs)}')
        self.summary.setText('Все значения прочитаны из артефактов выбранных прогонов.')
        rows = ('target', 'features', 'best_f1', 'precision', 'recall', 'baseline', 'gap')
        self.table.setColumnCount(len(runs) + 1)
        self.table.setHorizontalHeaderLabels(['Поле', *(run.name for run in runs)])
        self.table.setRowCount(len(rows))
        for row_index, key in enumerate(rows):
            self.table.setItem(row_index, 0, QTableWidgetItem(key))
            row_values: list[float] = []
            for column, run in enumerate(runs, 1):
                values = {
                    'target': run.target,
                    'features': run.features,
                    'best_f1': run.best_f1,
                    'precision': _number(run.best, 'precision'),
                    'recall': _number(run.best, 'recall'),
                    'baseline': run.baseline_f1,
                    'gap': (
                        run.best_f1 - run.baseline_f1
                        if run.best_f1 is not None and run.baseline_f1 is not None
                        else None
                    ),
                }
                value = values[key]
                if isinstance(value, float):
                    row_values.append(value)
                text = f'{value:.4f}' if isinstance(value, float) else str(value or '—')
                self.table.setItem(row_index, column, QTableWidgetItem(text))
            if key in {'best_f1', 'precision', 'recall', 'gap'} and row_values:
                best = max(row_values)
                for column in range(1, len(runs) + 1):
                    item = self.table.item(row_index, column)
                    try:
                        is_best = math.isclose(float(item.text()), best)
                    except (AttributeError, ValueError):
                        is_best = False
                    if is_best:
                        item.setBackground(QBrush(QColor('#E8F7EF')))
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
            elif key in {'target', 'features'}:
                texts = {
                    self.table.item(row_index, column).text()
                    for column in range(1, len(runs) + 1)
                }
                if len(texts) > 1:
                    for column in range(1, len(runs) + 1):
                        item = self.table.item(row_index, column)
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
        self.chart.set_values([run.best_f1 for run in runs if run.best_f1 is not None])
        self.params.setPlainText('\n\n'.join(
            f'{run.name}:\n{json.dumps(_best_params(run), ensure_ascii=False, indent=2)}'
            for run in runs
        ))
        self.note.clear()

    def save_note(self) -> None:
        run = self.current_run()
        if run is None:
            return
        path = run.path / 'studio.json'
        value = _read_json(path)
        value['note'] = self.note.toPlainText()
        try:
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(path)
        except OSError as error:
            QMessageBox.warning(self, 'Не удалось сохранить', str(error))

    def continue_run(self) -> None:
        run = self.current_run()
        if run is None:
            return
        dialog = LaunchDialog(self.context, parent=self)
        for name, value in (('run_name', run.name), ('target', run.target), ('features', run.features)):
            widget = getattr(dialog, name, None)
            if widget is None or not value:
                continue
            if hasattr(widget, 'setText'):
                widget.setText(value)
            elif hasattr(widget, 'setCurrentText'):
                widget.setCurrentText(value)
        launch = run.best.get('launch') if isinstance(run.best, dict) else None
        if isinstance(launch, dict):
            for key in ('seed', 'epochs', 'patch_points', 'patches_per_scene', 'workers'):
                widget = getattr(dialog, key, None)
                value = launch.get(key)
                if widget is not None and isinstance(value, int) and hasattr(widget, 'setValue'):
                    widget.setValue(value)
        if hasattr(dialog, '_update_command'):
            dialog._update_command()
        dialog.exec()

    def export_params(self) -> Path | None:
        run = self.current_run()
        root = self.context.project_root
        if run is None or root is None:
            return None
        params = _best_params(run)
        if not params:
            QMessageBox.warning(self, 'Нет параметров', 'В best.json нет params.')
            return None
        path = root / 'runs' / f'{run.name}.params.json'
        try:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(params, stream, ensure_ascii=False, indent=2)
        except FileExistsError:
            QMessageBox.information(self, 'Файл уже существует', str(path))
            return None
        except OSError as error:
            QMessageBox.warning(self, 'Не удалось записать', str(error))
            return None
        return path

    def final_run(self) -> None:
        run = self.current_run()
        root = self.context.project_root
        python = self.context.python_executable()
        if run is None or root is None or python is None:
            return
        final_data = root / 'exports_final'
        output = root / 'runs' / f'final_{run.name}'
        params_path = root / 'runs' / '.gui_configs' / f'final-{run.name}.params.json'
        if not final_data.is_dir():
            QMessageBox.warning(self, 'Нет финальных данных', f'Не найдено: {final_data}')
            return
        if output.exists():
            QMessageBox.warning(self, 'Финальный прогон уже есть', str(output))
            return
        params = _best_params(run)
        if not params:
            QMessageBox.warning(self, 'Нет параметров', 'В best.json нет params.')
            return
        epochs, accepted = QInputDialog.getInt(
            self, 'Финальный прогон', 'Число эпох', 30, 1, 10000
        )
        if not accepted:
            return
        try:
            write_json_exclusive(params_path, params)
        except (FileExistsError, OSError) as error:
            QMessageBox.warning(self, 'Не удалось записать params', str(error))
            return
        command = [
            str(root / 'hpo_pointnet.py'), '--data-dir', str(final_data), '--output-dir', str(output),
            '--target', run.target, '--features', run.features or 'xyz', '--fixed-params', str(params_path),
            '--n-trials', '1', '--epochs', str(epochs),
        ]
        process = self.context.processes.create(f'final:{run.name}', str(python), command, root)
        process.start()

    def open_folder(self) -> None:
        run = self.current_run()
        if run is not None:
            open_in_file_manager(run.path)

    def delete_run(self) -> None:
        run = self.current_run()
        if run is None:
            return
        if self._delete_path != run.path:
            self._delete_path = run.path
            self.delete_button.setText('Ещё раз — в корзину')
            QTimer.singleShot(5000, self._reset_delete)
            return
        path = run.path
        self._reset_delete()
        if not QFile.moveToTrash(str(path)):
            QMessageBox.warning(self, 'Не удалось удалить', f'Не удалось переместить в корзину: {path}')
        self.refresh()

    def _reset_delete(self) -> None:
        self._delete_path = None
        self.delete_button.setText('Удалить в корзину')


class AdvancedLivePage(StudioPage):
    """Мониторинг нескольких QProcess с метриками и полным логом."""

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self._connected: set[int] = set()
        self._stop_process: ManagedProcess | None = None
        self.add_header('Live', 'Реальные stdout/stderr, F1 и baseline каждого процесса.')
        body = QHBoxLayout()
        body.setSpacing(12)
        left, left_layout = panel()
        left.setMinimumWidth(270)
        left.setMaximumWidth(350)
        left_layout.addWidget(title_label('Процессы'))
        self.process_list = QListWidget()
        self.process_list.currentRowChanged.connect(self.select_process)
        left_layout.addWidget(self.process_list, 1)
        actions = QHBoxLayout()
        self.restart_button = QPushButton('Перезапустить')
        self.restart_button.setObjectName('secondaryButton')
        self.restart_button.clicked.connect(self.restart_process)
        self.stop_button = QPushButton('Остановить')
        self.stop_button.setObjectName('dangerButton')
        self.stop_button.clicked.connect(self.stop_process)
        actions.addWidget(self.restart_button)
        actions.addWidget(self.stop_button)
        left_layout.addLayout(actions)
        body.addWidget(left)

        center, center_layout = panel()
        header = QHBoxLayout()
        self.heading = title_label('Активных процессов нет')
        self.autoscroll = QCheckBox('Автопрокрутка')
        self.autoscroll.setChecked(True)
        header.addWidget(self.heading)
        header.addStretch(1)
        header.addWidget(self.autoscroll)
        center_layout.addLayout(header)
        self.summary = QLabel('Запустите обучение или тестирование.')
        self.summary.setObjectName('runSummary')
        self.summary.setWordWrap(True)
        center_layout.addWidget(self.summary)
        self.chart = SeriesChart()
        center_layout.addWidget(self.chart)
        self.trials = QTableWidget(0, 5)
        self.trials.setHorizontalHeaderLabels(['Trial', 'Эпоха', 'F1', 'Baseline', 'Отрыв'])
        self.trials.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.trials.verticalHeader().setVisible(False)
        self.trials.setMaximumHeight(190)
        center_layout.addWidget(self.trials)
        body.addWidget(center, 1)

        log_card, log_layout = panel()
        log_card.setMinimumWidth(420)
        log_header = QHBoxLayout()
        log_header.addWidget(title_label('Лента событий'))
        copy = QPushButton('Копировать')
        copy.setObjectName('secondaryButton')
        copy.clicked.connect(self.copy_log)
        log_header.addStretch(1)
        log_header.addWidget(copy)
        log_layout.addLayout(log_header)
        self.log = QPlainTextEdit()
        self.log.setObjectName('liveLog')
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.log, 1)
        continue_button = QPushButton('Продолжить в форме')
        continue_button.setObjectName('secondaryButton')
        continue_button.clicked.connect(self.continue_in_form)
        log_layout.addWidget(continue_button)
        body.addWidget(log_card, 1)
        self.root.addLayout(body, 1)
        self.context.processes.changed.connect(self.refresh)
        self.context.processes.process_added.connect(self._connect_process)
        for process in self.context.processes.processes:
            self._connect_process(process)
        self.refresh()

    def _connect_process(self, process: ManagedProcess) -> None:
        identity = id(process)
        if identity in self._connected:
            return
        self._connected.add(identity)
        process.output_received.connect(lambda _text, target=process: self.process_updated(target))

    def selected_process(self) -> ManagedProcess | None:
        row = self.process_list.currentRow()
        values = self.context.processes.processes
        return values[row] if 0 <= row < len(values) else None

    def refresh(self) -> None:
        current = self.process_list.currentRow()
        self.process_list.blockSignals(True)
        self.process_list.clear()
        for process in self.context.processes.processes:
            snapshot = process.snapshot()
            f1 = f' · F1 {snapshot.last_f1:.4f}' if snapshot.last_f1 is not None else ''
            minutes, seconds = divmod(snapshot.seconds, 60)
            self.process_list.addItem(f'{snapshot.name}\n{snapshot.status} · {minutes:02d}:{seconds:02d}{f1}')
            self._connect_process(process)
        self.process_list.blockSignals(False)
        if self.context.processes.processes:
            self.process_list.setCurrentRow(min(max(current, 0), len(self.context.processes.processes) - 1))
            self.select_process(self.process_list.currentRow())
        else:
            self.heading.setText('Активных процессов нет')
            self.summary.setText('Нет Live-метрик: запустите обучение или тест.')
            self.log.clear()
            self.chart.set_values([])
            self.trials.setRowCount(0)

    def select_process(self, _row: int) -> None:
        process = self.selected_process()
        if process is None:
            return
        self.heading.setText(process.name)
        self._render_process(process)

    def process_updated(self, process: ManagedProcess) -> None:
        if process is self.selected_process():
            self._render_process(process)

    def _render_process(self, process: ManagedProcess) -> None:
        snapshot = process.snapshot()
        text = (
            process.full_log_text()
            if hasattr(process, 'full_log_text')
            else ''.join(process.log)
        )
        metrics = (
            [
                {
                    'trial': event.trial,
                    'epoch': event.epoch,
                    'f1': event.f1,
                    'baseline': event.baseline,
                }
                for event in process.epochs
            ]
            if getattr(process, 'epochs', None)
            else parse_live_metrics(text)
        )
        if snapshot.kind == 'prediction':
            self._render_prediction_process(process, snapshot, text)
            return
        last = metrics[-1] if metrics else None
        f1 = last['f1'] if last else snapshot.last_f1
        baseline = last['baseline'] if last else snapshot.last_baseline
        gap = f1 - baseline if f1 is not None and baseline is not None else None
        self.summary.setText(
            f'{snapshot.status} · {snapshot.seconds} с · F1: '
            f'{f1:.4f} · Baseline: {baseline:.4f} · Отрыв: {gap:+.4f}\n'
            f'{" ".join(snapshot.command)}'
            if f1 is not None and baseline is not None and gap is not None
            else f'{snapshot.status} · {snapshot.seconds} с\n{" ".join(snapshot.command)}'
        )
        self.chart.set_values(
            [float(item['f1']) for item in metrics],
            float(baseline) if baseline is not None else None,
        )
        self.trials.setRowCount(len(metrics))
        for row, item in enumerate(metrics):
            values = (
                item['trial'], item['epoch'], item['f1'], item['baseline'],
                item['f1'] - item['baseline'] if item['baseline'] is not None else None,
            )
            for column, value in enumerate(values):
                display = f'{value:.4f}' if isinstance(value, float) else ('—' if value is None else str(value))
                self.trials.setItem(row, column, QTableWidgetItem(display))
        self._set_colored_log(text)

    def _render_prediction_process(
        self,
        process: ManagedProcess,
        snapshot,
        text: str,
    ) -> None:
        completed = [
            event
            for event in getattr(process, 'predictions', [])
            if event.f1 is not None
        ]
        last = completed[-1] if completed else None
        scene_progress = ''
        if last is not None and last.scene_index is not None and last.scene_total is not None:
            scene_progress = f' · Сцена {last.scene_index}/{last.scene_total}'
        self.summary.setText(
            f'{snapshot.status} · {snapshot.seconds} с{scene_progress}'
            + (f' · F1: {last.f1:.4f}' if last is not None and last.f1 is not None else '')
            + f'\n{" ".join(snapshot.command)}'
        )
        self.chart.set_values([float(event.f1) for event in completed if event.f1 is not None])
        self.trials.setColumnCount(5)
        self.trials.setHorizontalHeaderLabels(
            ['Сцена', 'F1', 'Precision', 'Recall', 'Baseline']
        )
        self.trials.setRowCount(len(completed))
        for row, event in enumerate(completed):
            values = (
                event.scene_name or '—',
                event.f1,
                event.precision,
                event.recall,
                event.baseline,
            )
            for column, value in enumerate(values):
                display = (
                    f'{value:.4f}'
                    if isinstance(value, float)
                    else ('—' if value is None else str(value))
                )
                self.trials.setItem(row, column, QTableWidgetItem(display))
        self._set_colored_log(text)

    def _set_colored_log(self, text: str) -> None:
        scrollbar = self.log.verticalScrollBar()
        old_value = scrollbar.value()
        self.log.clear()
        cursor = self.log.textCursor()
        for line in text.splitlines(keepends=True):
            fmt = QTextCharFormat()
            lowered = line.casefold()
            if 'error' in lowered or 'traceback' in lowered or 'ошибк' in lowered:
                fmt.setForeground(QColor('#FF7B84'))
            elif 'f1=' in lowered:
                fmt.setForeground(QColor('#77E2A8'))
            elif line.startswith('==='):
                fmt.setForeground(QColor('#F4C66A'))
            else:
                fmt.setForeground(QColor('#D6E2F4'))
            cursor.insertText(line, fmt)
        if self.autoscroll.isChecked():
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log.setTextCursor(cursor)
        else:
            scrollbar.setValue(old_value)

    def stop_process(self) -> None:
        process = self.selected_process()
        if process is None:
            return
        if self._stop_process is not process:
            self._stop_process = process
            self.stop_button.setText('Ещё раз — остановить')
            QTimer.singleShot(5000, self._reset_stop)
            return
        self._reset_stop()
        process.stop()

    def _reset_stop(self) -> None:
        self._stop_process = None
        self.stop_button.setText('Остановить')

    def restart_process(self) -> None:
        process = self.selected_process()
        if process is not None:
            process.restart()

    def copy_log(self) -> None:
        process = self.selected_process()
        if process is not None:
            text = (
                process.full_log_text()
                if hasattr(process, 'full_log_text')
                else ''.join(process.log)
            )
            QApplication.clipboard().setText(text)

    def continue_in_form(self) -> None:
        process = self.selected_process()
        if process is None:
            return
        dialog = LaunchDialog(self.context, parent=self)
        command = list(process.command)
        mapping = {'--target': 'target', '--features': 'features', '--output-dir': 'run_name'}
        for flag, name in mapping.items():
            if flag not in command:
                continue
            value = command[command.index(flag) + 1]
            widget = getattr(dialog, name, None)
            if name == 'run_name':
                value = Path(value).name
            if widget is not None and hasattr(widget, 'setText'):
                widget.setText(value)
            elif widget is not None and hasattr(widget, 'setCurrentText'):
                widget.setCurrentText(value)
        dialog.exec()


class AdvancedTestingPage(StudioPage):
    """Многосценовое тестирование и 3D-просмотр ошибок."""

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.models: list[ArtifactInfo] = []
        self.datasets: list[DatasetInfo] = []
        self.result_paths: list[Path] = []
        self.current_result: dict[str, Any] = {}
        self.current_result_path: Path | None = None
        self._cloud: PointCloudData | None = None
        self._prediction: np.ndarray | None = None
        self._target = ''
        refresh = QPushButton('Перечитать')
        refresh.setObjectName('secondaryButton')
        refresh.clicked.connect(self.refresh)
        self.add_header(
            'Тестирование',
            'Модель × равномерно выбранные сцены → predict_scene.py → метрики и 3D.',
            [refresh],
        )
        body = QHBoxLayout()
        body.setSpacing(12)
        controls, controls_layout = panel()
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(380)
        controls_layout.addWidget(title_label('Запуск теста'))
        controls_layout.addWidget(QLabel('Модель'))
        self.model_combo = QComboBox()
        controls_layout.addWidget(self.model_combo)
        load = QPushButton('Загрузить .pt в models/')
        load.setObjectName('secondaryButton')
        load.clicked.connect(self.import_model)
        controls_layout.addWidget(load)
        controls_layout.addWidget(QLabel('Датасет'))
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self.update_scene_limit)
        controls_layout.addWidget(self.dataset_combo)
        controls_layout.addWidget(QLabel('Число сцен'))
        self.scene_count = QSpinBox()
        self.scene_count.setRange(1, 1)
        controls_layout.addWidget(self.scene_count)
        run = QPushButton('Тестировать')
        run.setObjectName('primaryButton')
        run.clicked.connect(self.run_test)
        controls_layout.addWidget(run)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(title_label('Результаты'))
        self.result_list = QListWidget()
        self.result_list.currentRowChanged.connect(self.select_result)
        controls_layout.addWidget(self.result_list, 1)
        body.addWidget(controls)

        details, details_layout = panel()
        details.setMinimumWidth(350)
        details_layout.addWidget(title_label('Метрики'))
        self.result_scene = QComboBox()
        self.result_scene.currentIndexChanged.connect(self.select_result_scene)
        details_layout.addWidget(self.result_scene)
        self.metrics = QLabel('Результатов пока нет.')
        self.metrics.setObjectName('runSummary')
        self.metrics.setWordWrap(True)
        details_layout.addWidget(self.metrics)
        self.confusion = QTableWidget(2, 2)
        self.confusion.setHorizontalHeaderLabels(['Пред. фон', 'Пред. цель'])
        self.confusion.setVerticalHeaderLabels(['Ист. фон', 'Ист. цель'])
        self.confusion.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.confusion.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.confusion.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.confusion.setMaximumHeight(150)
        details_layout.addWidget(self.confusion)
        self.scene_table = QTableWidget()
        self.scene_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        details_layout.addWidget(self.scene_table, 1)
        body.addWidget(details, 1)

        preview, preview_layout = panel()
        preview.setMinimumWidth(430)
        header = QHBoxLayout()
        header.addWidget(title_label('3D-результат'))
        header.addStretch(1)
        self.mode_buttons: dict[str, QPushButton] = {}
        for key, text in (('classes', 'Классы'), ('prediction', 'Предсказание'), ('errors', 'Ошибки')):
            button = QPushButton(text)
            button.setObjectName('segmentButton')
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, mode=key: self.set_visual_mode(mode))
            header.addWidget(button)
            self.mode_buttons[key] = button
        self.mode_buttons['classes'].setChecked(True)
        preview_layout.addLayout(header)
        self.canvas = PointCloudCanvas()
        self.canvas.setMinimumSize(380, 320)
        preview_layout.addWidget(self.canvas, 1)
        self.legend = QLabel('Выберите результ и сцену.')
        self.legend.setObjectName('legendText')
        self.legend.setWordWrap(True)
        preview_layout.addWidget(self.legend)
        body.addWidget(preview, 1)
        self.root.addLayout(body, 1)
        self.refresh()

    def refresh(self) -> None:
        root = self.context.project_root
        self.models = []
        self.datasets = []
        self.result_paths = []
        self.model_combo.clear()
        self.dataset_combo.clear()
        self.result_list.clear()
        if root is None:
            return
        self.models = [item for item in scan_artifacts(root) if item.category == 'Модели']
        for item in self.models:
            self.model_combo.addItem(f'{item.run_name} · {item.path.name}', str(item.path))
        self.datasets = scan_datasets(root)
        for dataset in self.datasets:
            self.dataset_combo.addItem(f'{dataset.path.name} · {len(dataset.scenes)} сцен', str(dataset.path))
        tests = root / 'tests'
        if tests.is_dir():
            self.result_paths = sorted(tests.rglob('scenes__*.json'), reverse=True)
        for path in self.result_paths:
            self.result_list.addItem(str(path.relative_to(tests)))
        self.update_scene_limit()
        if self.result_paths:
            self.result_list.setCurrentRow(0)

    def update_scene_limit(self) -> None:
        index = self.dataset_combo.currentIndex()
        count = len(self.datasets[index].scenes) if 0 <= index < len(self.datasets) else 1
        self.scene_count.setMaximum(max(1, count))
        self.scene_count.setValue(min(self.scene_count.value(), max(1, count)))

    def import_model(self) -> None:
        root = self.context.project_root
        if root is None:
            return
        selected, _filter = QFileDialog.getOpenFileName(self, 'Загрузить модель', '', 'PyTorch (*.pt)')
        if not selected:
            return
        source = Path(selected)
        models = root / 'models'
        models.mkdir(parents=True, exist_ok=True)
        target = models / source.name
        if target.exists():
            QMessageBox.warning(self, 'Модель уже есть', str(target))
            return
        try:
            shutil.copy2(source, target)
        except OSError as error:
            QMessageBox.warning(self, 'Не удалось скопировать', str(error))
            return
        self.refresh()

    def run_test(self) -> None:
        root = self.context.project_root
        python = self.context.python_executable()
        model_data = self.model_combo.currentData()
        index = self.dataset_combo.currentIndex()
        if root is None or python is None or not model_data or not 0 <= index < len(self.datasets):
            QMessageBox.warning(self, 'Не готово к тесу', 'Выберите модель, датасет и проверьте окружение.')
            return
        dataset = self.datasets[index]
        count = min(self.scene_count.value(), len(dataset.scenes))
        indices = np.linspace(0, len(dataset.scenes) - 1, count, dtype=int)
        scenes = [dataset.scenes[int(i)].path for i in sorted(set(indices.tolist()))]
        checkpoint = Path(model_data)
        tag = datetime.now().strftime('%Y%m%d-%H%M%S')
        output = root / 'tests' / f'run-{checkpoint.stem}__{dataset.path.name}'
        arguments = [
            str(root / 'predict_scene.py'), '--checkpoint', str(checkpoint), '--scene',
            *(str(scene) for scene in scenes), '--out-dir', str(output), '--tag', tag,
        ]
        process = self.context.processes.create(f'test:{checkpoint.stem}', str(python), arguments, root)
        process.finished.connect(lambda _code: self.refresh())
        process.start()

    def select_result(self, row: int) -> None:
        if not 0 <= row < len(self.result_paths):
            return
        path = self.result_paths[row]
        result_data = _read_json(path)
        self.current_result_path = path
        self.current_result = result_data
        self._target = (
            str(result_data.get('model', {}).get('target', ''))
            if isinstance(result_data.get('model'), dict)
            else ''
        )
        scenes = (
            result_data.get('scenes')
            if isinstance(result_data.get('scenes'), list)
            else []
        )
        self.result_scene.blockSignals(True)
        self.result_scene.clear()
        for scene in scenes:
            if isinstance(scene, dict):
                self.result_scene.addItem(str(scene.get('scene', 'Сцена')), scene)
        self.result_scene.blockSignals(False)
        columns = ['scene', 'f1', 'precision', 'recall', 'baseline_f1', 'not_covered']
        self.scene_table.setColumnCount(len(columns))
        self.scene_table.setHorizontalHeaderLabels(columns)
        self.scene_table.setRowCount(len(scenes))
        for row_index, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                continue
            for column, key in enumerate(columns):
                cell_value = scene.get(key, '')
                text = (
                    f'{cell_value:.4f}'
                    if isinstance(cell_value, float)
                    else str(cell_value)
                )
                self.scene_table.setItem(row_index, column, QTableWidgetItem(text))
        if scenes:
            best = max((scene for scene in scenes if isinstance(scene, dict)), key=lambda item: float(item.get('f1', -1)), default={})
            self.metrics.setText(
                f'Готово: {result_data.get("done", len(scenes))}/'
                f'{result_data.get("total", len(scenes))} · '
                f'Лучшая сцена: {best.get("scene", "—")} · F1 {float(best.get("f1", 0)):.4f}'
            )
            self.result_scene.setCurrentIndex(0)
            self.select_result_scene(0)
        else:
            self.metrics.setText('Файл результата пока не содержит готовых сцен.')

    def select_result_scene(self, index: int) -> None:
        scene = self.result_scene.itemData(index)
        if not isinstance(scene, dict):
            return
        confusion = scene.get('confusion')
        if isinstance(confusion, list) and len(confusion) == 2:
            for row in range(2):
                for column in range(2):
                    try:
                        value = int(confusion[row][column])
                    except (IndexError, TypeError, ValueError):
                        value = 0
                    self.confusion.setItem(row, column, QTableWidgetItem(f'{value:,}'.replace(',', ' ')))
        self.metrics.setText(
            f'{scene.get("scene", "—")} · F1 {float(scene.get("f1", 0)):.4f} · '
            f'Precision {float(scene.get("precision", 0)):.4f} · '
            f'Recall {float(scene.get("recall", 0)):.4f} · '
            f'Baseline {float(scene.get("baseline_f1", 0)):.4f}'
        )
        self._load_result_cloud(scene)

    def _load_result_cloud(self, scene: dict[str, Any]) -> None:
        if self.current_result_path is None:
            return
        scene_path = Path(str(scene.get('path', '')))
        if not scene_path.is_absolute() and self.context.project_root is not None:
            scene_path = self.context.project_root / scene_path
        tag = self.current_result_path.stem.removeprefix('scenes__')
        prediction = self.current_result_path.parent / f'{scene_path.stem}__{tag}.npy'
        if not scene_path.is_file() or not prediction.is_file():
            self.legend.setText('Нет сцены или .npy-предсказания для 3D-просмотра.')
            return
        try:
            cloud = load_point_cloud(scene_path, max_points=250_000)
            raw = np.load(prediction, mmap_mode='r', allow_pickle=False)
            if raw.ndim != 1 or len(raw) != cloud.source_count:
                raise ValueError('длина предсказания не совпадает со сценой')
            if cloud.rendered_count < cloud.source_count:
                indices = np.linspace(0, cloud.source_count - 1, cloud.rendered_count, dtype=np.int64)
                selected = np.asarray(raw[indices], dtype=np.uint8)
            else:
                selected = np.asarray(raw, dtype=np.uint8)
        except (OSError, TypeError, ValueError) as error:
            self.legend.setText(f'Не удалось открыть 3D-результат: {error}')
            return
        self._cloud = cloud
        self._prediction = selected
        self.set_visual_mode(next((key for key, button in self.mode_buttons.items() if button.isChecked()), 'classes'))

    def set_visual_mode(self, mode: str) -> None:
        for key, button in self.mode_buttons.items():
            button.setChecked(key == mode)
        if self._cloud is None:
            return
        classes = self._cloud.classes
        legend = 'Исходные классы'
        if mode == 'prediction' and self._prediction is not None:
            classes = np.where(self._prediction == 255, 7, np.where(self._prediction == 1, 3, 1)).astype(np.int32)
            legend = 'Серый: фон · Зелёный: цель · Фиолетовый: не покрыто'
        elif mode == 'errors' and self._prediction is not None:
            targets = TARGET_CLASS_IDS.get(self._target, ())
            truth = np.isin(self._cloud.classes, targets)
            pred = self._prediction
            classes = np.full(len(pred), 1, dtype=np.int32)  # TN
            classes[(pred == 1) & truth] = 3  # TP, зелёный
            classes[(pred == 1) & ~truth] = 6  # FP, красный
            classes[(pred == 0) & truth] = 9  # FN, светло-голубой
            classes[pred == 255] = 7  # не покрыто
            legend = 'TP: зелёный · FP: красный · FN: голубой · TN: серый · не покрыто: фиолетовый'
        displayed = PointCloudData(
            path=self._cloud.path,
            points=self._cloud.points,
            classes=np.asarray(classes, dtype=np.int32),
            center=self._cloud.center,
            bounds_min=self._cloud.bounds_min,
            bounds_max=self._cloud.bounds_max,
            source_count=self._cloud.source_count,
            invalid_count=self._cloud.invalid_count,
            rgb=self._cloud.rgb,
        )
        self.canvas.set_cloud(displayed)
        self.legend.setText(legend)


class AdvancedArtifactsPage(StudioPage):
    """Документы, модели и логи без дублирования аналитики прогонов."""

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.artifacts: list[ArtifactInfo] = []
        refresh = QPushButton('Перечитать')
        refresh.setObjectName('secondaryButton')
        refresh.clicked.connect(self.refresh)
        self.add_header('Артефакты', 'Только файлы на диске: Markdown, модели .pt и логи.', [refresh])
        body = QHBoxLayout()
        body.setSpacing(12)
        left, left_layout = panel()
        left.setMinimumWidth(360)
        left.setMaximumWidth(470)
        self.category = QComboBox()
        self.category.addItems(('Все', 'Документы', 'Модели', 'Логи'))
        self.category.currentTextChanged.connect(self.populate)
        left_layout.addWidget(self.category)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.select_artifact)
        left_layout.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        open_button = QPushButton('Открыть файл')
        open_button.setObjectName('secondaryButton')
        open_button.clicked.connect(self.open_file)
        copy = QPushButton('Копировать')
        copy.setObjectName('secondaryButton')
        copy.clicked.connect(self.copy_artifact)
        self.test_button = QPushButton('Тестировать')
        self.test_button.setObjectName('primaryButton')
        self.test_button.clicked.connect(self.test_model)
        buttons.addWidget(open_button)
        buttons.addWidget(copy)
        buttons.addWidget(self.test_button)
        left_layout.addLayout(buttons)
        body.addWidget(left)
        right, right_layout = panel()
        self.heading = title_label('Предпросмотр')
        right_layout.addWidget(self.heading)
        self.preview_stack = QStackedWidget()
        self.markdown = QTextBrowser()
        self.markdown.setObjectName('artifactMarkdown')
        self.markdown.setOpenExternalLinks(True)
        self.plain = QPlainTextEdit()
        self.plain.setReadOnly(True)
        self.plain.setObjectName('artifactPreview')
        self.preview_stack.addWidget(self.markdown)
        self.preview_stack.addWidget(self.plain)
        right_layout.addWidget(self.preview_stack, 1)
        body.addWidget(right, 1)
        self.root.addLayout(body, 1)
        self.refresh()

    def visible(self) -> list[ArtifactInfo]:
        category = self.category.currentText()
        return [item for item in self.artifacts if category == 'Все' or item.category == category]

    def refresh(self) -> None:
        self.artifacts = scan_artifacts(self.context.project_root) if self.context.project_root else []
        self.populate()

    def populate(self) -> None:
        self.list.clear()
        for artifact in self.visible():
            prefix = f'{artifact.run_name} · ' if artifact.run_name else ''
            self.list.addItem(f'{prefix}{artifact.path.name}')
        if self.list.count():
            self.list.setCurrentRow(0)
        else:
            self.heading.setText('Артефакты не найдены')
            self.plain.setPlainText('Файлы появятся после обучения или тестирования.')
            self.preview_stack.setCurrentWidget(self.plain)

    def selected(self) -> ArtifactInfo | None:
        row = self.list.currentRow()
        values = self.visible()
        return values[row] if 0 <= row < len(values) else None

    def select_artifact(self, _row: int) -> None:
        artifact = self.selected()
        if artifact is None:
            return
        self.heading.setText(artifact.path.name)
        self.test_button.setEnabled(artifact.category == 'Модели')
        if artifact.category == 'Документы':
            self.markdown.setMarkdown(read_text_preview(artifact.path))
            self.preview_stack.setCurrentWidget(self.markdown)
        elif artifact.category == 'Логи':
            self.plain.setPlainText(read_text_preview(artifact.path))
            self.preview_stack.setCurrentWidget(self.plain)
        else:
            try:
                size = human_bytes(artifact.path.stat().st_size)
            except OSError:
                size = 'недоступен'
            self.plain.setPlainText(
                f'Модель PyTorch\n\nПуть: {artifact.path}\nРазмер: {size}\n'
                f'Прогон: {artifact.run_name or "—"}'
            )
            self.preview_stack.setCurrentWidget(self.plain)

    def open_file(self) -> None:
        artifact = self.selected()
        if artifact is not None:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(artifact.path)))

    def copy_artifact(self) -> None:
        artifact = self.selected()
        if artifact is None:
            return
        text = read_text_preview(artifact.path) if artifact.category != 'Модели' else str(artifact.path)
        QApplication.clipboard().setText(text)

    def test_model(self) -> None:
        artifact = self.selected()
        if artifact is not None and artifact.category == 'Модели':
            TestModelDialog(self.context, checkpoint=artifact.path, parent=self).exec()


ADVANCED_STUDIO_PAGES = {
    'runs': AdvancedRunsPage,
    'live': AdvancedLivePage,
    'testing': AdvancedTestingPage,
    'artifacts': AdvancedArtifactsPage,
}

