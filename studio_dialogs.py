"""Диалоги реального запуска обучения и тестирования моделей."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcessEnvironment, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from studio_context import StudioContext

TARGETS = ('ground', 'vegetation', 'noise', 'building', 'water')
RUN_ALL_TASKS = ('2', '3', '4', '5', '6', '7', 'final')
FIXED_PARAM_DEFAULTS: dict[str, Any] = {
    'lr': 0.001,
    'weight_decay': 0.0001,
    'dropout': 0.2,
    'batch_size': 2,
    'n_sa_layers': 3,
    'n_points_l0': 2048,
    'radius_l0': 0.1,
    'nsample': 32,
    'use_MSG': False,
}
STRUCTURED_FLAGS = {
    '--data-dir',
    '--output-dir',
    '--target',
    '--features',
    '--seed',
    '--n-trials',
    '--epochs',
    '--patch-points',
    '--patches-per-scene',
    '--workers',
    '--fixed-params',
    '--loss',
    '--focal-gamma',
    '--lovasz-weight',
    '--class-weight-power',
    '--equibatch',
    '--equibatch-positive-fraction',
    '--amp',
    '--data',
    '--final-data',
    '--runs',
    '--tasks',
    '--final-epochs',
}


def format_command(command: list[str]) -> str:
    if os.name == 'nt':
        return subprocess.list2cmdline(command)
    import shlex

    return shlex.join(command)


def parse_extra_arguments(value: str) -> list[str]:
    """Разобрать дополнительные CLI-аргументы без запуска shell."""
    if not value.strip():
        return []
    if os.name == 'nt':
        # POSIX-режим считает ``\`` экранированием и ломает
        # обычные Windows-пути вроде C:\\data\\exports. Режим
        # posix=False сохраняет пути, но оставляет обрамляющие
        # кавычки, поэтому снимаем только целую внешнюю пару.
        raw_arguments = shlex.split(value, posix=False)
        arguments = [
            argument[1:-1]
            if len(argument) >= 2
            and argument[0] == argument[-1]
            and argument[0] in {'"', "'"}
            else argument
            for argument in raw_arguments
        ]
    else:
        arguments = shlex.split(value, posix=True)
    if any('\x00' in argument for argument in arguments):
        raise ValueError('аргументы не могут содержать NUL')
    return arguments


def validate_fixed_params(value: object) -> dict[str, Any]:
    """Проверить ручные параметры по реальному SPACE_VERSION=7."""
    if not isinstance(value, dict):
        raise TypeError('конфигурация должна быть JSON-объектом')
    missing = set(FIXED_PARAM_DEFAULTS) - set(value)
    extra = set(value) - set(FIXED_PARAM_DEFAULTS)
    if missing or extra:
        parts = []
        if missing:
            parts.append('нет: ' + ', '.join(sorted(missing)))
        if extra:
            parts.append('лишние: ' + ', '.join(sorted(extra)))
        raise ValueError('неверный набор fixed-params (' + '; '.join(parts) + ')')

    result = dict(value)
    ranges = {
        'lr': (1e-4, 1e-2),
        'weight_decay': (1e-5, 1e-1),
        'dropout': (0.0, 0.5),
        'batch_size': (2, 4),
        'n_sa_layers': (2, 4),
        'n_points_l0': (1024, 4096),
        'radius_l0': (0.05, 0.5),
        'nsample': (16, 64),
    }
    integer_keys = {'batch_size', 'n_sa_layers', 'n_points_l0', 'nsample'}
    for key, (minimum, maximum) in ranges.items():
        raw = result[key]
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise TypeError(f'{key}: ожидается число')
        if key in integer_keys and not isinstance(raw, int):
            raise TypeError(f'{key}: ожидается целое число')
        if not math.isfinite(float(raw)) or not minimum <= raw <= maximum:
            raise ValueError(f'{key}: значение должно быть от {minimum:g} до {maximum:g}')
    if not isinstance(result['use_MSG'], bool):
        raise TypeError('use_MSG: ожидается true или false')
    return result


def write_json_exclusive(path: Path, value: object) -> None:
    """Записать JSON атомарно и не перезаписывать чужую конфигурацию."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        if path.read_text(encoding='utf-8') == payload:
            return
        raise FileExistsError(f'файл уже существует и отличается: {path}')
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def launch_compatibility_differences(output: Path, expected: dict[str, object]) -> list[str]:
    """Без Optuna проверить параметры продолжаемого study по split.json."""
    if not output.exists():
        return []
    if not (output / 'study.db').is_file():
        return ['каталог существует, но study.db отсутствует']
    split = output / 'split.json'
    if not split.is_file():
        return ['нельзя проверить совместимость: нет split.json']
    try:
        stored = json.loads(split.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        return [f'не читается split.json: {error}']
    differences = []
    for key, requested in expected.items():
        actual = stored.get(key)
        if key == 'data_dir' and isinstance(actual, str):
            try:
                equal = Path(actual).expanduser().resolve() == Path(str(requested)).expanduser().resolve()
            except OSError:
                equal = actual == requested
        else:
            equal = actual == requested
        if not equal:
            differences.append(f'{key}: было {actual!r}, запрошено {requested!r}')
    return differences


class PathField(QWidget):
    changed = Signal(str)

    def __init__(
        self,
        value: str = '',
        *,
        directory: bool = True,
        file_filter: str = 'Все файлы (*)',
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.directory = directory
        self.file_filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.edit = QLineEdit(value)
        self.button = QPushButton('Обзор…')
        self.button.setObjectName('secondaryButton')
        self.button.clicked.connect(self._browse)
        self.edit.textChanged.connect(self.changed)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value)

    def _browse(self) -> None:
        start = self.text()
        if self.directory:
            selected = QFileDialog.getExistingDirectory(self, 'Выберите папку', start)
        else:
            selected, _filter = QFileDialog.getOpenFileName(
                self,
                'Выберите файл',
                start,
                self.file_filter,
            )
        if selected:
            self.setText(selected)


class LaunchDialog(QDialog):
    process_started = Signal(object)

    def __init__(
        self,
        context: StudioContext,
        dataset: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.context = context
        self.setWindowTitle('Новый запуск ML Studio')
        self.resize(820, 820)
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(14)
        title = QLabel('Новый запуск')
        title.setObjectName('dialogTitle')
        subtitle = QLabel(
            'Одиночный подбор запускает hpo_pointnet.py, сценарий задач — '
            'run_all.py. Команда внизу совпадает с запускаемой.'
        )
        subtitle.setObjectName('dialogSubtitle')
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        form_host = QWidget()
        self.form = QFormLayout(form_host)
        self.form.setSpacing(10)
        project = context.project_root
        default_dataset = dataset or (project / 'exports' if project else Path())
        default_final = project / 'exports_final' if project else Path()
        default_runs = project / 'runs' if project else Path()

        self.mode = QComboBox()
        self.mode.addItem('Подбор для одного класса', 'single')
        self.mode.addItem('Сценарий задач run_all.py', 'scenario')
        self.dataset = PathField(str(default_dataset))
        self.final_dataset = PathField(str(default_final))
        self.runs_dir = PathField(str(default_runs))
        self.tasks = QLineEdit('2 3 4 5 6 7')
        self.tasks.setPlaceholderText('2 3 4 5 6 7 final')
        self.target = QComboBox()
        self.target.addItems(TARGETS)
        self.features = QComboBox()
        self.features.addItems(('xyz', 'xyz_hag'))
        self.run_name = QLineEdit('ground_xyz')
        self.seed = self._spin(0, 1_000_000, 42)
        self.trials = self._spin(1, 10_000, 25)
        self.epochs = self._spin(1, 10_000, 10)
        self.patch_points = self._spin(128, 10_000_000, 16_384)
        self.patches_per_scene = self._spin(1, 10_000, 2)
        self.workers = self._spin(0, 128, 0)
        self.loss = QComboBox()
        self.loss.addItems(('ce', 'focal', 'focal_lovasz'))
        self.focal_gamma = self._double_spin(0.01, 100.0, 2.0, 2)
        self.lovasz_weight = self._double_spin(0.0, 1.0, 0.5, 2)
        self.class_weight_power = self._double_spin(0.0, 1.0, 1.0, 2)
        self.equibatch = QCheckBox('Балансировать патчи цели и фона')
        self.equibatch.setChecked(True)
        self.target_fraction = self._double_spin(0.05, 0.95, 0.50, 2)
        self.target_fraction.setSingleStep(0.05)
        self.final_epochs = self._spin(1, 10_000, 30)
        self.device = QComboBox()
        self.device.addItem('Авто', 'auto')
        self.device.addItem('Только CPU', 'cpu')
        self.device.addItem('CUDA (GPU 0)', 'cuda')
        self.amp = QCheckBox('Использовать fp16 на CUDA')
        self.manual_params = QCheckBox('Своя конфигурация (--fixed-params)')
        self.fixed_params = QPlainTextEdit(
            json.dumps(FIXED_PARAM_DEFAULTS, ensure_ascii=False, indent=2)
        )
        self.fixed_params.setMaximumHeight(160)
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText('--boundary-space xy --boundary-weight 1.5')

        self.form.addRow('Режим', self.mode)
        self.form.addRow('Данные для подбора', self.dataset)
        self.form.addRow('Данные для финала', self.final_dataset)
        self.form.addRow('Каталог прогонов', self.runs_dir)
        self.form.addRow('Задачи', self.tasks)
        self.form.addRow('Целевой класс', self.target)
        self.form.addRow('Признаки', self.features)
        self.form.addRow('Имя запуска', self.run_name)
        self.form.addRow('Seed', self.seed)
        self.form.addRow('Trials', self.trials)
        self.form.addRow('Эпох на trial', self.epochs)
        self.form.addRow('Эпох финала', self.final_epochs)
        self.form.addRow('Точек в патче', self.patch_points)
        self.form.addRow('Патчей на сцену', self.patches_per_scene)
        self.form.addRow('Workers', self.workers)
        self.form.addRow('Loss', self.loss)
        self.form.addRow('Focal gamma', self.focal_gamma)
        self.form.addRow('Доля Lovász', self.lovasz_weight)
        self.form.addRow('Степень весов классов', self.class_weight_power)
        self.form.addRow('Баланс цель/фон', self.equibatch)
        self.form.addRow('Доля целевых патчей', self.target_fraction)
        self.form.addRow('Устройство', self.device)
        self.form.addRow('', self.amp)
        self.form.addRow('', self.manual_params)
        self.form.addRow('JSON fixed-params', self.fixed_params)
        self.form.addRow('Доп. аргументы CLI', self.extra_args)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(form_host)
        root.addWidget(scroll, 1)

        command_label = QLabel('Команда')
        command_label.setObjectName('sectionLabel')
        self.command_preview = QPlainTextEdit()
        self.command_preview.setObjectName('commandPreview')
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumHeight(112)
        root.addWidget(command_label)
        root.addWidget(self.command_preview)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton('Отмена')
        cancel.setObjectName('secondaryButton')
        cancel.clicked.connect(self.reject)
        start = QPushButton('Запустить')
        start.setObjectName('primaryButton')
        start.clicked.connect(self._start)
        buttons.addWidget(cancel)
        buttons.addWidget(start)
        root.addLayout(buttons)

        for widget in (
            self.mode,
            self.target,
            self.features,
            self.run_name,
            self.tasks,
            self.seed,
            self.trials,
            self.epochs,
            self.final_epochs,
            self.patch_points,
            self.patches_per_scene,
            self.workers,
            self.loss,
            self.focal_gamma,
            self.lovasz_weight,
            self.class_weight_power,
            self.equibatch,
            self.target_fraction,
            self.device,
            self.amp,
            self.manual_params,
            self.fixed_params,
            self.extra_args,
        ):
            signal = getattr(widget, 'currentTextChanged', None)
            if signal is None:
                signal = getattr(widget, 'textChanged', None)
            if signal is None:
                signal = getattr(widget, 'valueChanged', None)
            if signal is None:
                signal = getattr(widget, 'toggled', None)
            if signal is not None:
                signal.connect(self._update_command)
        self.dataset.changed.connect(self._update_command)
        self.final_dataset.changed.connect(self._update_command)
        self.runs_dir.changed.connect(self._update_command)
        self.target.currentTextChanged.connect(self._sync_run_name)
        self.features.currentTextChanged.connect(self._sync_run_name)
        self.mode.currentIndexChanged.connect(self._update_mode)
        self.manual_params.toggled.connect(self._update_mode)
        self.equibatch.toggled.connect(self.target_fraction.setEnabled)
        self._update_mode()
        self._update_command()

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setGroupSeparatorShown(True)
        return widget

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(0.05)
        widget.setValue(value)
        return widget

    def mode_name(self) -> str:
        return str(self.mode.currentData())

    def _update_mode(self, *_args) -> None:
        single = self.mode_name() == 'single'
        for field in (self.target, self.features, self.run_name, self.manual_params):
            self.form.setRowVisible(field, single)
        self.form.setRowVisible(self.fixed_params, single and self.manual_params.isChecked())
        for field in (self.tasks, self.final_dataset, self.final_epochs):
            self.form.setRowVisible(field, not single)
        self.fixed_params.setVisible(single and self.manual_params.isChecked())
        self._update_command()

    def _sync_run_name(self) -> None:
        self.run_name.setText(f'{self.target.currentText()}_{self.features.currentText()}')

    def _tasks(self) -> list[str]:
        return [value for value in self.tasks.text().split() if value]

    def output_directory(self) -> Path:
        return Path(self.runs_dir.text()).expanduser() / self.run_name.text().strip()

    def fixed_params_path(self) -> Path:
        safe_name = self.run_name.text().strip() or 'run'
        return Path(self.runs_dir.text()).expanduser() / '.gui_configs' / f'{safe_name}.fixed-params.json'

    def _training_arguments(self) -> list[str]:
        arguments = [
            '--seed',
            str(self.seed.value()),
            '--n-trials',
            str(self.trials.value()),
            '--epochs',
            str(self.epochs.value()),
            '--patch-points',
            str(self.patch_points.value()),
            '--patches-per-scene',
            str(self.patches_per_scene.value()),
            '--workers',
            str(self.workers.value()),
            '--loss',
            self.loss.currentText(),
            '--focal-gamma',
            f'{self.focal_gamma.value():g}',
            '--lovasz-weight',
            f'{self.lovasz_weight.value():g}',
            '--class-weight-power',
            f'{self.class_weight_power.value():g}',
        ]
        if self.equibatch.isChecked():
            arguments.extend(
                (
                    '--equibatch',
                    '--equibatch-positive-fraction',
                    f'{self.target_fraction.value():g}',
                )
            )
        if self.amp.isChecked():
            arguments.append('--amp')
        return arguments

    def command(self) -> list[str]:
        root = self.context.project_root
        python = self.context.python_executable()
        if root is None or python is None:
            return []
        if self.mode_name() == 'single':
            command = [
                str(python),
                str(root / 'hpo_pointnet.py'),
                '--data-dir',
                self.dataset.text(),
                '--output-dir',
                str(self.output_directory()),
                '--target',
                self.target.currentText(),
                '--features',
                self.features.currentText(),
                *self._training_arguments(),
            ]
            if self.manual_params.isChecked():
                command.extend(('--fixed-params', str(self.fixed_params_path())))
        else:
            command = [
                str(python),
                str(root / 'run_all.py'),
                '--tasks',
                *self._tasks(),
                '--data',
                self.dataset.text(),
                '--runs',
                self.runs_dir.text(),
                '--final-epochs',
                str(self.final_epochs.value()),
                *self._training_arguments(),
            ]
            if 'final' in self._tasks():
                command.extend(('--final-data', self.final_dataset.text()))
        command.extend(parse_extra_arguments(self.extra_args.text()))
        return command

    def display_command(self) -> str:
        try:
            rendered = format_command(self.command())
        except ValueError as error:
            return f'Ошибка в дополнительных аргументах: {error}'
        device = self.device.currentData()
        if device == 'auto' or not rendered:
            return rendered
        value = '' if device == 'cpu' else '0'
        if os.name == 'nt':
            return f'set "CUDA_VISIBLE_DEVICES={value}" && {rendered}'
        return f'CUDA_VISIBLE_DEVICES={shlex.quote(value)} {rendered}'

    def _update_command(self, *_args) -> None:
        self.command_preview.setPlainText(self.display_command())

    def _loss_description(self) -> str:
        loss = self.loss.currentText()
        if loss == 'ce':
            return 'ce'
        result = f'focal(gamma={self.focal_gamma.value():g})'
        if loss == 'focal_lovasz':
            weight = self.lovasz_weight.value()
            result = f'(1-{weight:g})*{result}+{weight:g}*lovasz'
        return result

    def _expected_compatibility(self) -> dict[str, object]:
        extras = parse_extra_arguments(self.extra_args.text())

        def option(flag: str, default: str) -> str:
            try:
                index = extras.index(flag)
            except ValueError:
                return default
            if index + 1 >= len(extras) or extras[index + 1].startswith('--'):
                raise ValueError(f'после {flag} нужно значение')
            return extras[index + 1]

        boundary_fraction = float(option('--boundary-patch-fraction', '0'))
        boundary_weight = float(option('--boundary-weight', '1'))
        boundary_enabled = boundary_fraction > 0 or boundary_weight > 1
        return {
            'data_dir': self.dataset.text(),
            'seed': self.seed.value(),
            'epochs': self.epochs.value(),
            'target': self.target.currentText(),
            'features': self.features.currentText(),
            'patch_points': self.patch_points.value(),
            'patches_per_scene': self.patches_per_scene.value(),
            'loss': self._loss_description(),
            'class_weight_power': self.class_weight_power.value(),
            'positive_fraction': (
                self.target_fraction.value()
                if self.equibatch.isChecked()
                else None
            ),
            'boundary_fraction': boundary_fraction,
            'boundary_weight': boundary_weight,
            'boundary_radius': float(option('--boundary-radius', '0.5')) if boundary_enabled else None,
            'boundary_space': option('--boundary-space', 'xyz') if boundary_enabled else None,
        }

    def _validate(self) -> str | None:
        root = self.context.project_root
        python = self.context.python_executable()
        if root is None:
            return 'сначала выберите проект'
        if python is None:
            return 'нет .venv311; подготовьте ML-окружение в разделе «Окружение»'
        script = root / ('hpo_pointnet.py' if self.mode_name() == 'single' else 'run_all.py')
        if not script.is_file():
            return f'не найден скрипт: {script}'
        dataset = Path(self.dataset.text()).expanduser()
        if not dataset.is_dir():
            return f'не найдены данные для подбора: {dataset}'
        runs = Path(self.runs_dir.text()).expanduser()
        if runs.exists() and not runs.is_dir():
            return f'путь прогонов не является папкой: {runs}'
        try:
            extras = parse_extra_arguments(self.extra_args.text())
        except ValueError as error:
            return f'неверные дополнительные аргументы: {error}'
        duplicate = sorted(
            flag
            for flag in STRUCTURED_FLAGS
            if flag in extras or any(value.startswith(flag + '=') for value in extras)
        )
        if duplicate:
            return 'эти аргументы уже заданы полями формы: ' + ', '.join(duplicate)

        if self.mode_name() == 'single':
            name = self.run_name.text().strip()
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
                return 'имя запуска может содержать только латиницу, цифры, ., _ и -'
            try:
                expected = self._expected_compatibility()
            except (IndexError, ValueError) as compatibility_error:
                return f'неверные аргументы обучения: {compatibility_error}'
            differences = launch_compatibility_differences(self.output_directory(), expected)
            if differences:
                return 'нельзя продолжить исследование:\n• ' + '\n• '.join(differences)
            if self.manual_params.isChecked():
                try:
                    validate_fixed_params(json.loads(self.fixed_params.toPlainText()))
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    return f'ошибка fixed-params: {error}'
        else:
            tasks = self._tasks()
            invalid = [value for value in tasks if value not in RUN_ALL_TASKS]
            if not tasks:
                return 'укажите хотя бы одну задачу'
            if invalid:
                return 'неизвестные задачи: ' + ', '.join(invalid)
            if len(set(tasks)) != len(tasks):
                return 'задачи не должны повторяться'
            if 'final' in tasks:
                final_data = Path(self.final_dataset.text()).expanduser()
                if not final_data.is_dir():
                    return f'не найдены данные для финала: {final_data}'
                conflicts = sorted(runs.glob('final_*')) + sorted(runs.glob('*.params.json')) if runs.is_dir() else []
                if conflicts:
                    shown = '\n• '.join(str(path) for path in conflicts[:8])
                    return 'финальный прогон не перезаписывает существующие артефакты:\n• ' + shown
        return None

    def _configure_environment(self, process) -> None:
        environment = QProcessEnvironment.systemEnvironment()
        device = self.device.currentData()
        if device == 'cpu':
            environment.insert('CUDA_VISIBLE_DEVICES', '')
        elif device == 'cuda':
            environment.insert('CUDA_VISIBLE_DEVICES', '0')
        environment.insert('PYTHONUNBUFFERED', '1')
        environment.insert('PYTHONIOENCODING', 'utf-8')
        process.process.setProcessEnvironment(environment)

    def _write_launch_metadata(self, command: list[str]) -> None:
        root = self.context.project_root
        if root is None:
            return
        payload = {
            'command': command,
            'display_command': self.display_command(),
            'device': self.device.currentData(),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'mode': self.mode_name(),
        }
        if self.mode_name() == 'single':
            output = self.output_directory()
            if not output.is_dir():
                return
            path = output / 'studio.json'
            if path.exists():
                path = output / f'studio-continue-{time.strftime("%Y%m%d-%H%M%S")}.json'
        else:
            path = Path(self.runs_dir.text()).expanduser() / '.studio_launches' / f'{time.strftime("%Y%m%d-%H%M%S")}.json'
        try:
            write_json_exclusive(path, payload)
        except (OSError, FileExistsError):
            pass

    def _start(self) -> None:
        root = self.context.project_root
        error = self._validate()
        if root is None or error is not None:
            QMessageBox.warning(self, 'Запуск невозможен', error or 'Проект не выбран')
            return
        runs = Path(self.runs_dir.text()).expanduser()
        runs.mkdir(parents=True, exist_ok=True)
        if self.mode_name() == 'single' and self.manual_params.isChecked():
            try:
                value = validate_fixed_params(json.loads(self.fixed_params.toPlainText()))
                write_json_exclusive(self.fixed_params_path(), value)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as fixed_error:
                QMessageBox.warning(self, 'Не удалось записать fixed-params', str(fixed_error))
                return
        command = self.command()
        process_name = self.run_name.text().strip() if self.mode_name() == 'single' else 'Сценарий ' + ' '.join(self._tasks())
        process = self.context.processes.create(
            process_name,
            command[0],
            command[1:],
            root,
        )
        self._configure_environment(process)
        process.start()
        QTimer.singleShot(1200, lambda: self._write_launch_metadata(command))
        self.process_started.emit(process)
        self.accept()


class TestModelDialog(QDialog):
    process_started = Signal(object)

    def __init__(
        self,
        context: StudioContext,
        scene: Path | None = None,
        checkpoint: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.context = context
        self.setWindowTitle('Тестирование модели')
        self.resize(720, 430)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(14)
        title = QLabel('Тестирование модели')
        title.setObjectName('dialogTitle')
        root.addWidget(title)
        form = QFormLayout()
        self.checkpoint = PathField(
            str(checkpoint or ''),
            directory=False,
            file_filter='Модели PyTorch (*.pt);;Все файлы (*)',
        )
        self.scene = PathField(
            str(scene or ''),
            directory=False,
            file_filter='Сцены NumPy (*.npy);;Все файлы (*)',
        )
        default_out = (
            context.project_root / 'tests' if context.project_root is not None else Path()
        )
        self.output = PathField(str(default_out))
        self.tag = QLineEdit('gui')
        form.addRow('Модель', self.checkpoint)
        form.addRow('Сцена', self.scene)
        form.addRow('Папка результатов', self.output)
        form.addRow('Тег', self.tag)
        root.addLayout(form)
        self.command_preview = QPlainTextEdit()
        self.command_preview.setObjectName('commandPreview')
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumHeight(100)
        root.addWidget(self.command_preview)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton('Отмена')
        cancel.setObjectName('secondaryButton')
        cancel.clicked.connect(self.reject)
        start = QPushButton('Запустить тест')
        start.setObjectName('primaryButton')
        start.clicked.connect(self._start)
        buttons.addWidget(cancel)
        buttons.addWidget(start)
        root.addLayout(buttons)
        for field in (self.checkpoint, self.scene, self.output):
            field.changed.connect(self._update_command)
        self.tag.textChanged.connect(self._update_command)
        self._update_command()

    def command(self) -> list[str]:
        root = self.context.project_root
        python = self.context.python_executable()
        if root is None or python is None:
            return []
        return [
            str(python),
            str(root / 'predict_scene.py'),
            '--checkpoint',
            self.checkpoint.text(),
            '--scene',
            self.scene.text(),
            '--out-dir',
            self.output.text(),
            '--tag',
            self.tag.text().strip() or 'gui',
        ]

    def _update_command(self, *_args) -> None:
        self.command_preview.setPlainText(format_command(self.command()))

    def _start(self) -> None:
        root = self.context.project_root
        command = self.command()
        if root is None:
            QMessageBox.warning(self, 'Проект не выбран', 'Сначала выберите проект.')
            return
        if not command:
            QMessageBox.warning(
                self,
                'Не готово ML-окружение',
                'Создайте .venv311 с помощью run_all.cmd перед запуском теста.',
            )
            return
        for label, value in (
            ('Модель', self.checkpoint.text()),
            ('Сцена', self.scene.text()),
        ):
            if not Path(value).expanduser().is_file():
                QMessageBox.warning(self, f'{label} не найдена', value)
                return
        Path(self.output.text()).expanduser().mkdir(parents=True, exist_ok=True)
        process = self.context.processes.create(
            f'test:{Path(self.scene.text()).stem}',
            command[0],
            command[1:],
            root,
        )
        process.start()
        self.process_started.emit(process)
        self.accept()

