"""Рабочие разделы ML Studio внутри бело-синей оболочки PointCloud."""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from point_cloud_canvas import PointCloudCanvas
from point_cloud_data import PointCloudData, class_style_for
from point_cloud_viewer import PointCloudLoadWorker, PointCloudViewerDialog
from studio_context import StudioContext
from studio_data import (
    ArtifactInfo,
    DatasetInfo,
    RunInfo,
    create_mini_dataset,
    human_bytes,
    human_count,
    read_text_preview,
    scan_artifacts,
    scan_datasets,
    scan_runs,
)
from studio_dialogs import LaunchDialog, TestModelDialog
from studio_environment import EnvironmentInstaller, EnvironmentStatus, inspect_environment
from studio_processes import ManagedProcess
from studio_widgets import ProjectRequired, open_in_file_manager


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child is not None:
            clear_layout(child)


def panel() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName('studioCard')
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)
    return frame, layout


def title_label(text: str, object_name: str = 'panelTitle') -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    return label


class StudioPage(QWidget):
    project_selection_requested = Signal()

    def __init__(self, context: StudioContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.root.setSpacing(14)
        self.context.project_changed.connect(self.on_project_changed)

    def add_header(
        self,
        title: str,
        subtitle: str,
        actions: list[QPushButton] | None = None,
    ) -> None:
        row = QHBoxLayout()
        texts = QVBoxLayout()
        texts.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName('studioPageTitle')
        description = QLabel(subtitle)
        description.setObjectName('studioPageSubtitle')
        description.setWordWrap(True)
        texts.addWidget(heading)
        texts.addWidget(description)
        row.addLayout(texts, 1)
        for button in actions or []:
            row.addWidget(button, alignment=Qt.AlignmentFlag.AlignBottom)
        self.root.addLayout(row)

    def show_project_required(self) -> None:
        empty = ProjectRequired()
        empty.choose_requested.connect(self.project_selection_requested)
        self.root.addWidget(empty, 1)

    def on_project_changed(self, _root: Path) -> None:
        self.refresh()

    def refresh(self) -> None:
        raise NotImplementedError


class EnvironmentCheckWorker(QObject):
    """Проверяет тяжёлое ML-окружение вне GUI-потока."""

    finished = Signal(object, object)

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    @Slot()
    def run(self) -> None:
        self.finished.emit(self.root, inspect_environment(self.root))


class MiniDatasetWorker(QObject):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, dataset: DatasetInfo, count: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.count = count

    @Slot()
    def run(self) -> None:
        try:
            target = create_mini_dataset(
                self.dataset,
                self.count,
                progress=self._report_progress,
            )
        except InterruptedError:
            self.failed.emit('Создание мини-копии отменено.')
        except (OSError, ValueError) as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(target)

    def _report_progress(self, done: int, total: int, path: Path) -> bool:
        self.progress.emit(done, total, path.name)
        return not QThread.currentThread().isInterruptionRequested()


class ProjectsPage(StudioPage):
    project_opened = Signal(object)

    def __init__(self, context: StudioContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        choose = QPushButton('Открыть проект')
        choose.setObjectName('primaryButton')
        choose.clicked.connect(self.choose_project)
        self.add_header(
            'Проекты',
            'Корень проекта содержит hpo_pointnet.py, датасеты exports* и каталог runs/.',
            [choose],
        )
        card, layout = panel()
        layout.addWidget(title_label('Недавние проекты'))
        self.current_label = QLabel('Проект не выбран')
        self.current_label.setObjectName('projectPath')
        self.current_label.setWordWrap(True)
        layout.addWidget(self.current_label)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Проект', 'Раздел', 'Состояние'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            self.table.horizontalHeader().setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda _index: self.open_selected())
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        open_button = QPushButton('Открыть выбранный')
        open_button.setObjectName('secondaryButton')
        open_button.clicked.connect(self.open_selected)
        folder_button = QPushButton('Показать в проводнике')
        folder_button.setObjectName('secondaryButton')
        folder_button.clicked.connect(self.open_folder)
        favorite_button = QPushButton('★ / ☆ Избранное')
        favorite_button.setObjectName('secondaryButton')
        favorite_button.clicked.connect(self.toggle_favorite)
        remove_button = QPushButton('Убрать из недавних')
        remove_button.setObjectName('dangerButton')
        remove_button.clicked.connect(self.remove_recent)
        buttons.addWidget(folder_button)
        buttons.addWidget(favorite_button)
        buttons.addWidget(remove_button)
        buttons.addWidget(open_button)
        layout.addLayout(buttons)
        self.root.addWidget(card, 1)
        self.context.favorites_changed.connect(self.refresh)
        self.context.recent_projects_changed.connect(self.refresh)
        self.refresh()

    def choose_project(self) -> None:
        start = str(self.context.project_root or Path.home())
        selected = QFileDialog.getExistingDirectory(self, 'Выберите проект ML Studio', start)
        if not selected:
            return
        if not self.context.set_project(selected):
            QMessageBox.warning(
                self,
                'Это не проект ML Studio',
                'В выбранной папке нет hpo_pointnet.py.',
            )
            return
        self.project_opened.emit(self.context.project_root)

    def open_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        path = Path(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))
        if self.context.set_project(path):
            self.project_opened.emit(path)
        else:
            QMessageBox.warning(self, 'Проект недоступен', str(path))

    def open_folder(self) -> None:
        path = self.selected_project() or self.context.project_root
        if path is not None and path.exists():
            open_in_file_manager(path)

    def selected_project(self) -> Path | None:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return Path(value) if value else None

    def toggle_favorite(self) -> None:
        path = self.selected_project()
        if path is not None:
            self.context.toggle_favorite(path)

    def remove_recent(self) -> None:
        path = self.selected_project()
        if path is not None:
            self.context.remove_recent(path)

    def refresh(self) -> None:
        current = self.context.project_root
        self.current_label.setText(
            f'Текущий проект: {current}' if current is not None else 'Проект не выбран'
        )
        favorites = list(self.context.favorite_projects)
        recent = [path for path in self.context.recent_projects if path not in favorites]
        if current is not None and current not in favorites and current not in recent:
            recent.insert(0, current)
        projects = [(path, 'Избранное') for path in favorites]
        projects.extend((path, 'Недавнее') for path in recent)
        self.table.setRowCount(len(projects))
        for row, (path, group) in enumerate(projects):
            path_item = QTableWidgetItem(path.name)
            path_item.setToolTip(str(path))
            path_item.setData(Qt.ItemDataRole.UserRole, str(path))
            state = QTableWidgetItem('Текущий' if path == current else 'Доступен')
            self.table.setItem(row, 0, path_item)
            self.table.setItem(row, 1, QTableWidgetItem(group))
            self.table.setItem(row, 2, state)


class DatasetsPage(StudioPage):
    def __init__(self, context: StudioContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.datasets: list[DatasetInfo] = []
        self.selected_dataset: DatasetInfo | None = None
        self.selected_scene: Path | None = None
        self._preview_thread: QThread | None = None
        self._preview_worker: PointCloudLoadWorker | None = None
        self._viewer_windows: set[PointCloudViewerDialog] = set()
        self._mini_thread: QThread | None = None
        self._mini_worker: MiniDatasetWorker | None = None
        self._mini_progress: QProgressDialog | None = None

        self.folder_button = QPushButton('Открыть папку')
        self.folder_button.setObjectName('secondaryButton')
        self.folder_button.clicked.connect(self.open_dataset_folder)
        self.mini_button = QPushButton('Создать мини-копию')
        self.mini_button.setObjectName('secondaryButton')
        self.mini_button.clicked.connect(self.create_mini_copy)
        self.launch_button = QPushButton('Запустить обучение')
        self.launch_button.setObjectName('primaryButton')
        self.launch_button.clicked.connect(self.open_launch)
        self.add_header(
            'Датасеты',
            'Подготовленные сцены .npy для настройки нейросети и запуска HPO.',
            [self.mini_button, self.folder_button, self.launch_button],
        )

        content = QHBoxLayout()
        content.setSpacing(12)
        datasets_card, datasets_layout = panel()
        datasets_card.setMinimumWidth(220)
        datasets_card.setMaximumWidth(285)
        datasets_layout.addWidget(title_label('Наборы данных'))
        self.dataset_list = QListWidget()
        self.dataset_list.setObjectName('datasetList')
        self.dataset_list.currentRowChanged.connect(self.select_dataset)
        datasets_layout.addWidget(self.dataset_list, 1)
        refresh = QPushButton('Перечитать')
        refresh.setObjectName('secondaryButton')
        refresh.clicked.connect(self.refresh)
        datasets_layout.addWidget(refresh)
        content.addWidget(datasets_card)

        scenes_card, scenes_layout = panel()
        scenes_card.setMinimumWidth(390)
        self.dataset_title = title_label('Датасет не выбран')
        self.dataset_path = QLabel('')
        self.dataset_path.setObjectName('mutedText')
        self.dataset_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        scenes_layout.addWidget(self.dataset_title)
        scenes_layout.addWidget(self.dataset_path)
        self.stats_layout = QHBoxLayout()
        self.stat_scenes = self._add_stat(self.stats_layout, 'Сцены')
        self.stat_points = self._add_stat(self.stats_layout, 'Точки')
        self.stat_size = self._add_stat(self.stats_layout, 'Размер')
        self.stat_columns = self._add_stat(self.stats_layout, 'Колонки')
        scenes_layout.addLayout(self.stats_layout)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Поиск по сценам…')
        self.search.textChanged.connect(self.filter_scenes)
        scenes_layout.addWidget(self.search)
        self.scene_table = QTableWidget(0, 3)
        self.scene_table.setHorizontalHeaderLabels(['Сцена', 'Точек', 'Размер'])
        self.scene_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            self.scene_table.horizontalHeader().setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        self.scene_table.verticalHeader().setVisible(False)
        self.scene_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.scene_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.scene_table.itemSelectionChanged.connect(self.scene_selected)
        scenes_layout.addWidget(self.scene_table, 1)
        content.addWidget(scenes_card, 1)

        preview_card, preview_layout = panel()
        preview_card.setMinimumWidth(430)
        preview_header = QHBoxLayout()
        preview_header.addWidget(title_label('3D-просмотр'))
        preview_header.addStretch(1)
        self.class_button = QPushButton('Классы')
        self.height_button = QPushButton('Высота')
        for button in (self.class_button, self.height_button):
            button.setCheckable(True)
            button.setObjectName('segmentButton')
        self.class_button.setChecked(True)
        self.class_button.clicked.connect(lambda: self.set_color_mode('classification'))
        self.height_button.clicked.connect(lambda: self.set_color_mode('height'))
        preview_header.addWidget(self.class_button)
        preview_header.addWidget(self.height_button)
        reset_view = QPushButton('Исходный вид')
        reset_view.setObjectName('secondaryButton')
        preview_header.addWidget(reset_view)
        preview_layout.addLayout(preview_header)
        self.preview_status = QLabel('Выберите сцену')
        self.preview_status.setObjectName('mutedText')
        preview_layout.addWidget(self.preview_status)
        self.canvas = PointCloudCanvas()
        self.canvas.setMinimumSize(360, 300)
        self.canvas.set_point_size(2.0)
        preview_layout.addWidget(self.canvas, 1)
        self.legend = QLabel('')
        self.legend.setObjectName('legendText')
        self.legend.setWordWrap(True)
        preview_layout.addWidget(self.legend)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel('Быстрая проверка моделью'))
        self.preview_model = QComboBox()
        self.preview_model.setToolTip('Лучшие checkpoint-файлы из готовых прогонов')
        model_row.addWidget(self.preview_model, 1)
        preview_layout.addLayout(model_row)
        preview_buttons = QHBoxLayout()
        full = QPushButton('Открыть в 3D')
        full.setObjectName('secondaryButton')
        full.clicked.connect(self.open_full_viewer)
        test = QPushButton('Тестировать модель')
        test.setObjectName('primaryButton')
        test.clicked.connect(self.open_test)
        preview_buttons.addWidget(full)
        preview_buttons.addWidget(test)
        preview_layout.addLayout(preview_buttons)
        content.addWidget(preview_card, 1)
        self.root.addLayout(content, 1)
        reset_view.clicked.connect(self.canvas.reset_camera)
        self.refresh()

    @staticmethod
    def _add_stat(layout: QHBoxLayout, title: str) -> QLabel:
        box = QVBoxLayout()
        name = QLabel(title.upper())
        name.setObjectName('statName')
        value = QLabel('—')
        value.setObjectName('statValue')
        box.addWidget(name)
        box.addWidget(value)
        layout.addLayout(box, 1)
        return value

    def refresh(self) -> None:
        current_row = self.dataset_list.currentRow()
        self.dataset_list.clear()
        self.datasets = (
            scan_datasets(self.context.project_root)
            if self.context.project_root is not None
            else []
        )
        self.preview_model.clear()
        if self.context.project_root is not None:
            for artifact in scan_artifacts(self.context.project_root):
                if artifact.category == 'Модели':
                    self.preview_model.addItem(
                        f'{artifact.run_name} · {artifact.path.name}',
                        str(artifact.path),
                    )
        for dataset in self.datasets:
            item = QListWidgetItem(dataset.path.name)
            item.setToolTip(
                f'{len(dataset.scenes)} сцен · {human_bytes(dataset.bytes_on_disk)} · '
                f'{human_count(dataset.points)} точек'
            )
            self.dataset_list.addItem(item)
        if self.datasets:
            self.dataset_list.setCurrentRow(min(max(current_row, 0), len(self.datasets) - 1))
        else:
            self.selected_dataset = None
            self.scene_table.setRowCount(0)
            self.dataset_title.setText('Датасеты не найдены')
            self.dataset_path.setText(
                'Ожидаются каталоги exports* со сценами scene_*.npy.'
                if self.context.project_root is not None
                else 'Сначала откройте проект.'
            )

    def select_dataset(self, row: int) -> None:
        if not 0 <= row < len(self.datasets):
            return
        self.selected_dataset = self.datasets[row]
        dataset = self.selected_dataset
        self.dataset_title.setText(dataset.path.name)
        self.dataset_path.setText(str(dataset.path))
        self.stat_scenes.setText(str(len(dataset.scenes)))
        self.stat_points.setText(human_count(dataset.points))
        self.stat_size.setText(human_bytes(dataset.bytes_on_disk))
        columns = sorted({scene.columns for scene in dataset.scenes})
        self.stat_columns.setText(', '.join(map(str, columns)))
        self.populate_scenes(dataset.scenes)
        if dataset.scenes:
            self.scene_table.selectRow(0)

    def populate_scenes(self, scenes) -> None:
        query = self.search.text().strip().casefold()
        visible = [scene for scene in scenes if query in scene.path.name.casefold()]
        self.scene_table.setRowCount(len(visible))
        for row, scene in enumerate(visible):
            name = QTableWidgetItem(scene.path.name)
            name.setData(Qt.ItemDataRole.UserRole, str(scene.path))
            self.scene_table.setItem(row, 0, name)
            self.scene_table.setItem(row, 1, QTableWidgetItem(f'{scene.points:,}'.replace(',', ' ')))
            self.scene_table.setItem(row, 2, QTableWidgetItem(human_bytes(scene.bytes_on_disk)))

    def filter_scenes(self) -> None:
        if self.selected_dataset is not None:
            self.populate_scenes(self.selected_dataset.scenes)

    def scene_selected(self) -> None:
        row = self.scene_table.currentRow()
        if row < 0 or self.scene_table.item(row, 0) is None:
            return
        self.selected_scene = Path(
            self.scene_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        )
        self.load_preview(self.selected_scene)

    def load_preview(self, path: Path) -> None:
        if self._preview_thread is not None:
            self._preview_thread.requestInterruption()
            return
        self.preview_status.setText(f'Загрузка {path.name}…')
        thread = QThread(self)
        worker = PointCloudLoadWorker(path, max_points=180_000)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self.preview_loaded)
        worker.failed.connect(self.preview_failed)
        worker.cancelled.connect(thread.quit)
        worker.loaded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.preview_finished)
        self._preview_thread = thread
        self._preview_worker = worker
        thread.start()

    @Slot(object)
    def preview_loaded(self, cloud: PointCloudData) -> None:
        self.canvas.set_cloud(cloud)
        self.preview_status.setText(
            f'{cloud.path.name} · {human_count(cloud.source_count)} точек'
        )
        parts = []
        for class_id, count in sorted(cloud.class_counts().items()):
            style = class_style_for(class_id)
            share = count / max(1, cloud.rendered_count) * 100
            parts.append(f'{style.title}: {share:.0f}%')
        self.legend.setText('  ·  '.join(parts))

    @Slot(str)
    def preview_failed(self, error: str) -> None:
        self.preview_status.setText(f'Не удалось построить превью: {error}')

    @Slot()
    def preview_finished(self) -> None:
        self._preview_thread = None
        self._preview_worker = None

    def set_color_mode(self, mode: str) -> None:
        self.class_button.setChecked(mode == 'classification')
        self.height_button.setChecked(mode == 'height')
        self.canvas.set_color_mode(mode)

    def open_dataset_folder(self) -> None:
        if self.selected_dataset is not None:
            open_in_file_manager(self.selected_dataset.path)

    def create_mini_copy(self) -> None:
        dataset = self.selected_dataset
        if dataset is None:
            QMessageBox.information(self, 'Датасет не выбран', 'Сначала выберите датасет.')
            return
        target = dataset.path.with_name(f'{dataset.path.name}_mini')
        if target.exists():
            QMessageBox.warning(
                self,
                'Мини-копия уже есть',
                f'Каталог не будет перезаписан:\n{target}',
            )
            return
        count, accepted = QInputDialog.getInt(
            self,
            'Мини-копия датасета',
            'Сколько сцен скопировать:',
            min(10, len(dataset.scenes)),
            1,
            len(dataset.scenes),
        )
        if not accepted:
            return

        progress = QProgressDialog('Подготовка…', 'Отмена', 0, count, self)
        progress.setWindowTitle('Создание мини-копии')
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        thread = QThread(self)
        worker = MiniDatasetWorker(dataset, count)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._mini_progress_changed)
        worker.completed.connect(self._mini_completed)
        worker.failed.connect(self._mini_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._mini_finished)
        progress.canceled.connect(thread.requestInterruption)
        self._mini_progress = progress
        self._mini_thread = thread
        self._mini_worker = worker
        self.mini_button.setEnabled(False)
        thread.start()

    @Slot(int, int, str)
    def _mini_progress_changed(self, done: int, total: int, name: str) -> None:
        if self._mini_progress is None:
            return
        self._mini_progress.setMaximum(total)
        self._mini_progress.setValue(done)
        self._mini_progress.setLabelText(f'Копируется {name}')

    @Slot(object)
    def _mini_completed(self, target: Path) -> None:
        if self._mini_progress is not None:
            self._mini_progress.setValue(self._mini_progress.maximum())
        self.refresh()
        for index, dataset in enumerate(self.datasets):
            if dataset.path == target:
                self.dataset_list.setCurrentRow(index)
                break
        QMessageBox.information(self, 'Мини-копия готова', str(target))

    @Slot(str)
    def _mini_failed(self, message: str) -> None:
        if self._mini_progress is not None:
            self._mini_progress.cancel()
        if 'отменено' not in message.casefold():
            QMessageBox.warning(self, 'Не удалось создать мини-копию', message)

    @Slot()
    def _mini_finished(self) -> None:
        self.mini_button.setEnabled(True)
        self._mini_progress = None
        self._mini_thread = None
        self._mini_worker = None

    def open_launch(self) -> None:
        if self.context.project_root is None:
            self.project_selection_requested.emit()
            return
        dialog = LaunchDialog(
            self.context,
            self.selected_dataset.path if self.selected_dataset is not None else None,
            self,
        )
        dialog.exec()

    def open_test(self) -> None:
        if self.context.project_root is None:
            self.project_selection_requested.emit()
            return
        checkpoint = self.preview_model.currentData()
        dialog = TestModelDialog(
            self.context,
            scene=self.selected_scene,
            checkpoint=Path(checkpoint) if checkpoint else None,
            parent=self,
        )
        dialog.exec()

    def open_full_viewer(self) -> None:
        if self.selected_scene is None:
            return
        viewer = PointCloudViewerDialog(self.selected_scene, self)
        viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        viewer.destroyed.connect(lambda: self._viewer_windows.discard(viewer))
        self._viewer_windows.add(viewer)
        viewer.showMaximized()


class RunsPage(StudioPage):
    def __init__(self, context: StudioContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.runs: list[RunInfo] = []
        refresh = QPushButton('Перечитать runs/')
        refresh.setObjectName('secondaryButton')
        refresh.clicked.connect(self.refresh)
        launch = QPushButton('Новый запуск')
        launch.setObjectName('primaryButton')
        launch.clicked.connect(self.open_launch)
        self.add_header(
            'Запуски',
            'Реальные best.json, trials.csv и study.db из выбранного проекта.',
            [refresh, launch],
        )
        content = QHBoxLayout()
        content.setSpacing(12)

        list_card, list_layout = panel()
        list_card.setMinimumWidth(270)
        list_card.setMaximumWidth(340)
        list_layout.addWidget(title_label('Прогоны'))
        self.run_list = QListWidget()
        self.run_list.setObjectName('runList')
        self.run_list.currentRowChanged.connect(self.select_run)
        list_layout.addWidget(self.run_list, 1)
        content.addWidget(list_card)

        detail_card, detail_layout = panel()
        header = QHBoxLayout()
        self.run_title = title_label('Прогон не выбран')
        self.run_status = QLabel('')
        self.run_status.setObjectName('statusChip')
        header.addWidget(self.run_title)
        header.addStretch(1)
        header.addWidget(self.run_status)
        detail_layout.addLayout(header)
        self.summary = QLabel('')
        self.summary.setObjectName('runSummary')
        self.summary.setWordWrap(True)
        detail_layout.addWidget(self.summary)

        self.trial_table = QTableWidget()
        self.trial_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.trial_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.trial_table.verticalHeader().setVisible(False)
        detail_layout.addWidget(self.trial_table, 1)
        actions = QHBoxLayout()
        self.folder_button = QPushButton('Открыть папку')
        self.folder_button.setObjectName('secondaryButton')
        self.folder_button.clicked.connect(self.open_folder)
        self.continue_button = QPushButton('Продолжить в форме')
        self.continue_button.setObjectName('primaryButton')
        self.continue_button.clicked.connect(self.continue_run)
        actions.addStretch(1)
        actions.addWidget(self.folder_button)
        actions.addWidget(self.continue_button)
        detail_layout.addLayout(actions)
        content.addWidget(detail_card, 1)

        params_card, params_layout = panel()
        params_card.setMinimumWidth(300)
        params_card.setMaximumWidth(400)
        params_layout.addWidget(title_label('Лучшие параметры'))
        self.best_preview = QPlainTextEdit()
        self.best_preview.setObjectName('jsonPreview')
        self.best_preview.setReadOnly(True)
        params_layout.addWidget(self.best_preview, 1)
        content.addWidget(params_card)
        self.root.addLayout(content, 1)
        self.refresh()

    def refresh(self) -> None:
        current_name = None
        if 0 <= self.run_list.currentRow() < len(self.runs):
            current_name = self.runs[self.run_list.currentRow()].name
        self.runs = (
            scan_runs(self.context.project_root)
            if self.context.project_root is not None
            else []
        )
        self.run_list.clear()
        selected = 0
        for index, run in enumerate(self.runs):
            score = f' · F1 {run.best_f1:.4f}' if run.best_f1 is not None else ''
            item = QListWidgetItem(f'{run.name}\n{run.status}{score}')
            item.setToolTip(str(run.path))
            self.run_list.addItem(item)
            if run.name == current_name:
                selected = index
        if self.runs:
            self.run_list.setCurrentRow(selected)
        else:
            self.run_title.setText('Прогоны не найдены')
            self.run_status.clear()
            self.summary.setText('Каталог runs/ пуст или ещё не создан.')
            self.trial_table.setRowCount(0)
            self.best_preview.clear()

    def select_run(self, row: int) -> None:
        if not 0 <= row < len(self.runs):
            return
        run = self.runs[row]
        self.run_title.setText(run.name)
        self.run_status.setText(run.status.upper())
        self.run_status.setProperty('status', run.status.casefold())
        self.run_status.style().unpolish(self.run_status)
        self.run_status.style().polish(self.run_status)
        f1 = f'{run.best_f1:.4f}' if run.best_f1 is not None else '—'
        baseline = f'{run.baseline_f1:.4f}' if run.baseline_f1 is not None else '—'
        margin = (
            f'{run.best_f1 - run.baseline_f1:+.4f}'
            if run.best_f1 is not None and run.baseline_f1 is not None
            else '—'
        )
        self.summary.setText(
            f'Класс: {run.target or "—"}   ·   Признаки: {run.features or "—"}   ·   '
            f'Лучший F1: {f1}   ·   Baseline: {baseline}   ·   Отрыв: {margin}'
        )
        self.best_preview.setPlainText(
            json.dumps(run.best, ensure_ascii=False, indent=2) if run.best else 'best.json пока нет'
        )
        rows = run.trials
        preferred = ('number', 'state', 'f1', 'precision', 'recall', 'seconds')
        columns = [key for key in preferred if any(key in item for item in rows)]
        if not columns and rows:
            columns = list(rows[0])[:8]
        self.trial_table.setColumnCount(len(columns))
        self.trial_table.setHorizontalHeaderLabels(columns)
        self.trial_table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column_index, key in enumerate(columns):
                self.trial_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(values.get(key, '')),
                )
        if columns:
            self.trial_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )

    def selected_run(self) -> RunInfo | None:
        row = self.run_list.currentRow()
        return self.runs[row] if 0 <= row < len(self.runs) else None

    def open_folder(self) -> None:
        run = self.selected_run()
        if run is not None:
            open_in_file_manager(run.path)

    def open_launch(self) -> None:
        if self.context.project_root is None:
            self.project_selection_requested.emit()
            return
        LaunchDialog(self.context, parent=self).exec()

    def continue_run(self) -> None:
        run = self.selected_run()
        if run is None or self.context.project_root is None:
            return
        dialog = LaunchDialog(self.context, parent=self)
        dialog.run_name.setText(run.name)
        if run.target in {dialog.target.itemText(i) for i in range(dialog.target.count())}:
            dialog.target.setCurrentText(run.target)
        if run.features:
            dialog.features.setCurrentText(run.features)
        launch = run.best.get('launch') if isinstance(run.best, dict) else None
        if isinstance(launch, dict):
            for key, widget in (
                ('seed', dialog.seed),
                ('epochs', dialog.epochs),
                ('patch_points', dialog.patch_points),
                ('patches_per_scene', dialog.patches_per_scene),
                ('workers', dialog.workers),
            ):
                value = launch.get(key)
                if isinstance(value, int):
                    widget.setValue(value)
        dialog._update_command()
        dialog.exec()


class LivePage(StudioPage):
    def __init__(self, context: StudioContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.add_header(
            'Live',
            'Живой stdout/stderr всех процессов, запущенных из этого окна.',
        )
        content = QHBoxLayout()
        content.setSpacing(12)
        process_card, process_layout = panel()
        process_card.setMinimumWidth(300)
        process_card.setMaximumWidth(390)
        process_layout.addWidget(title_label('Процессы'))
        self.process_list = QListWidget()
        self.process_list.currentRowChanged.connect(self.select_process)
        process_layout.addWidget(self.process_list, 1)
        process_actions = QHBoxLayout()
        self.restart_button = QPushButton('Перезапустить')
        self.restart_button.setObjectName('secondaryButton')
        self.restart_button.clicked.connect(self.restart_process)
        self.stop_button = QPushButton('Остановить')
        self.stop_button.setObjectName('dangerButton')
        self.stop_button.clicked.connect(self.stop_process)
        process_actions.addWidget(self.restart_button)
        process_actions.addWidget(self.stop_button)
        process_layout.addLayout(process_actions)
        content.addWidget(process_card)

        log_card, log_layout = panel()
        log_header = QHBoxLayout()
        self.process_title = title_label('Процесс не выбран')
        self.process_summary = QLabel('')
        self.process_summary.setObjectName('mutedText')
        copy_button = QPushButton('Копировать лог')
        copy_button.setObjectName('secondaryButton')
        copy_button.clicked.connect(self.copy_log)
        log_header.addWidget(self.process_title)
        log_header.addStretch(1)
        log_header.addWidget(copy_button)
        log_layout.addLayout(log_header)
        log_layout.addWidget(self.process_summary)
        self.log = QPlainTextEdit()
        self.log.setObjectName('liveLog')
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.log, 1)
        content.addWidget(log_card, 1)
        self.root.addLayout(content, 1)

        self.context.processes.changed.connect(self.refresh)
        self.context.processes.process_added.connect(self._connect_process)
        for process in self.context.processes.processes:
            self._connect_process(process)
        self.refresh()

    def _connect_process(self, process: ManagedProcess) -> None:
        process.output_received.connect(
            lambda text, target=process: self.append_output(target, text)
        )

    def refresh(self) -> None:
        current = self.process_list.currentRow()
        self.process_list.blockSignals(True)
        self.process_list.clear()
        for process in self.context.processes.processes:
            snapshot = process.snapshot()
            f1 = f' · F1 {snapshot.last_f1:.4f}' if snapshot.last_f1 is not None else ''
            minutes, seconds = divmod(snapshot.seconds, 60)
            self.process_list.addItem(
                f'{snapshot.name}\n{snapshot.status} · {minutes:02d}:{seconds:02d}{f1}'
            )
        self.process_list.blockSignals(False)
        if self.context.processes.processes:
            self.process_list.setCurrentRow(
                min(max(current, 0), len(self.context.processes.processes) - 1)
            )
            self.select_process(self.process_list.currentRow())
        else:
            self.process_title.setText('Активных процессов нет')
            self.process_summary.setText('Запустите обучение или тестирование из соответствующего раздела.')
            self.log.clear()

    def selected_process(self) -> ManagedProcess | None:
        row = self.process_list.currentRow()
        processes = self.context.processes.processes
        return processes[row] if 0 <= row < len(processes) else None

    def select_process(self, _row: int) -> None:
        process = self.selected_process()
        if process is None:
            return
        snapshot = process.snapshot()
        self.process_title.setText(snapshot.name)
        self.process_summary.setText(
            f'{snapshot.status} · {snapshot.seconds} с · {" ".join(snapshot.command)}'
        )
        self.log.setPlainText(''.join(process.log))
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def append_output(self, process: ManagedProcess, text: str) -> None:
        if process is self.selected_process():
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(text)
            self.log.moveCursor(QTextCursor.MoveOperation.End)

    def stop_process(self) -> None:
        process = self.selected_process()
        if process is not None:
            process.stop()

    def restart_process(self) -> None:
        process = self.selected_process()
        if process is not None:
            process.restart()

    def copy_log(self) -> None:
        process = self.selected_process()
        if process is not None:
            QApplication.clipboard().setText(''.join(process.log))


class TestingPage(StudioPage):
    def __init__(self, context: StudioContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.models: list[ArtifactInfo] = []
        self.scenes: list[Path] = []
        self._viewer_windows: set[PointCloudViewerDialog] = set()
        refresh = QPushButton('Перечитать')
        refresh.setObjectName('secondaryButton')
        refresh.clicked.connect(self.refresh)
        run = QPushButton('Запустить тест')
        run.setObjectName('primaryButton')
        run.clicked.connect(self.run_test)
        self.add_header(
            'Тестирование',
            'predict_scene.py запускается на выбранной модели и реальной сцене.',
            [refresh, run],
        )
        content = QHBoxLayout()
        content.setSpacing(12)
        selection, selection_layout = panel()
        selection.setMinimumWidth(430)
        selection_layout.addWidget(title_label('Модель и сцена'))
        selection_layout.addWidget(QLabel('Модель'))
        self.model_combo = QComboBox()
        selection_layout.addWidget(self.model_combo)
        selection_layout.addWidget(QLabel('Сцена'))
        self.scene_combo = QComboBox()
        selection_layout.addWidget(self.scene_combo)
        open_scene = QPushButton('Открыть сцену в 3D')
        open_scene.setObjectName('secondaryButton')
        open_scene.clicked.connect(self.open_scene)
        selection_layout.addWidget(open_scene)
        note = QLabel(
            'Результаты сохраняются в tests/. Ход выполнения и stdout доступны в Live.'
        )
        note.setObjectName('infoNote')
        note.setWordWrap(True)
        selection_layout.addWidget(note)
        selection_layout.addStretch(1)
        content.addWidget(selection)

        results, results_layout = panel()
        results_layout.addWidget(title_label('Последние результаты'))
        self.result_list = QListWidget()
        self.result_list.currentItemChanged.connect(self.show_result)
        results_layout.addWidget(self.result_list, 1)
        content.addWidget(results, 1)

        preview, preview_layout = panel()
        preview.setMinimumWidth(360)
        preview_layout.addWidget(title_label('Метрики JSON'))
        self.result_preview = QPlainTextEdit()
        self.result_preview.setObjectName('jsonPreview')
        self.result_preview.setReadOnly(True)
        preview_layout.addWidget(self.result_preview, 1)
        content.addWidget(preview, 1)
        self.root.addLayout(content, 1)
        self.refresh()

    def refresh(self) -> None:
        root = self.context.project_root
        self.models = []
        self.scenes = []
        self.model_combo.clear()
        self.scene_combo.clear()
        self.result_list.clear()
        if root is None:
            self.result_preview.setPlainText('Сначала откройте проект.')
            return
        self.models = [item for item in scan_artifacts(root) if item.category == 'Модели']
        for item in self.models:
            self.model_combo.addItem(f'{item.run_name} · {item.path.name}', str(item.path))
        for dataset in scan_datasets(root):
            for scene in dataset.scenes:
                self.scenes.append(scene.path)
                self.scene_combo.addItem(
                    f'{dataset.path.name} / {scene.path.name}',
                    str(scene.path),
                )
        tests = root / 'tests'
        if tests.is_dir():
            for path in sorted(tests.rglob('*.json'), reverse=True):
                item = QListWidgetItem(str(path.relative_to(tests)))
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.result_list.addItem(item)
        if self.result_list.count():
            self.result_list.setCurrentRow(0)
        elif not self.models:
            self.result_preview.setPlainText(
                'Модели .pt не найдены. Они появятся после завершённого запуска.'
            )

    def run_test(self) -> None:
        if self.context.project_root is None:
            self.project_selection_requested.emit()
            return
        checkpoint = (
            Path(self.model_combo.currentData()) if self.model_combo.currentData() else None
        )
        scene = Path(self.scene_combo.currentData()) if self.scene_combo.currentData() else None
        TestModelDialog(
            self.context,
            scene=scene,
            checkpoint=checkpoint,
            parent=self,
        ).exec()

    def open_scene(self) -> None:
        value = self.scene_combo.currentData()
        if not value:
            return
        viewer = PointCloudViewerDialog(Path(value), self)
        viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        viewer.destroyed.connect(lambda: self._viewer_windows.discard(viewer))
        self._viewer_windows.add(viewer)
        viewer.showMaximized()

    def show_result(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is None:
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        self.result_preview.setPlainText(read_text_preview(path))


class ScenarioRunner(QWidget):
    changed = Signal(str)
    step_changed = Signal(int, str)
    finished = Signal(bool, str)

    def __init__(
        self,
        context: StudioContext,
        name: str,
        steps: list[dict],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.context = context
        self.name = name
        self.steps = steps
        self.index = 0
        self.current: ManagedProcess | None = None

    def start(self) -> None:
        self.index = 0
        self._start_next()

    def _start_next(self) -> None:
        root = self.context.project_root
        if root is None:
            self.finished.emit(False, 'Проект не выбран')
            return
        if self.index >= len(self.steps):
            self.finished.emit(True, 'Сценарий завершён')
            return
        step = self.steps[self.index]
        arguments = step.get('args')
        if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
            self.finished.emit(False, f'Шаг {self.index + 1}: args должен быть списком строк')
            return
        fixed = step.get('fixed_params')
        is_single_hpo = bool(arguments) and Path(arguments[0]).name == 'hpo_pointnet.py'
        if is_single_hpo and isinstance(fixed, dict) and fixed:
            runtime = root / 'scenarios' / '.runtime'
            runtime.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', self.name)
            params_path = runtime / f'{safe}_{self.index + 1}.params.json'
            params_path.write_text(
                json.dumps(fixed, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            if '--fixed-params' not in arguments:
                arguments = [*arguments, '--fixed-params', str(params_path)]
        title = str(step.get('title') or f'Шаг {self.index + 1}')
        self.changed.emit(f'{title} ({self.index + 1}/{len(self.steps)})')
        self.step_changed.emit(self.index, 'ИДЁТ')
        python = self.context.python_executable()
        if python is None:
            self.finished.emit(
                False,
                'ML-окружение не готово: запустите run_all.cmd, '
                'чтобы создать .venv311',
            )
            return
        process = self.context.processes.create(
            f'{self.name}: {title}',
            str(python),
            arguments,
            root,
        )
        self.current = process
        process.finished.connect(self._step_finished)
        process.start()

    def _step_finished(self, code: int) -> None:
        if code != 0:
            self.step_changed.emit(self.index, 'ОШИБКА')
            self.finished.emit(False, f'Шаг {self.index + 1} завершился с кодом {code}')
            return
        self.step_changed.emit(self.index, 'ГОТОВО')
        self.index += 1
        self._start_next()

    def stop(self) -> None:
        if self.current is not None:
            self.current.stop()


class ScenariosPage(StudioPage):
    def __init__(self, context: StudioContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.paths: list[Path] = []
        self.steps: list[dict] = []
        self._loading_step = False
        self.runner: ScenarioRunner | None = None
        new_button = QPushButton('Новый сценарий')
        new_button.setObjectName('secondaryButton')
        new_button.clicked.connect(self.new_scenario)
        save_button = QPushButton('Сохранить')
        save_button.setObjectName('secondaryButton')
        save_button.clicked.connect(self.save_scenario)
        run_button = QPushButton('Запустить сценарий')
        run_button.setObjectName('primaryButton')
        run_button.clicked.connect(self.run_scenario)
        self.add_header(
            'Сценарии',
            'Последовательности реальных Python-команд из scenarios/*.json.',
            [new_button, save_button, run_button],
        )
        content = QHBoxLayout()
        content.setSpacing(12)
        list_card, list_layout = panel()
        list_card.setMinimumWidth(260)
        list_card.setMaximumWidth(340)
        list_layout.addWidget(title_label('Сценарии проекта'))
        self.scenario_list = QListWidget()
        self.scenario_list.currentRowChanged.connect(self.select_scenario)
        list_layout.addWidget(self.scenario_list, 1)
        content.addWidget(list_card)
        editor_card, editor_layout = panel()
        self.scenario_title = QLineEdit()
        self.scenario_title.setPlaceholderText('Название сценария')
        self.scenario_title.editingFinished.connect(self.rename_scenario)
        editor_layout.addWidget(self.scenario_title)

        step_content = QHBoxLayout()
        step_content.setSpacing(12)
        step_column = QVBoxLayout()
        step_header = QHBoxLayout()
        step_header.addWidget(title_label('Шаги'))
        step_header.addStretch(1)
        add_step = QPushButton('+ Шаг')
        add_step.setObjectName('secondaryButton')
        add_step.clicked.connect(self.add_step)
        remove_step = QPushButton('Убрать')
        remove_step.setObjectName('dangerButton')
        remove_step.clicked.connect(self.remove_step)
        step_header.addWidget(add_step)
        step_header.addWidget(remove_step)
        step_column.addLayout(step_header)
        self.step_list = QListWidget()
        self.step_list.setMinimumWidth(230)
        self.step_list.currentRowChanged.connect(self.load_step)
        step_column.addWidget(self.step_list, 1)
        step_content.addLayout(step_column, 1)

        form_column = QVBoxLayout()
        form_column.addWidget(title_label('Параметры шага'))
        form = QFormLayout()
        form.setSpacing(8)
        self.step_name = QLineEdit()
        self.step_script = QComboBox()
        self.step_script.addItem('Подбор одного класса', 'hpo_pointnet.py')
        self.step_script.addItem('Сценарий задач 2–7/final', 'run_all.py')
        self.step_target = QComboBox()
        self.step_target.addItems(('ground', 'vegetation', 'noise', 'building', 'water'))
        self.step_features = QComboBox()
        self.step_features.addItems(('xyz', 'xyz_hag'))
        self.step_trials = QSpinBox()
        self.step_trials.setRange(1, 10_000)
        self.step_trials.setValue(25)
        self.step_epochs = QSpinBox()
        self.step_epochs.setRange(1, 10_000)
        self.step_epochs.setValue(10)
        self.step_seed = QSpinBox()
        self.step_seed.setRange(0, 1_000_000)
        self.step_seed.setValue(42)
        self.step_extra = QLineEdit()
        self.step_extra.setPlaceholderText('--data-dir exports --output-dir runs/name')
        self.step_fixed = QPlainTextEdit()
        self.step_fixed.setObjectName('jsonPreview')
        self.step_fixed.setPlaceholderText('{}')
        self.step_fixed.setMaximumHeight(110)
        form.addRow('Имя', self.step_name)
        form.addRow('Скрипт', self.step_script)
        form.addRow('Класс', self.step_target)
        form.addRow('Признаки', self.step_features)
        form.addRow('Trials', self.step_trials)
        form.addRow('Эпохи', self.step_epochs)
        form.addRow('Seed', self.step_seed)
        form.addRow('Доп. аргументы', self.step_extra)
        form.addRow('Fixed params (JSON)', self.step_fixed)
        form_column.addLayout(form)
        form_column.addStretch(1)
        step_content.addLayout(form_column, 2)
        editor_layout.addLayout(step_content, 1)
        self.status = QLabel('')
        self.status.setObjectName('infoNote')
        editor_layout.addWidget(self.status)
        content.addWidget(editor_card, 1)
        self.root.addLayout(content, 1)
        for widget in (
            self.step_name,
            self.step_script,
            self.step_target,
            self.step_features,
            self.step_trials,
            self.step_epochs,
            self.step_seed,
            self.step_extra,
            self.step_fixed,
        ):
            signal = getattr(widget, 'textChanged', None)
            if signal is None:
                signal = getattr(widget, 'currentIndexChanged', None)
            if signal is None:
                signal = getattr(widget, 'valueChanged', None)
            if signal is not None:
                signal.connect(self.update_current_step)
        self.refresh()

    def refresh(self) -> None:
        root = self.context.project_root
        self.paths = []
        self.scenario_list.clear()
        if root is None:
            self.status.setText('Сначала откройте проект.')
            return
        directory = root / 'scenarios'
        if directory.is_dir():
            self.paths = sorted(directory.glob('*.json'))
        for path in self.paths:
            self.scenario_list.addItem(path.stem)
        if self.paths:
            self.scenario_list.setCurrentRow(0)
        else:
            self.scenario_title.clear()
            self.steps = []
            self.populate_step_list()
            self.status.setText('Сценариев пока нет. Создайте первый и добавьте шаги.')

    def select_scenario(self, row: int) -> None:
        if not 0 <= row < len(self.paths):
            return
        path = self.paths[row]
        self.scenario_title.setText(path.stem)
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            steps = value.get('steps', []) if isinstance(value, dict) else []
            if not isinstance(steps, list):
                raise TypeError('поле steps должно быть списком')
            self.steps = [step for step in steps if isinstance(step, dict)]
            self.status.clear()
        except (OSError, TypeError, ValueError) as error:
            self.steps = []
            self.status.setText(f'Сценарий не читается: {error}')
        self.populate_step_list()

    def current_path(self) -> Path | None:
        row = self.scenario_list.currentRow()
        return self.paths[row] if 0 <= row < len(self.paths) else None

    def new_scenario(self) -> None:
        root = self.context.project_root
        if root is None:
            self.project_selection_requested.emit()
            return
        directory = root / 'scenarios'
        directory.mkdir(parents=True, exist_ok=True)
        index = 1
        while (directory / f'сценарий {index}.json').exists():
            index += 1
        path = directory / f'сценарий {index}.json'
        path.write_text(
            json.dumps({'steps': []}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        self.refresh()
        self.scenario_list.setCurrentRow(self.paths.index(path))

    def rename_scenario(self) -> None:
        path = self.current_path()
        if path is None:
            return
        name = self.scenario_title.text().strip()
        if not name or any(symbol in name for symbol in '<>:"/\\|?*'):
            self.scenario_title.setText(path.stem)
            QMessageBox.warning(self, 'Некорректное имя', 'Измените название сценария.')
            return
        target = path.with_name(f'{name}.json')
        if target == path:
            return
        if target.exists():
            self.scenario_title.setText(path.stem)
            QMessageBox.warning(self, 'Сценарий уже есть', str(target))
            return
        path.rename(target)
        self.refresh()
        self.scenario_list.setCurrentRow(self.paths.index(target))

    def add_step(self) -> None:
        self.steps.append(
            {
                'title': f'Шаг {len(self.steps) + 1}',
                'args': [
                    'hpo_pointnet.py',
                    '--target', 'ground',
                    '--features', 'xyz',
                    '--n-trials', '25',
                    '--epochs', '10',
                    '--seed', '42',
                ],
                'fixed_params': {},
            }
        )
        self.populate_step_list(len(self.steps) - 1)

    def remove_step(self) -> None:
        row = self.step_list.currentRow()
        if not 0 <= row < len(self.steps):
            return
        self.steps.pop(row)
        self.populate_step_list(min(row, len(self.steps) - 1))

    def populate_step_list(self, current: int = 0) -> None:
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for index, step in enumerate(self.steps, 1):
            title = str(step.get('title') or f'Шаг {index}')
            self.step_list.addItem(f'{index}. {title}  ·  ЖДЁТ')
        self.step_list.blockSignals(False)
        if self.steps:
            self.step_list.setCurrentRow(max(0, min(current, len(self.steps) - 1)))
            self.load_step(self.step_list.currentRow())
        else:
            self._set_step_form_enabled(False)

    @staticmethod
    def _option(arguments: list[str], name: str, default: str) -> str:
        try:
            return arguments[arguments.index(name) + 1]
        except (ValueError, IndexError):
            return default

    def load_step(self, row: int) -> None:
        if not 0 <= row < len(self.steps):
            self._set_step_form_enabled(False)
            return
        self._set_step_form_enabled(True)
        step = self.steps[row]
        arguments = [str(value) for value in step.get('args', [])]
        script = arguments[0] if arguments else 'hpo_pointnet.py'
        known = {
            '--target', '--features', '--n-trials', '--epochs', '--seed',
        }
        extras: list[str] = []
        index = 1
        while index < len(arguments):
            current = arguments[index]
            if current in known and index + 1 < len(arguments):
                index += 2
                continue
            extras.append(current)
            index += 1
        self._loading_step = True
        self.step_name.setText(str(step.get('title') or f'Шаг {row + 1}'))
        script_index = self.step_script.findData(script)
        self.step_script.setCurrentIndex(max(0, script_index))
        self.step_target.setCurrentText(self._option(arguments, '--target', 'ground'))
        self.step_features.setCurrentText(self._option(arguments, '--features', 'xyz'))
        def integer_option(name: str, default: int) -> int:
            try:
                return int(self._option(arguments, name, str(default)))
            except ValueError:
                return default

        self.step_trials.setValue(integer_option('--n-trials', 25))
        self.step_epochs.setValue(integer_option('--epochs', 10))
        self.step_seed.setValue(integer_option('--seed', 42))
        self.step_extra.setText(shlex.join(extras))
        fixed = step.get('fixed_params')
        self.step_fixed.setPlainText(
            json.dumps(fixed if isinstance(fixed, dict) else {}, ensure_ascii=False, indent=2)
        )
        self._loading_step = False

    def _set_step_form_enabled(self, enabled: bool) -> None:
        for widget in (
            self.step_name,
            self.step_script,
            self.step_target,
            self.step_features,
            self.step_trials,
            self.step_epochs,
            self.step_seed,
            self.step_extra,
            self.step_fixed,
        ):
            widget.setEnabled(enabled)

    def update_current_step(self, *_args) -> bool:
        if self._loading_step:
            return True
        row = self.step_list.currentRow()
        if not 0 <= row < len(self.steps):
            return False
        try:
            extras = shlex.split(self.step_extra.text(), posix=os.name != 'nt')
            fixed = json.loads(self.step_fixed.toPlainText() or '{}')
            if not isinstance(fixed, dict):
                raise TypeError('fixed_params должен быть JSON-объектом')
        except (TypeError, ValueError) as error:
            self.status.setText(f'Ошибка в шаге: {error}')
            return False
        arguments = [str(self.step_script.currentData())]
        if self.step_script.currentData() == 'hpo_pointnet.py':
            arguments.extend(
                ['--target', self.step_target.currentText(), '--features', self.step_features.currentText()]
            )
        arguments.extend(
            [
                '--n-trials', str(self.step_trials.value()),
                '--epochs', str(self.step_epochs.value()),
                '--seed', str(self.step_seed.value()),
                *extras,
            ]
        )
        if self.step_script.currentData() != 'hpo_pointnet.py':
            fixed = {}
        self.steps[row] = {
            'title': self.step_name.text().strip() or f'Шаг {row + 1}',
            'args': arguments,
            'fixed_params': fixed,
        }
        self.status.clear()
        item = self.step_list.item(row)
        item.setText(f'{row + 1}. {self.steps[row]["title"]}  ·  ЖДЁТ')
        return True

    def save_scenario(self) -> None:
        path = self.current_path()
        if path is None:
            return
        try:
            if self.steps and not self.update_current_step():
                raise ValueError('Исправьте ошибку в текущем шаге.')
            path.write_text(
                json.dumps({'steps': self.steps}, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.warning(self, 'Не удалось сохранить сценарий', str(error))
            return
        self.status.setText('Сценарий сохранён.')

    def run_scenario(self) -> None:
        path = self.current_path()
        if path is None:
            return
        if not self.update_current_step():
            QMessageBox.warning(
                self,
                'Некорректный шаг',
                self.status.text() or 'Исправьте параметры шага.',
            )
            return
        if not self.steps:
            QMessageBox.warning(self, 'Пустой сценарий', 'Добавьте хотя бы один шаг.')
            return
        self.save_scenario()
        self.populate_step_list()
        self.runner = ScenarioRunner(self.context, path.stem, self.steps, self)
        self.runner.changed.connect(self.status.setText)
        self.runner.step_changed.connect(self.set_step_status)
        self.runner.finished.connect(self.scenario_finished)
        self.runner.start()

    @Slot(int, str)
    def set_step_status(self, index: int, status: str) -> None:
        item = self.step_list.item(index)
        if item is None:
            return
        title = str(self.steps[index].get('title') or f'Шаг {index + 1}')
        item.setText(f'{index + 1}. {title}  ·  {status}')

    def scenario_finished(self, success: bool, message: str) -> None:
        self.status.setText(message)
        if not success:
            QMessageBox.warning(self, 'Сценарий остановлен', message)


class ArtifactsPage(StudioPage):
    def __init__(self, context: StudioContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.artifacts: list[ArtifactInfo] = []
        refresh = QPushButton('Перечитать')
        refresh.setObjectName('secondaryButton')
        refresh.clicked.connect(self.refresh)
        self.add_header(
            'Артефакты',
            'Документы, модели и логи, которые действительно находятся на диске.',
            [refresh],
        )
        content = QHBoxLayout()
        content.setSpacing(12)
        left, left_layout = panel()
        left.setMinimumWidth(380)
        left_layout.addWidget(title_label('Файлы'))
        self.category = QComboBox()
        self.category.addItems(('Все', 'Документы', 'Модели', 'Логи'))
        self.category.currentTextChanged.connect(self.filter_artifacts)
        left_layout.addWidget(self.category)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.select_artifact)
        left_layout.addWidget(self.list, 1)
        actions = QHBoxLayout()
        folder = QPushButton('Открыть папку')
        folder.setObjectName('secondaryButton')
        folder.clicked.connect(self.open_folder)
        self.test_button = QPushButton('Тестировать модель')
        self.test_button.setObjectName('primaryButton')
        self.test_button.clicked.connect(self.test_model)
        actions.addWidget(folder)
        actions.addWidget(self.test_button)
        left_layout.addLayout(actions)
        content.addWidget(left)

        right, right_layout = panel()
        self.preview_title = title_label('Предпросмотр')
        right_layout.addWidget(self.preview_title)
        self.preview = QPlainTextEdit()
        self.preview.setObjectName('artifactPreview')
        self.preview.setReadOnly(True)
        right_layout.addWidget(self.preview, 1)
        content.addWidget(right, 1)
        self.root.addLayout(content, 1)
        self.refresh()

    def refresh(self) -> None:
        self.artifacts = (
            scan_artifacts(self.context.project_root)
            if self.context.project_root is not None
            else []
        )
        self.filter_artifacts()

    def visible_artifacts(self) -> list[ArtifactInfo]:
        selected = self.category.currentText()
        return [
            item for item in self.artifacts if selected == 'Все' or item.category == selected
        ]

    def filter_artifacts(self) -> None:
        self.list.clear()
        for artifact in self.visible_artifacts():
            prefix = f'{artifact.run_name} · ' if artifact.run_name else ''
            self.list.addItem(f'{prefix}{artifact.path.name}')
        if self.list.count():
            self.list.setCurrentRow(0)
        else:
            self.preview_title.setText('Артефакты не найдены')
            self.preview.setPlainText('Файлы появятся после запуска обучения или тестирования.')
            self.test_button.setEnabled(False)

    def selected_artifact(self) -> ArtifactInfo | None:
        row = self.list.currentRow()
        values = self.visible_artifacts()
        return values[row] if 0 <= row < len(values) else None

    def select_artifact(self, _row: int) -> None:
        artifact = self.selected_artifact()
        if artifact is None:
            return
        self.preview_title.setText(artifact.path.name)
        self.test_button.setEnabled(artifact.category == 'Модели')
        if artifact.category in {'Документы', 'Логи'}:
            self.preview.setPlainText(read_text_preview(artifact.path))
        else:
            self.preview.setPlainText(
                f'Модель PyTorch\n\nПуть: {artifact.path}\n'
                f'Размер: {human_bytes(artifact.path.stat().st_size)}\n'
                f'Прогон: {artifact.run_name or "—"}'
            )

    def open_folder(self) -> None:
        artifact = self.selected_artifact()
        if artifact is not None:
            open_in_file_manager(artifact.path.parent)

    def test_model(self) -> None:
        artifact = self.selected_artifact()
        if artifact is not None and artifact.category == 'Модели':
            TestModelDialog(self.context, checkpoint=artifact.path, parent=self).exec()


class EnvironmentPage(StudioPage):
    def __init__(self, context: StudioContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self.installer = EnvironmentInstaller(context, self)
        self.installer.finished.connect(self.installation_finished)
        self._check_thread: QThread | None = None
        self._check_worker: EnvironmentCheckWorker | None = None
        self._check_root: Path | None = None
        self._check_queued = False
        self._last_status: EnvironmentStatus | None = None
        self._last_status_root: Path | None = None

        refresh = QPushButton('Проверить снова')
        refresh.setObjectName('secondaryButton')
        refresh.clicked.connect(self.refresh)
        folder = QPushButton('Открыть проект')
        folder.setObjectName('secondaryButton')
        folder.clicked.connect(self.open_project)
        self.install_button = QPushButton('Установить окружение')
        self.install_button.setObjectName('primaryButton')
        self.install_button.clicked.connect(self.install_environment)
        self.add_header(
            'Окружение',
            'Python 3.11, PyTorch, CUDA и ключевые зависимости текущего проекта.',
            [folder, refresh, self.install_button],
        )

        content = QHBoxLayout()
        content.setSpacing(12)
        status_card, status_layout = panel()
        status_layout.addWidget(title_label('Проверка окружения'))
        self.python_value = self._status_row(status_layout, 'Python')
        self.venv_value = self._status_row(status_layout, '.venv311')
        self.torch_value = self._status_row(status_layout, 'PyTorch')
        self.cuda_value = self._status_row(status_layout, 'CUDA')
        self.numpy_value = self._status_row(status_layout, 'NumPy')
        self.optuna_value = self._status_row(status_layout, 'Optuna')
        self.pointcloud_value = self._status_row(status_layout, 'torch-pointcloud')
        status_layout.addStretch(1)
        content.addWidget(status_card, 1)

        command_card, command_layout = panel()
        command_layout.addWidget(title_label('Автоматическая настройка'))
        note = QLabel(
            'Кнопка установки создаёт отдельную .venv311. '
            'Полный живой лог появится в разделе Live.'
        )
        note.setObjectName('infoNote')
        note.setWordWrap(True)
        command_layout.addWidget(note)
        self.commands = QPlainTextEdit()
        self.commands.setObjectName('commandPreview')
        self.commands.setReadOnly(True)
        command_layout.addWidget(self.commands, 1)
        copy = QPushButton('Копировать команды')
        copy.setObjectName('secondaryButton')
        copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self.commands.toPlainText())
        )
        command_layout.addWidget(copy)
        content.addWidget(command_card, 1)
        self.root.addLayout(content, 1)
        self.refresh()

    @staticmethod
    def _status_row(layout: QVBoxLayout, title: str) -> QLabel:
        row = QHBoxLayout()
        name = QLabel(title)
        name.setObjectName('environmentName')
        value = QLabel('—')
        value.setObjectName('environmentValue')
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(value)
        layout.addLayout(row)
        return value

    def refresh(self) -> None:
        root = self.context.project_root
        values = (
            self.python_value,
            self.venv_value,
            self.torch_value,
            self.cuda_value,
            self.numpy_value,
            self.optuna_value,
            self.pointcloud_value,
        )
        if root is None:
            for value in values:
                value.setText('Проект не выбран')
            self.commands.clear()
            self.install_button.setEnabled(False)
            return

        if self._check_thread is not None and self._check_thread.isRunning():
            self._check_queued = self._check_root != root
            return
        for value in values:
            value.setText('Проверка…')
        self.install_button.setEnabled(False)
        self.commands.setPlainText(
            'Проверяем Python 3.11, PyTorch, CUDA и зависимости в фоновом режиме…'
        )
        thread = QThread(self)
        worker = EnvironmentCheckWorker(root)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._environment_checked)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(
            lambda current=thread: self._environment_thread_finished(current)
        )
        thread.finished.connect(thread.deleteLater)
        self._check_thread = thread
        self._check_worker = worker
        self._check_root = root
        thread.start()

    @Slot(object, object)
    def _environment_checked(
        self,
        root: Path,
        status: EnvironmentStatus,
    ) -> None:
        if self.context.project_root != root:
            return
        self._last_status = status
        self._last_status_root = root
        self.venv_value.setText('Готово' if status.ready else 'Требует настройки')
        self.python_value.setText(status.python_version or 'Не найден')
        self.numpy_value.setText(status.numpy or 'Не установлен')
        self.torch_value.setText(status.torch or 'Не установлен')
        self.optuna_value.setText(status.optuna or 'Не установлен')
        self.pointcloud_value.setText(status.torch_pointcloud or 'Не установлен')
        self.cuda_value.setText('Доступна' if status.cuda is True else 'Недоступна')
        self.install_button.setEnabled(True)
        self.install_button.setText(
            'Переустановить окружение'
            if status.ready
            else 'Установить окружение'
        )
        self.commands.setPlainText(
            'setup_ml_environment.cmd\n\n'
            ':: принудительная переустановка\n'
            'setup_ml_environment.cmd -Force\n\n'
            f'Статус: {status.error or "окружение готово"}\n'
        )

    def _environment_thread_finished(self, thread: QThread) -> None:
        if self._check_thread is thread:
            self._check_thread = None
            self._check_worker = None
            self._check_root = None
        if self._check_queued:
            self._check_queued = False
            self.refresh()

    def install_environment(self) -> None:
        root = self.context.project_root
        if root is None:
            self.project_selection_requested.emit()
            return
        status = (
            self._last_status
            if self._last_status_root == root
            else None
        )
        if status is None:
            self.refresh()
            QMessageBox.information(
                self,
                'Проверка окружения',
                'Дождитесь завершения фоновой проверки и повторите действие.',
            )
            return
        force = status.ready
        if force:
            answer = QMessageBox.question(
                self,
                'Переустановить окружение?',
                'Пакеты в .venv311 будут переустановлены. Продолжить?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self.installer.start(force=force)
        except (FileNotFoundError, RuntimeError) as error:
            QMessageBox.warning(self, 'Автоустановка недоступна', str(error))
            return
        self.install_button.setEnabled(False)
        self.commands.setPlainText('Установка запущена. Полный вывод доступен в Live.')

    @Slot(bool, str)
    def installation_finished(self, success: bool, message: str) -> None:
        self.refresh()
        if success:
            QMessageBox.information(self, 'Окружение готово', message)
        else:
            QMessageBox.warning(
                self,
                'Не удалось подготовить окружение',
                f'{message}. Подробности смотрите в Live.',
            )

    def open_project(self) -> None:
        if self.context.project_root is not None:
            open_in_file_manager(self.context.project_root)


STUDIO_PAGE_CLASSES = {
    'projects': ProjectsPage,
    'datasets': DatasetsPage,
    'runs': RunsPage,
    'live': LivePage,
    'testing': TestingPage,
    'scenarios': ScenariosPage,
    'artifacts': ArtifactsPage,
    'environment': EnvironmentPage,
}




