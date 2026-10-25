from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, QTimer, Slot
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from card import apply_shadow, create_card
from resources import icon
from screens.base import BaseScreen
from ui_components import DropZone, RotatingIcon, StyledProgressBar
from workers import OptimizationWorker


class PreprocessingScreen(BaseScreen):
    step_title = 'Предобработка'
    title = 'Предобработка данных'
    subtitle = 'Загрузите файлы с облаками точек и запустите оптимизацию данных'

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.input_dir: str
        self.output_dir: str
        self.all_files: list[Path]
        self.excluded_files: list[Path]
        self.processing_times: dict[str, float]

        self._thread: QThread | None
        self._worker: OptimizationWorker | None

        self.rotation_angle: int
        self.rotation_timer: QTimer

    def init_state(self) -> None:
        self.input_dir = ''
        self.output_dir = ''
        self.all_files = []
        self.excluded_files = []
        self.processing_times = {}

        self._thread = None
        self._worker = None

        self.rotation_angle = 0
        self.rotation_timer = QTimer(self)
        self.rotation_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.rotation_timer.setInterval(16)
        self.rotation_timer.timeout.connect(self.rotate_status_icon)

    def build_content(self, layout: QVBoxLayout) -> None:
        columns = QHBoxLayout()
        columns.setSpacing(20)
        columns.addWidget(self._build_files_card(), 1)

        right = QVBoxLayout()
        right.setSpacing(20)
        right.addWidget(self._build_save_card(), 1)
        right.addWidget(self._build_status_card(), 1)
        columns.addLayout(right, 1)

        layout.addLayout(columns, 1)

    def _build_files_card(self) -> QFrame:
        card, layout = create_card(
            '1. Выбор файлов',
            'Выберите папку с исходными данными (можно несколько файлов).',
            icon_name='folder.svg',
        )
        layout.addSpacing(5)

        drop_zone = DropZone(callback=self.load_directory)
        drop_zone.setObjectName('dropZone')
        drop_zone.setMinimumHeight(120)
        drop_layout = QVBoxLayout(drop_zone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        drop_text1 = QLabel('Перетащите папку сюда')
        drop_text1.setStyleSheet('font-size: 18px; font-weight: 600; color: #171717;')
        drop_text2 = QLabel('или выберите вручную')
        drop_text2.setStyleSheet('font-size: 13px; color: #737B89; margin-bottom: 10px;')

        self.input_button = QPushButton('Выберите папку')
        self.input_button.setObjectName('primaryButton')
        self.input_button.setFixedWidth(180)
        self.input_button.clicked.connect(self.select_input_dir)

        drop_layout.addWidget(drop_text1, alignment=Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(drop_text2, alignment=Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(self.input_button, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(drop_zone)
        layout.addSpacing(10)
        apply_shadow(drop_zone, blur=15, y_offset=3, alpha=25)

        self.files_count_label = QLabel('Выбранные файлы (0)')
        self.files_count_label.setStyleSheet('font-weight: bold; font-size: 14px; color: #171717;')
        layout.addWidget(self.files_count_label)

        self.file_table = QTableWidget()
        self.file_table.setColumnCount(5)
        self.file_table.setHorizontalHeaderLabels(['Имя файла', 'Формат', 'Размер', 'Время обработки', ''])
        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.setShowGrid(False)
        self.file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.file_table)

        return card

    def _build_save_card(self) -> QFrame:
        card, layout = create_card(
            '2. Параметры сохранения',
            'Настройки сохранения данных.',
            icon_name='gear.svg',
        )

        path_label = QLabel('Папка для хранения результатов')
        path_label.setStyleSheet('font-size: 13px; color: #171717;')
        layout.addWidget(path_label)

        path_row = QHBoxLayout()
        self.output_label = QLabel('Папка не выбрана')
        self.output_label.setObjectName('pathLabel')
        apply_shadow(self.output_label, blur=12, y_offset=2, alpha=25)

        self.output_button = QPushButton()
        self.output_button.setObjectName('iconButton')
        self.output_button.setFixedSize(38, 38)
        self.output_button.setIcon(icon('loupe-search.svg'))
        self.output_button.setIconSize(QSize(20, 20))
        self.output_button.clicked.connect(self.select_output_dir)

        path_row.addWidget(self.output_label, 1)
        path_row.addWidget(self.output_button)
        layout.addLayout(path_row)
        layout.addSpacing(10)

        self.optimize_button = QPushButton('Оптимизировать данные')
        self.optimize_button.setObjectName('primaryButton')
        self.optimize_button.setMinimumHeight(50)
        self.optimize_button.clicked.connect(self.start_optimization)
        layout.addWidget(self.optimize_button)

        return card

    def _build_status_card(self) -> QFrame:
        card, layout = create_card(
            '3. Статус обработки',
            'Здесь будет отображаться ход выполнения и результат.',
        )

        status_container = QFrame()
        status_container.setObjectName('statusContainer')
        status_container.setMinimumHeight(80)
        container_layout = QHBoxLayout(status_container)
        container_layout.setContentsMargins(20, 15, 20, 15)
        container_layout.setSpacing(15)

        status_pixmap: QPixmap = icon('circle.svg').pixmap(QSize(32, 32))
        self.status_icon_label = RotatingIcon(status_pixmap)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        text_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.status_title = QLabel('Ожидает запуска')
        self.status_title.setObjectName('statusTitle')
        self.status_desc = QLabel('Загрузите файлы и запустите оптимизацию данных.')
        self.status_desc.setObjectName('statusDesc')
        self.status_desc.setWordWrap(True)
        text_layout.addWidget(self.status_title)
        text_layout.addWidget(self.status_desc)

        container_layout.addWidget(self.status_icon_label)
        container_layout.addLayout(text_layout, 1)

        apply_shadow(status_container, blur=15, y_offset=3, alpha=25)
        layout.addWidget(status_container)

        self.progress_bar = StyledProgressBar()
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('Ожидание: %p%')
        layout.addWidget(self.progress_bar)

        layout.addStretch(1)
        return card

    def load_directory(self, directory: str) -> None:
        self.input_dir = directory
        try:
            self.all_files = sorted(
                f for f in Path(directory).iterdir()
                if f.is_file() and f.suffix.lower() == '.xyz'
            )
        except OSError:
            self.all_files = []
        self.excluded_files = []
        self.update_table()

    def select_input_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, 'Выберите папку с файлами .xyz')
        if directory:
            self.load_directory(directory)

    def select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, 'Выберите папку для результатов')
        if directory:
            self.output_dir = directory
            self.output_label.setText(directory)

    def remove_file(self, file_path: Path) -> None:
        if file_path not in self.excluded_files:
            self.excluded_files.append(file_path)
            self.update_table()

    def update_table(self) -> None:
        self.file_table.setRowCount(0)
        files_to_show = [f for f in self.all_files if f not in self.excluded_files]
        self.files_count_label.setText(f'Выбранные файлы ({len(files_to_show)})')

        text_color = QColor('#171717')
        close_icon = icon('close.svg')

        for row, f in enumerate(files_to_show):
            self.file_table.insertRow(row)
            size_mb = f.stat().st_size / (1024 * 1024)
            elapsed = self.processing_times.get(str(f))

            cells = [
                f.name,
                f.suffix.lower(),
                f'{size_mb:.1f} МБ',
                f'{elapsed:.2f} с' if elapsed is not None else '—',
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(text_color)
                self.file_table.setItem(row, column, item)

            btn = QPushButton()
            btn.setIcon(close_icon)
            btn.setIconSize(QSize(14, 14))
            btn.setStyleSheet('background: transparent; border: none;')
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, p=f: self.remove_file(p))
            self.file_table.setCellWidget(row, 4, btn)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.optimize_button.setEnabled(enabled)
        self.input_button.setEnabled(enabled)
        self.output_button.setEnabled(enabled)

    def start_optimization(self) -> None:
        if not self.input_dir:
            QMessageBox.warning(self, 'Не выбрана папка', 'Сначала выберите папку с исходными файлами.')
            return

        if not self.output_dir:
            QMessageBox.warning(self, 'Не выбрана папка', 'Сначала выберите папку для результатов.')
            return

        input_path = Path(self.input_dir).resolve()
        output_path = Path(self.output_dir).resolve()
        if (
            input_path == output_path
            or input_path in output_path.parents
            or output_path in input_path.parents
        ):
            QMessageBox.warning(self, 'Некорректные папки', 'Выбери две разные папки, не вложенные друг в друга.')
            return

        if not self.all_files:
            QMessageBox.warning(self, 'Нет файлов', 'В выбранной папке нет файлов .xyz.')
            return

        if len(self.all_files) == len(self.excluded_files):
            QMessageBox.warning(self, 'Нет файлов', 'Все файлы исключены из обработки.')
            return

        self._set_controls_enabled(False)
        self.status_title.setText('Подготовка к обработке...')
        self.status_desc.setText('Инициализация процесса...')
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('Обработка: %p%')
        self.rotation_angle = 0
        self.status_icon_label.set_angle(0)
        self.rotation_timer.start()
        self.set_busy(True)

        thread = QThread(self)
        worker = OptimizationWorker(self.input_dir, self.output_dir, self.excluded_files)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self.update_progress)
        worker.file_processed.connect(self.update_file_processing_time)
        worker.finished.connect(self.optimization_finished)
        worker.failed.connect(self.optimization_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)

        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    @Slot(str, int, int)
    def update_progress(self, message: str, completed: int, total: int) -> None:
        self.status_title.setText('Идет обработка...')
        self.status_desc.setText(message)
        self.progress_bar.setValue(round(completed / total * 100) if total else 0)

    @Slot(str, float)
    def update_file_processing_time(self, file_path: str, elapsed: float) -> None:
        self.processing_times[file_path] = elapsed
        self.update_table()

    def rotate_status_icon(self) -> None:
        self.rotation_angle = (self.rotation_angle + 6) % 360
        self.status_icon_label.set_angle(self.rotation_angle)

    def stop_status_animation(self) -> None:
        self.rotation_timer.stop()
        self.rotation_angle = 0
        self.status_icon_label.set_angle(0)

    @Slot(int, int, str)
    def optimization_finished(self, success_count: int, total: int, report: str) -> None:
        self._set_controls_enabled(True)
        self.stop_status_animation()
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat('Завершено: %p%')
        self.set_busy(False)

        if success_count == total:
            self.status_title.setText('Оптимизация успешно завершена')
            self.status_desc.setText(f'Обработано файлов: {success_count} из {total}.')
            QMessageBox.information(
                self,
                'Готово',
                f'Успешно обработано файлов: {success_count} из {total}.\n\n'
                f'Результаты сохранены в:\n{self.output_dir}',
            )
        else:
            self.status_title.setText('Обработка завершена с ошибками')
            self.status_desc.setText(f'Успешно: {success_count} из {total}. Ошибки в окне сообщения.')
            QMessageBox.warning(
                self,
                'Обработка завершена не полностью',
                f'Успешно обработано файлов: {success_count} из {total}.'
                f'{report}\n\nРезультаты находятся в:\n{self.output_dir}',
            )

    @Slot(str)
    def optimization_failed(self, error: str) -> None:
        self._set_controls_enabled(True)
        self.stop_status_animation()
        self.progress_bar.setFormat('Ошибка обработки')
        self.status_title.setText('Не удалось выполнить оптимизацию')
        self.status_desc.setText('Проверьте исходные файлы и попробуйте снова.')
        self.set_busy(False)
        QMessageBox.critical(self, 'Ошибка оптимизации', error)
