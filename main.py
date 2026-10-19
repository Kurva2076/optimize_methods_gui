import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtGui import QCloseEvent, QColor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGraphicsDropShadowEffect,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView
)

from lib.data.xyz_loader import XYZLoader
from lib.data.xyz_saver import save_points_to_xyz
from lib.optimization.filtering import clean, ALS_PRESET


class OptimizationWorker(QObject):
    """Обрабатывает XYZ-файлы в отдельном потоке."""

    progress = Signal(str)
    finished = Signal(int, int, str)
    failed = Signal(str)

    def __init__(self, input_dir: str, output_dir: str) -> None:
        super().__init__()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    @Slot()
    def run(self) -> None:
        try:
            files = sorted(
                path
                for path in self.input_dir.iterdir()
                if path.is_file() and path.suffix.lower() == '.xyz'
            )
            if not files:
                raise ValueError('В выбранной папке нет файлов с расширением .xyz.')

            self.output_dir.mkdir(parents=True, exist_ok=True)

            success_count = 0
            error_messages = []
            for index, file_path in enumerate(files, start=1):
                self.progress.emit(f'Обработка: {file_path.name}')

                try:
                    classes, points = XYZLoader(str(file_path)).load(dtype=np.float64)
                    result = clean(points, classes, steps=ALS_PRESET)

                    clean_classes = classes[result.keep]
                    clean_points = points[result.keep]
                    clean_path = self.output_dir / f'{file_path.stem}_clean.xyz'

                    save_points_to_xyz(
                        points=clean_points,
                        filepath=str(clean_path),
                        classes=clean_classes,
                    )

                    noise_mask = ~result.keep
                    if np.any(noise_mask):
                        noise_path = self.output_dir / f'{file_path.stem}_noise.xyz'

                        save_points_to_xyz(
                            points=points[noise_mask],
                            filepath=str(noise_path),
                            classes=result.codes[noise_mask],
                        )

                    success_count += 1

                except Exception as error:
                    error_messages.append(f'{file_path.name}: {error}')

                self.progress.emit(f'Обработано файлов: {index} из {len(files)}')

            report = '\n\nОшибки:\n' + '\n'.join(error_messages) if error_messages else ''
            self.finished.emit(success_count, len(files), report)

        except Exception as e:
            self.failed.emit(str(e))


class DropZone(QFrame):
    """Кастомный фрейм, который принимает перетаскивание папок и файлов."""

    def __init__(self, parent=None, callback=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.callback = callback

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("drag_active", "true")
            self.style().unpolish(self)
            self.style().polish(self)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("drag_active", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        event.accept()

    def dropEvent(self, event):
        self.setProperty("drag_active", "false")
        self.style().unpolish(self)
        self.style().polish(self)

        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            target_dir = path if os.path.isdir(path) else os.path.dirname(path)

            if self.callback:
                self.callback(target_dir)
        event.acceptProposedAction()

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Оптимизация облака точек')
        self.resize(1100, 550)
        self.setMinimumSize(1100, 550)

        screen_geometry = QApplication.primaryScreen().availableGeometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

        self.input_dir = ''
        self.output_dir = ''
        self.thread: Optional[QThread] = None
        self.worker: Optional[OptimizationWorker] = None

        self.input_button: Optional[QPushButton] = None
        self.files_count_label: Optional[QLabel] = None
        self.file_table: Optional[QTableWidget] = None
        self.output_label: Optional[QLabel] = None
        self.output_button: Optional[QPushButton] = None
        self.optimize_button: Optional[QPushButton] = None
        self.status_label: Optional[QLabel] = None

        self.init_ui()

    @staticmethod
    def apply_shadow(widget, blur=22, y_offset=5, alpha=38):
        """Вспомогательная функция для создания теней."""
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur)
        shadow.setXOffset(0)
        shadow.setYOffset(y_offset)
        shadow.setColor(QColor(25, 40, 65, alpha))
        widget.setGraphicsEffect(shadow)

    def init_ui(self) -> None:
        central_widget = QWidget()
        central_widget.setObjectName('centralWidget')
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(26, 26, 26, 26)
        main_layout.setSpacing(20)

        left_card = QFrame()
        left_card.setObjectName('card')
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(22, 22, 22, 22)
        left_layout.setSpacing(10)

        title_layout = QHBoxLayout()
        icon_label = QLabel("🗀")
        icon_label.setStyleSheet("font-size: 24px; color: #171717;")
        title_label = QLabel("1. Выбор файлов")
        title_label.setObjectName('cardTitle')
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        left_layout.addLayout(title_layout)

        desc_label = QLabel("Выберите папку с исходными данными (можно несколько файлов).")
        desc_label.setObjectName('cardDescription')
        left_layout.addWidget(desc_label)
        left_layout.addSpacing(10)

        drop_zone = DropZone(callback=self.process_input_directory)
        drop_zone.setObjectName('dropZone')
        drop_zone.setObjectName('dropZone')
        drop_zone.setMinimumHeight(120)
        drop_layout = QVBoxLayout(drop_zone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        drop_text1 = QLabel("Перетащите папку сюда")
        drop_text1.setStyleSheet("font-size: 18px; font-weight: 600; color: #171717;")
        drop_text2 = QLabel("или выберите вручную")
        drop_text2.setStyleSheet("font-size: 13px; color: #737B89; margin-bottom: 10px;")

        self.input_button = QPushButton("Выберите папку")
        self.input_button.setObjectName('primaryButton')
        self.input_button.setFixedWidth(180)
        self.input_button.clicked.connect(self.select_input_dir)

        drop_layout.addWidget(drop_text1, alignment=Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(drop_text2, alignment=Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(self.input_button, alignment=Qt.AlignmentFlag.AlignCenter)

        left_layout.addWidget(drop_zone)
        left_layout.addSpacing(10)

        self.apply_shadow(drop_zone, blur=15, y_offset=3, alpha=25)

        self.files_count_label = QLabel("Выбранные файлы (0)")
        self.files_count_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #171717;")
        left_layout.addWidget(self.files_count_label)

        self.file_table = QTableWidget()
        self.file_table.setColumnCount(4)
        self.file_table.setHorizontalHeaderLabels(["Имя файла", "Формат", "Размер", ""])

        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.file_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        self.file_table.verticalHeader().setVisible(False)
        self.file_table.setShowGrid(False)
        self.file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        left_layout.addWidget(self.file_table)

        self.apply_shadow(left_card)
        main_layout.addWidget(left_card, 1)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)

        card2 = QFrame()
        card2.setObjectName('card')
        c2_layout = QVBoxLayout(card2)
        c2_layout.setContentsMargins(22, 22, 22, 22)
        c2_layout.setSpacing(10)

        c2_title_layout = QHBoxLayout()
        c2_icon = QLabel("⚙\ufe0e")
        c2_icon.setStyleSheet("font-family: 'Segoe UI Symbol', 'Arial'; font-size: 24px; color: #171717;")
        c2_title = QLabel("2. Параметры сохранения")
        c2_title.setObjectName('cardTitle')
        c2_title_layout.addWidget(c2_icon)
        c2_title_layout.addWidget(c2_title)
        c2_title_layout.addStretch()
        c2_layout.addLayout(c2_title_layout)

        c2_desc = QLabel("Настройки сохранения данных.")
        c2_desc.setObjectName('cardDescription')
        c2_layout.addWidget(c2_desc)
        c2_layout.addSpacing(5)

        c2_path_label = QLabel("Папка для хранения результатов")
        c2_path_label.setStyleSheet("font-size: 13px; color: #171717;")
        c2_layout.addWidget(c2_path_label)

        path_row = QHBoxLayout()
        self.output_label = QLabel("Папка не выбрана")
        self.output_label.setObjectName('pathLabel')

        self.apply_shadow(self.output_label, blur=12, y_offset=2, alpha=25)

        self.output_button = QPushButton("⌕")
        self.output_button.setObjectName('iconButton')
        self.output_button.setFixedSize(38, 38)
        self.output_button.clicked.connect(self.select_output_dir)

        path_row.addWidget(self.output_label, 1)
        path_row.addWidget(self.output_button)
        c2_layout.addLayout(path_row)
        c2_layout.addSpacing(10)

        self.optimize_button = QPushButton("Оптимизировать данные")
        self.optimize_button.setObjectName('primaryButton')
        self.optimize_button.setMinimumHeight(50)
        self.optimize_button.clicked.connect(self.start_optimization)
        c2_layout.addWidget(self.optimize_button)

        self.apply_shadow(card2)
        right_layout.addWidget(card2, 1)

        card3 = QFrame()
        card3.setObjectName('card')
        c3_layout = QVBoxLayout(card3)
        c3_layout.setContentsMargins(22, 22, 22, 22)
        c3_layout.setSpacing(10)

        c3_title_layout = QHBoxLayout()
        c3_title = QLabel("3. Статус обработки")
        c3_title.setObjectName('cardTitle')
        c3_title_layout.addWidget(c3_title)
        c3_title_layout.addStretch()
        c3_layout.addLayout(c3_title_layout)

        c3_desc = QLabel("Здесь будет отображаться ход выполнения и результат.")
        c3_desc.setObjectName('cardDescription')
        c3_layout.addWidget(c3_desc)
        c3_layout.addSpacing(5)

        self.status_label = QLabel("Готово к работе\nЗагрузите файлы и запустите оптимизацию данных.")
        self.status_label.setObjectName('statusLabel')
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(70)
        self.apply_shadow(self.status_label, blur=15, y_offset=3, alpha=25)
        c3_layout.addWidget(self.status_label)

        c3_layout.addStretch(1)
        self.apply_shadow(card3)
        right_layout.addWidget(card3, 1)

        main_layout.addLayout(right_layout, 1)

    def select_input_dir(self) -> None:
        # Эта функция открывает диалоговое окно
        directory = QFileDialog.getExistingDirectory(self, 'Выберите папку с файлами .xyz')
        if directory:
            self.process_input_directory(directory)

    def process_input_directory(self, directory: str) -> None:
        self.input_dir = directory
        self.file_table.setRowCount(0)
        path = Path(directory)

        try:
            xyz_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() == '.xyz']
        except OSError:
            xyz_files = []

        self.files_count_label.setText(f"Выбранные файлы ({len(xyz_files)})")

        for row, f in enumerate(xyz_files):
            self.file_table.insertRow(row)
            size_mb = f.stat().st_size / (1024 * 1024)

            self.file_table.setItem(row, 0, QTableWidgetItem(f.name))
            self.file_table.setItem(row, 1, QTableWidgetItem(f.suffix.lower()))
            self.file_table.setItem(row, 2, QTableWidgetItem(f"{size_mb:.1f} МБ"))
            self.file_table.setItem(row, 3, QTableWidgetItem("✕"))

    def select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, 'Выберите папку для результатов')
        if directory:
            self.output_dir = directory
            self.output_label.setText(directory)

    def start_optimization(self) -> None:
        if not self.input_dir:
            QMessageBox.warning(self, 'Не выбрана папка', 'Сначала выберите папку с исходными файлами.')
            return

        if not self.output_dir:
            QMessageBox.warning(self, 'Не выбрана папка', 'Сначала выберите папку для результатов.')
            return

        input_path = Path(self.input_dir).resolve()
        output_path = Path(self.output_dir).resolve()

        if input_path == output_path or input_path in output_path.parents or output_path in input_path.parents:
            QMessageBox.warning(self, 'Некорректные папки', 'Выбери две разные папки, не вложенные друг в друга.')
            return

        self.optimize_button.setEnabled(False)
        self.input_button.setEnabled(False)
        self.output_button.setEnabled(False)

        self.status_label.setText('Подготовка к обработке...')

        self.thread = QThread(self)
        self.worker = OptimizationWorker(self.input_dir, self.output_dir)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.optimization_finished)
        self.worker.failed.connect(self.optimization_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.cleanup_thread)

        self.thread.start()

    @Slot(str)
    def update_progress(self, message: str) -> None:
        self.status_label.setText(message)

    @Slot(int, int, str)
    def optimization_finished(self, success_count: int, total: int, report: str) -> None:
        if success_count == total:
            self.status_label.setText('Оптимизация успешно завершена.')
            QMessageBox.information(
                self, 'Готово',
                f'Успешно обработано файлов: {success_count} из {total}.\n\nРезультаты сохранены в:\n{self.output_dir}'
            )
        else:
            self.status_label.setText('Обработка завершена с ошибками.')
            QMessageBox.warning(
                self, 'Обработка завершена не полностью',
                f'Успешно обработано файлов: {success_count} из {total}.{report}\n\nРезультаты находятся в:\n{self.output_dir}'
            )

    @Slot(str)
    def optimization_failed(self, error: str) -> None:
        self.status_label.setText('Не удалось выполнить оптимизацию.')
        QMessageBox.critical(self, 'Ошибка оптимизации', error)

    @Slot()
    def cleanup_thread(self) -> None:
        self.thread = None
        self.worker = None

        self.optimize_button.setEnabled(True)
        self.input_button.setEnabled(True)
        self.output_button.setEnabled(True)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, 'Идёт обработка', 'Дождитесь завершения оптимизации перед закрытием окна.')
            event.ignore()
            return
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    style_path = Path(__file__).resolve().parent / 'styles.qss'
    app.setStyleSheet(style_path.read_text(encoding='utf-8'))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()