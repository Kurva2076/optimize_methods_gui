import sys
from pathlib import Path

import numpy as np

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle('Оптимизация облака точек')
        self.resize(700, 300)

        self.input_dir = ''
        self.output_dir = ''

        self.thread = None
        self.worker = None

        self.input_label = QLabel('Папка с данными не выбрана')
        self.input_label.setWordWrap(True)

        self.output_label = QLabel('Папка результатов не выбрана')
        self.output_label.setWordWrap(True)

        self.input_button = QPushButton('Выбрать папку с данными')
        self.output_button = QPushButton('Выбрать папку результатов')
        self.optimize_button = QPushButton('Оптимизировать данные')

        self.status_label = QLabel('Готово к работе')
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout()

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_button)
        input_row.addWidget(self.input_label, 1)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_button)
        output_row.addWidget(self.output_label, 1)

        layout.addLayout(input_row)
        layout.addLayout(output_row)
        layout.addSpacing(15)
        layout.addWidget(self.optimize_button)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.input_button.clicked.connect(self.select_input_dir)
        self.output_button.clicked.connect(self.select_output_dir)
        self.optimize_button.clicked.connect(self.start_optimization)

    def select_input_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, 'Выберите папку с файлами .xyz')
        if directory:
            self.input_dir = directory
            self.input_label.setText(directory)

    def select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, 'Выберите папку для результатов')
        if directory:
            self.output_dir = directory
            self.output_label.setText(directory)

    def start_optimization(self) -> None:
        if not self.input_dir:
            QMessageBox.warning(
                self,
                'Не выбрана папка',
                'Сначала выберите папку с исходными файлами.',
            )
            return

        if not self.output_dir:
            QMessageBox.warning(
                self,
                'Не выбрана папка',
                'Сначала выберите папку для результатов.',
            )
            return

        input_path = Path(self.input_dir).resolve()
        output_path = Path(self.output_dir).resolve()

        if input_path == output_path or input_path in output_path.parents or output_path in input_path.parents:
            QMessageBox.warning(
                self,
                'Некорректные папки',
                'Выбери две разные папки, не вложенные друг в друга.',
            )
            return

        self.optimize_button.setEnabled(False)
        self.input_button.setEnabled(False)
        self.output_button.setEnabled(False)

        self.status_label.setText('Подготовка к обработке...')

        self.thread = QThread(self)
        self.worker = OptimizationWorker(
            self.input_dir,
            self.output_dir,
        )

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
                self,
                'Готово',
                f'Успешно обработано файлов: {success_count} из {total}.\n\n'
                f'Результаты сохранены в:\n{self.output_dir}',
            )
        else:
            self.status_label.setText('Обработка завершена с ошибками.')
            QMessageBox.warning(
                self,
                'Обработка завершена не полностью',
                f'Успешно обработано файлов: {success_count} из {total}.'
                f'{report}\n\nРезультаты находятся в:\n{self.output_dir}',
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
            QMessageBox.warning(
                self,
                'Идёт обработка',
                'Дождитесь завершения оптимизации перед закрытием окна.',
            )
            event.ignore()
            return

        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
