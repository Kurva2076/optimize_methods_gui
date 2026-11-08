"""Открыть экран результатов обучения с демонстрационными данными."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from qt_runtime import prepare_qt_runtime
from resources import apply_application_style
from screens.training import TrainingScreen


def main() -> None:
    prepare_qt_runtime()
    app = QApplication(sys.argv)
    apply_application_style(app)

    window = MainWindow()
    window.go_to(2)
    training_screen = window.screens[2]
    if not isinstance(training_screen, TrainingScreen):
        raise TypeError('Третий шаг приложения не является экраном обучения.')

    demo_dir = ROOT_DIR / 'example' / 'training_run_demo'
    if not training_screen.load_results(demo_dir):
        raise RuntimeError(f'Не удалось загрузить демонстрационные данные: {demo_dir}')

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
