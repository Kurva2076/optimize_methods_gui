import sys

from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from resources import STYLE_FILES


def load_styles() -> str:
    return '\n'.join(
        path.read_text(encoding='utf-8') for path in STYLE_FILES if path.exists()
    )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(load_styles())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
