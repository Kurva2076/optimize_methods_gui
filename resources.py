from collections.abc import Iterable
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyleFactory

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / 'images'
DATA_DIR = BASE_DIR / 'data'
STYLE_FILES = [
    BASE_DIR / 'styles.qss',
    BASE_DIR / 'styles_shell.qss',
    BASE_DIR / 'styles_popup.qss',
    BASE_DIR / 'styles_training.qss',
    BASE_DIR / 'styles_viewer.qss',
]


def load_styles(style_files: Iterable[Path] | None = None) -> str:
    """Загрузить все QSS-файлы или явно сообщить о неполной сборке."""
    selected_files = tuple(STYLE_FILES if style_files is None else style_files)
    missing_files = [path for path in selected_files if not path.is_file()]
    if missing_files:
        missing = ', '.join(str(path) for path in missing_files)
        raise FileNotFoundError(f'Не найдены файлы стилей приложения: {missing}')
    return '\n'.join(path.read_text(encoding='utf-8') for path in selected_files)


def apply_application_style(app: QApplication) -> None:
    """Применить одинаковую базовую тему Qt на macOS, Windows и Linux."""
    fusion_style = QStyleFactory.create('Fusion')
    if fusion_style is None:
        raise RuntimeError('В установленной версии Qt отсутствует стиль Fusion.')
    app.setStyle(fusion_style)
    app.setStyleSheet(load_styles())


def image_path(name: str) -> Path:
    return IMAGES_DIR / name


def icon(name: str) -> QIcon:
    return QIcon(str(image_path(name)))
