from pathlib import Path

from PySide6.QtGui import QIcon

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / 'images'
DATA_DIR = BASE_DIR / 'data'
STYLE_FILES = [
    BASE_DIR / 'styles.qss',
    BASE_DIR / 'styles_shell.qss',
    BASE_DIR / 'styles_popup.qss',
]


def image_path(name: str) -> Path:
    return IMAGES_DIR / name


def icon(name: str) -> QIcon:
    return QIcon(str(image_path(name)))
