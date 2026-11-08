from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from point_cloud_viewer import PointCloudViewerDialog
from qt_runtime import prepare_qt_runtime
from resources import STYLE_FILES


def load_styles() -> str:
    return '\n'.join(
        path.read_text(encoding='utf-8')
        for path in STYLE_FILES
        if path.exists()
    )


def main() -> None:
    prepare_qt_runtime()
    source = (
        Path(sys.argv[1]).expanduser().resolve()
        if len(sys.argv) > 1
        else ROOT_DIR / 'example' / 'input_dataset' / 'Ogorod (1).xyz'
    )
    app = QApplication(sys.argv)
    app.setApplicationName('PointCloud')
    app.setStyleSheet(load_styles())
    viewer = PointCloudViewerDialog(source)
    viewer.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
