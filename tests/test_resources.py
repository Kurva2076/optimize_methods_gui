from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from resources import apply_application_style, load_styles


def test_application_uses_fusion_style_on_every_platform() -> None:
    app = QApplication.instance() or QApplication([])

    apply_application_style(app)

    stylesheet = app.styleSheet()
    app.setStyleSheet('')
    assert app.style().objectName().casefold() == 'fusion'
    app.setStyleSheet(stylesheet)


def test_viewer_styles_leave_platform_controls_to_fusion() -> None:
    stylesheet = load_styles()

    assert 'QComboBox::drop-down' not in stylesheet
    assert 'QDoubleSpinBox::up-button' not in stylesheet
    assert 'QDoubleSpinBox::down-button' not in stylesheet


def test_missing_styles_are_reported_instead_of_silently_skipped(
    tmp_path: Path,
) -> None:
    missing_file = tmp_path / 'missing.qss'

    with pytest.raises(FileNotFoundError, match='missing.qss'):
        load_styles([missing_file])
