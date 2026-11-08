from types import SimpleNamespace

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

import point_cloud_canvas
from point_cloud_viewer import PointCloudViewerDialog


def test_immersive_layout_can_be_entered_hidden_and_closed(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    viewer = PointCloudViewerDialog()
    monkeypatch.setattr(viewer, 'showFullScreen', lambda: None)
    monkeypatch.setattr(viewer, 'showMaximized', lambda: None)
    monkeypatch.setattr(viewer, 'showNormal', lambda: None)

    viewer.set_immersive_mode(True)

    assert viewer._immersive_mode is True
    assert viewer.header.isHidden()
    assert viewer.controls_panel.parent() is viewer.canvas_card
    assert all(not section.is_expanded() for section in viewer._control_sections)

    appearance_section = viewer._control_sections[1]
    appearance_section.header.click()
    assert appearance_section.is_expanded()
    assert not appearance_section.content.isHidden()

    viewer._set_immersive_menu_visible(False)

    assert viewer.controls_panel.isHidden()
    assert not viewer.immersive_menu_button.isHidden()

    viewer.reject()

    assert viewer._immersive_mode is False
    assert not viewer.header.isHidden()
    assert viewer.controls_panel.parent() is viewer
    assert viewer.immersive_menu_button.isHidden()
    viewer.close()
    app.processEvents()


def test_free_flight_accepts_russian_keyboard_layout() -> None:
    app = QApplication.instance() or QApplication([])
    viewer = PointCloudViewerDialog()
    viewer.canvas.set_navigation_mode('flight')

    press = QKeyEvent(
        QEvent.Type.KeyPress,
        ord('Ц'),
        Qt.KeyboardModifier.NoModifier,
        'ц',
    )
    viewer.canvas.keyPressEvent(press)

    assert Qt.Key.Key_W in viewer.canvas._pressed_keys

    release = QKeyEvent(
        QEvent.Type.KeyRelease,
        ord('Ц'),
        Qt.KeyboardModifier.NoModifier,
        'ц',
    )
    viewer.canvas.keyReleaseEvent(release)

    assert Qt.Key.Key_W not in viewer.canvas._pressed_keys
    viewer.close()
    app.processEvents()


def test_free_flight_repaints_immediately_while_looking_around(
    monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    viewer = PointCloudViewerDialog()
    canvas = viewer.canvas
    canvas.set_navigation_mode('flight')
    canvas_repaints: list[bool] = []
    window_repaints: list[object] = []
    window_syncs: list[bool] = []
    monkeypatch.setattr(canvas, 'repaint', lambda: canvas_repaints.append(True))
    monkeypatch.setattr(
        viewer,
        'repaint',
        lambda region: window_repaints.append(region),
    )
    monkeypatch.setattr(
        point_cloud_canvas,
        'QGuiApplication',
        SimpleNamespace(
            sync=lambda: window_syncs.append(True)
        ),
    )

    canvas.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(100.0, 100.0),
            QPointF(100.0, 100.0),
            QPointF(100.0, 100.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    previous_yaw = canvas.yaw
    canvas.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(140.0, 85.0),
            QPointF(140.0, 85.0),
            QPointF(140.0, 85.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )

    assert canvas.yaw != previous_yaw
    assert canvas_repaints == [True]
    assert len(window_repaints) == 1
    assert window_syncs == [True]

    canvas._last_drag_repaint = float('inf')
    canvas.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(160.0, 70.0),
            QPointF(160.0, 70.0),
            QPointF(160.0, 70.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert canvas_repaints == [True]

    canvas.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(160.0, 70.0),
            QPointF(160.0, 70.0),
            QPointF(160.0, 70.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert canvas_repaints == [True, True]
    assert len(window_repaints) == 2
    assert window_syncs == [True, True]

    viewer.close()
    app.processEvents()


def test_flight_speed_is_only_visible_in_free_flight() -> None:
    app = QApplication.instance() or QApplication([])
    viewer = PointCloudViewerDialog()

    viewer.navigation_combo.setCurrentIndex(
        viewer.navigation_combo.findData('orbit')
    )
    assert viewer.flight_speed_row.isHidden()

    viewer.navigation_combo.setCurrentIndex(
        viewer.navigation_combo.findData('flight')
    )
    assert not viewer.flight_speed_row.isHidden()

    viewer.close()
    app.processEvents()


def test_controls_panel_never_scrolls_horizontally() -> None:
    app = QApplication.instance() or QApplication([])
    viewer = PointCloudViewerDialog()
    viewer.show()
    app.processEvents()

    scroll = viewer.controls_panel
    assert scroll.widget().width() == scroll.viewport().width()
    assert scroll.horizontalScrollBar().maximum() == 0

    scroll.horizontalScrollBar().setValue(50)
    assert scroll.horizontalScrollBar().value() == 0

    viewer.close()
    app.processEvents()
