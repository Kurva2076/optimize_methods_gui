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
    assert viewer.camera_toolbar.parent() is viewer.canvas_card
    assert not viewer.camera_toolbar.isHidden()
    assert all(not section.is_expanded() for section in viewer._control_sections)

    appearance_section = viewer._control_sections[1]
    appearance_section.header.click()
    assert appearance_section.is_expanded()
    assert not appearance_section.content.isHidden()

    viewer._set_immersive_menu_visible(False)

    assert viewer.controls_panel.isHidden()
    assert not viewer.immersive_menu_button.isHidden()
    assert not viewer.camera_toolbar.isHidden()

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

    viewer.orbit_button.click()
    assert viewer.flight_speed_row.isHidden()
    assert viewer.canvas.navigation_mode == 'orbit'

    viewer.flight_button.click()
    assert not viewer.flight_speed_row.isHidden()
    assert viewer.canvas.navigation_mode == 'flight'

    viewer.close()
    app.processEvents()


def test_camera_toolbar_controls_update_the_canvas(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    viewer = PointCloudViewerDialog()

    viewer.field_of_view.setValue(90.0)
    assert viewer.canvas.field_of_view == 90.0

    viewer.orthographic_button.click()
    assert viewer.canvas.perspective is False
    assert not viewer.field_of_view.isEnabled()

    viewer.perspective_button.click()
    assert viewer.canvas.perspective is True
    assert viewer.field_of_view.isEnabled()

    viewer.flight_button.click()
    viewer.flight_speed.setValue(42.5)
    assert viewer.canvas.flight_speed == 42.5

    selected_views: list[str] = []
    monkeypatch.setattr(
        viewer.canvas,
        'set_view',
        lambda view: selected_views.append(view),
    )
    viewer.camera_view_actions['top'].trigger()
    assert selected_views == ['top']

    viewer.canvas.distance = 123.0
    viewer.camera_reset_button.click()
    assert viewer.canvas.distance != 123.0

    assert all(section.title != 'Камера' for section in viewer._control_sections)

    viewer.close()
    app.processEvents()


def test_camera_toolbar_compacts_for_the_minimum_window_width() -> None:
    app = QApplication.instance() or QApplication([])
    viewer = PointCloudViewerDialog()
    viewer.resize(viewer.minimumSize())
    viewer.show()
    viewer.flight_button.click()
    app.processEvents()

    assert viewer.camera_toolbar.property('compact') is True
    assert viewer.camera_toolbar.width() <= viewer.canvas.width() - 24

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
