from PySide6.QtWidgets import QApplication

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
