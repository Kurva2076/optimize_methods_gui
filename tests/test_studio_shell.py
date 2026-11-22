import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from resources import apply_application_style
from studio_context import StudioContext
from studio_theme import apply_studio_theme, stylesheet_for
from studio_widgets import StudioSidebar


def _settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / 'studio.ini'), QSettings.Format.IniFormat)


def _project(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / 'hpo_pointnet.py').write_text('# marker\n', encoding='utf-8')
    return path


def test_context_persists_recent_favorites_and_theme(tmp_path: Path) -> None:
    context = StudioContext(settings=_settings(tmp_path))
    first = _project(tmp_path / 'first')
    second = _project(tmp_path / 'second')
    theme_spy = QSignalSpy(context.theme_changed)

    assert context.set_project(first)
    assert (first / 'predict_scene.py').is_file()
    assert 'model=dict(' in (first / 'predict_scene.py').read_text(encoding='utf-8')
    assert context.set_project(second)
    assert context.recent_projects == [second.resolve(), first.resolve()]
    assert context.set_favorite(first)
    assert context.favorite_projects == [first.resolve()]
    assert context.toggle_favorite(first)
    assert context.favorite_projects == []

    context.remove_recent(first)
    assert context.recent_projects == [second.resolve()]
    context.set_theme('dark')
    assert context.theme == 'dark'
    assert theme_spy.count() == 1

    restored = StudioContext(settings=_settings(tmp_path))
    assert restored.project_root == second.resolve()
    assert restored.theme == 'dark'


def test_dark_stylesheet_is_a_complete_application_overlay() -> None:
    dark = stylesheet_for('dark')
    light = stylesheet_for('light')

    assert 'QWidget#studioWorkspace' in light
    assert '#0B1220' in dark
    assert len(dark) > len(light)


def test_sidebar_counts_project_entities_and_shows_running_process(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    apply_application_style(app)
    context = StudioContext(settings=_settings(tmp_path))
    root = _project(tmp_path / 'project')
    dataset = root / 'exports_small'
    dataset.mkdir()
    (dataset / 'scene_001_cleaned.npy').write_bytes(b'placeholder')
    run = root / 'runs' / 'ground_xyz'
    run.mkdir(parents=True)
    (run / 'best.json').write_text('{}', encoding='utf-8')
    assert context.set_project(root)

    sidebar = StudioSidebar(context)
    assert sidebar.buttons['datasets'].text() == 'Датасеты · 1'
    assert sidebar.buttons['runs'].text() == 'Запуски · 1'
    assert sidebar.project_path.text() == str(root.resolve())

    process = context.processes.create(
        'smoke',
        sys.executable,
        ['-c', 'import time; time.sleep(0.8)'],
        root,
    )
    process.start()
    QTest.qWait(150)
    sidebar.update_processes()
    assert not sidebar.process_panel.isHidden()
    assert sidebar.process_list_layout.count() == 1

    process.process.kill()
    process.process.waitForFinished(1000)
    sidebar.deleteLater()
    apply_studio_theme('light', app)
