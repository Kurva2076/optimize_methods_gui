import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from screens.studio import DatasetsPage, ScenariosPage
from screens.studio_advanced import (
    AdvancedArtifactsPage,
    AdvancedLivePage,
    AdvancedRunsPage,
    AdvancedTestingPage,
)
from studio_context import StudioContext
from studio_data import (
    create_mini_dataset,
    evenly_spaced_scenes,
    scan_datasets,
    scan_runs,
)


def test_dataset_scan_reads_npy_shapes_and_totals(tmp_path: Path) -> None:
    exports = tmp_path / 'exports_small'
    exports.mkdir()
    first = np.zeros((12, 9), dtype=np.float32)
    second = np.zeros((7, 9), dtype=np.float32)
    np.save(exports / 'scene_001_cleaned.npy', first)
    np.save(exports / 'scene_002_cleaned.npy', second)

    datasets = scan_datasets(tmp_path)

    assert len(datasets) == 1
    assert datasets[0].path == exports
    assert [scene.points for scene in datasets[0].scenes] == [12, 7]
    assert datasets[0].points == 19
    assert all(scene.columns == 9 for scene in datasets[0].scenes)


def test_run_scan_uses_real_best_and_trials_artifacts(tmp_path: Path) -> None:
    run = tmp_path / 'runs' / 'ground_xyz'
    run.mkdir(parents=True)
    (run / 'best.json').write_text(
        json.dumps(
            {
                'target': 'ground',
                'features': 'xyz',
                'f1': 0.91,
                'baseline_f1': 0.72,
                'params': {'batch_size': 2},
            }
        ),
        encoding='utf-8',
    )
    with (run / 'trials.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['number', 'state', 'f1'])
        writer.writeheader()
        writer.writerow({'number': 0, 'state': 'COMPLETE', 'f1': 0.91})

    runs = scan_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].name == 'ground_xyz'
    assert runs[0].status == 'Завершён'
    assert runs[0].best_f1 == 0.91
    assert runs[0].baseline_f1 == 0.72


def test_datasets_workspace_highlights_second_workflow_stage() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    window.show_studio_section('datasets')

    assert isinstance(window.stack.currentWidget(), DatasetsPage)
    assert window.stepper._items[1].circle.property('active') is True
    assert window.stepper._items[0].circle.property('active') is False
    window.deleteLater()
    app.processEvents()


def test_main_window_uses_advanced_real_data_pages() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert isinstance(window.studio_pages['runs'], AdvancedRunsPage)
    assert isinstance(window.studio_pages['live'], AdvancedLivePage)
    assert isinstance(window.studio_pages['testing'], AdvancedTestingPage)
    assert isinstance(window.studio_pages['artifacts'], AdvancedArtifactsPage)
    window.deleteLater()
    app.processEvents()


def test_mini_dataset_uses_evenly_spaced_scenes_and_never_overwrites(
    tmp_path: Path,
) -> None:
    exports = tmp_path / 'exports_full'
    exports.mkdir()
    for index in range(5):
        np.save(
            exports / f'scene_{index:03d}_cleaned.npy',
            np.full((index + 1, 9), index, dtype=np.float32),
        )
    dataset = scan_datasets(tmp_path)[0]

    selected = evenly_spaced_scenes(dataset.scenes, 3)
    target = create_mini_dataset(dataset, 3)

    assert [scene.path.name for scene in selected] == [
        'scene_000_cleaned.npy',
        'scene_002_cleaned.npy',
        'scene_004_cleaned.npy',
    ]
    assert sorted(path.name for path in target.glob('*.npy')) == [
        'scene_000_cleaned.npy',
        'scene_002_cleaned.npy',
        'scene_004_cleaned.npy',
    ]
    with pytest.raises(FileExistsError):
        create_mini_dataset(dataset, 3)


def test_scenario_editor_saves_portable_structured_steps(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'hpo_pointnet.py').write_text('# marker\n', encoding='utf-8')
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    context = StudioContext(settings=settings)
    assert context.set_project(project)
    page = ScenariosPage(context)

    page.new_scenario()
    page.add_step()
    page.step_name.setText('Грунт')
    page.step_target.setCurrentText('ground')
    page.step_features.setCurrentText('xyz_hag')
    page.step_trials.setValue(3)
    page.step_extra.setText('--data-dir exports_small --output-dir runs/ground')
    page.update_current_step()
    page.save_scenario()

    payload = json.loads(page.current_path().read_text(encoding='utf-8'))
    assert payload['steps'][0]['title'] == 'Грунт'
    assert payload['steps'][0]['args'][0] == 'hpo_pointnet.py'
    assert '--data-dir' in payload['steps'][0]['args']
    assert not any('python.exe' in argument for argument in payload['steps'][0]['args'])
    page.deleteLater()
    app.processEvents()

