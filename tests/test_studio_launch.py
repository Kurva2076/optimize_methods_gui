import json
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

import studio_dialogs
from studio_dialogs import (
    FIXED_PARAM_DEFAULTS,
    LaunchDialog,
    launch_compatibility_differences,
    parse_extra_arguments,
    validate_fixed_params,
    write_json_exclusive,
)
from studio_environment import inspect_environment
from studio_processes import ProcessRegistry


class FakeContext:
    def __init__(self, root: Path) -> None:
        self.project_root = root
        self.processes = ProcessRegistry()

    @staticmethod
    def python_executable() -> Path:
        return Path(sys.executable)


def project(tmp_path: Path) -> Path:
    (tmp_path / 'hpo_pointnet.py').write_text('', encoding='utf-8')
    (tmp_path / 'run_all.py').write_text('', encoding='utf-8')
    (tmp_path / 'exports').mkdir()
    (tmp_path / 'exports_final').mkdir()
    return tmp_path


def test_single_launch_command_uses_real_hpo_arguments(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    root = project(tmp_path)
    dialog = LaunchDialog(FakeContext(root), root / 'exports')
    dialog.runs_dir.setText(str(root / 'custom-runs'))
    dialog.run_name.setText('vegetation_xyz_hag')
    dialog.target.setCurrentText('vegetation')
    dialog.features.setCurrentText('xyz_hag')
    dialog.extra_args.setText('--boundary-space xy')
    dialog.manual_params.setChecked(True)

    command = dialog.command()

    assert command[1] == str(root / 'hpo_pointnet.py')
    assert command[command.index('--data-dir') + 1] == str(root / 'exports')
    assert command[command.index('--output-dir') + 1] == str(
        root / 'custom-runs' / 'vegetation_xyz_hag'
    )
    assert command[command.index('--target') + 1] == 'vegetation'
    assert command[command.index('--features') + 1] == 'xyz_hag'
    assert command[-2:] == ['--boundary-space', 'xy']
    assert '--fixed-params' in command
    dialog.deleteLater()
    app.processEvents()


def test_scenario_command_uses_run_all_and_final_dataset(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    root = project(tmp_path)
    dialog = LaunchDialog(FakeContext(root), root / 'exports')
    dialog.mode.setCurrentIndex(1)
    dialog.tasks.setText('2 3 final 7')
    dialog.runs_dir.setText(str(root / 'runs-other'))
    dialog.final_dataset.setText(str(root / 'exports_final'))

    command = dialog.command()

    assert command[1] == str(root / 'run_all.py')
    tasks = command[command.index('--tasks') + 1 : command.index('--data')]
    assert tasks == ['2', '3', 'final', '7']
    assert command[command.index('--final-data') + 1] == str(root / 'exports_final')
    assert command[command.index('--runs') + 1] == str(root / 'runs-other')
    dialog.deleteLater()
    app.processEvents()


def test_extra_arguments_are_parsed_without_shell() -> None:
    assert parse_extra_arguments('--name "two words" --flag') == [
        '--name',
        'two words',
        '--flag',
    ]
    with pytest.raises(ValueError):
        parse_extra_arguments('--name "not closed')


def test_extra_arguments_keep_windows_backslashes(monkeypatch) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(studio_dialogs.os, 'name', 'nt')
        assert parse_extra_arguments(r'--data C:\LiDAR\exports --flag') == [
            '--data',
            r'C:\LiDAR\exports',
            '--flag',
        ]


def test_fixed_params_validate_ranges_and_never_overwrite(tmp_path: Path) -> None:
    valid = validate_fixed_params(FIXED_PARAM_DEFAULTS)
    path = tmp_path / 'params.json'
    write_json_exclusive(path, valid)
    write_json_exclusive(path, valid)
    changed = dict(valid, lr=0.002)
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, changed)
    with pytest.raises(ValueError, match='dropout'):
        validate_fixed_params(dict(valid, dropout=0.8))


def test_continuation_reports_exact_incompatible_field(tmp_path: Path) -> None:
    output = tmp_path / 'runs' / 'ground_xyz'
    output.mkdir(parents=True)
    (output / 'study.db').touch()
    expected = {
        'data_dir': str(tmp_path / 'exports'),
        'seed': 42,
        'epochs': 10,
    }
    (output / 'split.json').write_text(
        json.dumps(dict(expected, epochs=8)),
        encoding='utf-8',
    )

    differences = launch_compatibility_differences(output, expected)

    assert differences == ['epochs: было 8, запрошено 10']


def test_environment_check_is_explicit_when_venv_missing(tmp_path: Path) -> None:
    status = inspect_environment(tmp_path)
    assert status.ready is False
    assert status.error == '.venv311 не найдена'


def test_windows_build_is_one_file() -> None:
    build = Path('build_windows.ps1').read_text(encoding='utf-8')
    assert '--onefile' in build
    assert '--onedir' not in build
    assert '--windowed' in build
