import sys
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication

from studio_processes import ManagedProcess, _parse_duration


def _app() -> QCoreApplication:
    return QCoreApplication.instance() or QCoreApplication([])


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    app = _app()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def test_training_log_is_parsed_into_epoch_and_trial_history(tmp_path: Path) -> None:
    process = ManagedProcess('training', sys.executable, [], tmp_path)
    process.ingest_output(
        'INFO: trial=2 epoch=3 F1=0.7345 best=0.7345 baseline=0.5100 '
        'lr=0.001 время=0:01:02\n'
    )
    process.ingest_output('INFO: trial=2 COMPLETE за 0:02:05\n')

    assert process.last_f1 == 0.7345
    assert process.last_baseline == 0.51
    assert process.epochs[0].trial == 2
    assert process.epochs[0].epoch == 3
    assert process.epochs[0].elapsed_seconds == 62
    assert process.trials[0].state == 'COMPLETE'
    assert process.trials[0].best_f1 == 0.7345
    assert process.trials[0].elapsed_seconds == 125
    snapshot = process.snapshot()
    assert snapshot.epoch_count == 1
    assert snapshot.trial_count == 1


def test_prediction_progress_tracks_scene_patches_and_metrics(tmp_path: Path) -> None:
    process = ManagedProcess(
        'prediction',
        sys.executable,
        ['predict_scene.py'],
        tmp_path,
    )
    process.ingest_output('=== сцена 2/5: scene_002_cleaned.npy\n')
    process.ingest_output('патч 10/40 7s\n')
    process.ingest_output(
        '=== F1=0.8123 precision=0.8300 recall=0.7950 '
        'baseline=0.5400 не покрыто=0\n'
    )

    assert process.kind == 'prediction'
    assert len(process.predictions) == 3
    patch = process.predictions[1]
    assert (patch.scene_index, patch.scene_total) == (2, 5)
    assert (patch.patch_index, patch.patch_total) == (10, 40)
    assert patch.elapsed_seconds == 7
    result = process.predictions[2]
    assert result.scene_name == 'scene_002_cleaned.npy'
    assert result.f1 == 0.8123
    assert result.baseline == 0.54
    assert process.snapshot().prediction_count == 3


def test_memory_log_is_bounded_but_full_log_stays_on_disk(tmp_path: Path) -> None:
    process = ManagedProcess(
        'bounded log',
        sys.executable,
        [],
        tmp_path,
        max_log_chars=32,
    )
    process._open_log()
    source = ''.join(f'line-{index:02d}\n' for index in range(20))
    process.ingest_output(source)
    process._close_log()

    assert len(process.log_text()) <= 32
    assert process.log_path is not None
    full = process.full_log_text()
    assert 'line-00' in full
    assert 'line-19' in full
    assert '$ ' in full


def test_real_process_reports_complete_and_persists_output(tmp_path: Path) -> None:
    _app()
    code = (
        "print('trial=0 epoch=1 F1=0.7000 best=0.7000 "
        "baseline=0.4000 lr=0.001 время=0:00:01', flush=True)"
    )
    process = ManagedProcess('complete', sys.executable, ['-c', code], tmp_path)
    process.start()

    assert _wait_until(lambda: not process.is_running())
    assert process.status_key == 'complete'
    assert process.last_f1 == 0.7
    assert process.log_path is not None and process.log_path.is_file()
    assert 'trial=0 epoch=1' in process.full_log_text()


def test_user_stop_has_stopped_status(tmp_path: Path) -> None:
    _app()
    process = ManagedProcess(
        'stopped',
        sys.executable,
        ['-c', 'import time; time.sleep(10)'],
        tmp_path,
    )
    process.start()
    assert _wait_until(process.is_running)
    process.stop()

    assert _wait_until(lambda: not process.is_running())
    assert process.status_key == 'stopped'
    assert process.status == 'Остановлен'


def test_duration_parser_accepts_log_formats() -> None:
    assert _parse_duration('0:01:02') == 62
    assert _parse_duration('02:03') == 123
    assert _parse_duration('7s') == 7
    assert _parse_duration(None) is None
