import csv
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from screens.studio_advanced import (
    AdvancedArtifactsPage,
    AdvancedLivePage,
    AdvancedRunsPage,
    AdvancedTestingPage,
    parse_live_metrics,
)
from studio_context import StudioContext


def _context(root: Path) -> StudioContext:
    (root / 'hpo_pointnet.py').write_text('# marker\n', encoding='utf-8')
    context = StudioContext(
        settings=QSettings(
            str(root / 'studio-test.ini'),
            QSettings.Format.IniFormat,
        )
    )
    assert context.set_project(root)
    return context


def _run(root: Path, name: str, f1: float, baseline: float) -> Path:
    path = root / 'runs' / name
    trial = path / 'trial_0001'
    trial.mkdir(parents=True)
    (path / 'best.json').write_text(
        json.dumps(
            {
                'target': 'ground',
                'features': 'xyz',
                'f1': f1,
                'iou': f1 - 0.05,
                'precision': f1 - 0.01,
                'recall': f1 + 0.01,
                'train': {
                    'iou': f1 - 0.08,
                    'f1': f1 - 0.03,
                    'precision': f1 - 0.04,
                    'recall': f1 - 0.02,
                    'confusion': [[80, 8], [10, 42]],
                },
                'val': {
                    'iou': f1 - 0.05,
                    'f1': f1,
                    'precision': f1 - 0.01,
                    'recall': f1 + 0.01,
                    'confusion': [[70, 9], [8, 38]],
                },
                'baseline_f1': baseline,
                'params': {'lr': 0.001, 'batch_size': 2},
                'checkpoint': 'trial_0001/best.pt',
            }
        ),
        encoding='utf-8',
    )
    (trial / 'best.pt').write_bytes(b'model')
    with (path / 'trials.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=['number', 'state', 'f1', 'precision', 'recall', 'seconds'],
        )
        writer.writeheader()
        writer.writerow(
            {
                'number': 1,
                'state': 'COMPLETE',
                'f1': f1,
                'precision': f1 - 0.01,
                'recall': f1 + 0.01,
                'seconds': 12,
            }
        )
    with (trial / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=['epoch', 'train_loss', 'val_loss', 'f1', 'train_f1', 'iou', 'train_iou'],
        )
        writer.writeheader()
        writer.writerow(
            {'epoch': 1, 'train_loss': 0.62, 'val_loss': 0.68, 'f1': f1 - 0.1,
             'train_f1': f1 - 0.12, 'iou': f1 - 0.15, 'train_iou': f1 - 0.17}
        )
        writer.writerow(
            {'epoch': 2, 'train_loss': 0.41, 'val_loss': 0.49, 'f1': f1,
             'train_f1': f1 - 0.03, 'iou': f1 - 0.05, 'train_iou': f1 - 0.08}
        )
    return path


def test_parse_live_metrics_uses_real_f1_and_baseline() -> None:
    values = parse_live_metrics(
        'noise\ntrial=2 epoch=3 F1=0.8123 precision=0.8 baseline=0.5000\n'
    )

    assert values == [
        {'trial': 2, 'epoch': 3, 'f1': 0.8123, 'baseline': 0.5}
    ]


def test_advanced_runs_reads_compares_and_exports_real_artifacts(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    context = _context(tmp_path)
    _run(tmp_path, 'ground_xyz', 0.81, 0.50)
    _run(tmp_path, 'ground_xyz_hag', 0.86, 0.50)

    page = AdvancedRunsPage(context)

    assert page.run_list.count() == 2
    assert '0.8100' in page.summary.text()
    assert np.allclose(page.chart.values, [0.62, 0.41])
    assert 'IoU: 0.7600' in page.summary.text()
    assert page.train_confusion.item(0, 0).text() == '80'
    assert page.val_confusion.item(1, 1).text() == '38'
    params = page.export_params()
    assert params == tmp_path / 'runs' / 'ground_xyz.params.json'
    assert json.loads(params.read_text(encoding='utf-8'))['lr'] == 0.001

    page.run_list.item(0).setSelected(True)
    page.run_list.item(1).setSelected(True)
    page.show_selection()
    assert page.table.columnCount() == 3
    assert 'Сравнение' in page.heading.text()
    page.deleteLater()
    app.processEvents()


def test_live_page_parses_process_log_without_starting_process(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    context = _context(tmp_path)
    process = context.processes.create('ground', 'python', ['train.py'], tmp_path)
    process.ingest_output('trial=1 epoch=1 F1=0.7000 baseline=0.5000\n')

    page = AdvancedLivePage(context)
    page.process_list.setCurrentRow(0)
    page.select_process(0)

    assert page.chart.values == [0.7]
    assert page.trials.item(0, 4).text() == '0.2000'
    assert 'Baseline: 0.5000' in page.summary.text()
    page.deleteLater()
    app.processEvents()


def test_testing_page_reads_metrics_confusion_and_prediction_cloud(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    context = _context(tmp_path)
    run = _run(tmp_path, 'ground_xyz', 0.81, 0.50)
    exports = tmp_path / 'exports_small'
    exports.mkdir()
    scene = exports / 'scene_001_cleaned.npy'
    array = np.zeros((8, 9), dtype=np.float32)
    array[:, 0] = [2, 2, 1, 1, 2, 1, 2, 1]
    array[:, 1:4] = np.arange(24, dtype=np.float32).reshape(8, 3)
    np.save(scene, array)
    result_dir = tmp_path / 'tests' / 'run-ground_xyz__exports_small'
    result_dir.mkdir(parents=True)
    np.save(result_dir / 'scene_001_cleaned__demo.npy', np.array([1, 0, 0, 1, 1, 0, 1, 255], dtype=np.uint8))
    result = {
        'model': {'target': 'ground', 'checkpoint': str(run / 'trial_0001' / 'best.pt')},
        'done': 1,
        'total': 1,
        'scenes': [
            {
                'scene': scene.name,
                'path': str(scene),
                'f1': 0.75,
                'iou': 0.60,
                'precision': 0.75,
                'recall': 0.75,
                'baseline_f1': 0.66,
                'not_covered': 1,
                'confusion': [[2, 1], [1, 3]],
            }
        ],
    }
    (result_dir / 'scenes__demo.json').write_text(
        json.dumps(result), encoding='utf-8'
    )

    page = AdvancedTestingPage(context)

    assert page.result_list.count() == 1
    assert 'F1 0.7500' in page.metrics.text()
    assert 'IoU 0.6000' in page.metrics.text()
    assert 'Все сцены: IoU 0.6000' in page.overall_metrics.text()
    assert page.confusion.item(1, 1).text() == '3'
    page.set_visual_mode('errors')
    assert page.canvas.cloud is not None
    assert set(page.canvas.cloud.classes.tolist()) >= {1, 3, 6, 7, 9}
    assert page.canvas.class_colors[6].name().upper() == '#FF304F'
    page.deleteLater()
    app.processEvents()


def test_artifacts_page_renders_markdown_and_models(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    context = _context(tmp_path)
    run = _run(tmp_path, 'ground_xyz', 0.81, 0.50)
    (run / 'report.md').write_text('# Отчёт\n\n- F1: 0.81\n', encoding='utf-8')
    (run / 'train.log').write_text('epoch 1\n', encoding='utf-8')
    (run / 'best_model.pt').write_bytes(b'final model')

    page = AdvancedArtifactsPage(context)

    assert page.list.count() == 3
    assert all('ground_xyz ·' not in page.list.item(index).text() for index in range(page.list.count()))
    document_row = next(
        index
        for index, artifact in enumerate(page.visible())
        if artifact.category == 'Документы'
    )
    page.list.setCurrentRow(document_row)
    page.select_artifact(document_row)
    assert 'Отчёт' in page.markdown.toPlainText()
    page.deleteLater()
    app.processEvents()
