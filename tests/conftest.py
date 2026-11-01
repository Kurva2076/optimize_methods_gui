import csv
import json
from pathlib import Path

import pytest


@pytest.fixture
def completed_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / 'runs' / 'baseline_01'
    results_dir = run_dir / 'results'
    results_dir.mkdir(parents=True)

    history_fields = [
        'epoch',
        'train_loss',
        'val_loss',
        'val_miou',
        'val_iou_ground',
        'val_iou_non_ground',
        'val_precision_ground',
        'val_precision_non_ground',
        'val_recall_ground',
        'val_recall_non_ground',
        'val_f1_ground',
        'val_f1_non_ground',
        'val_macro_precision',
        'val_macro_recall',
        'val_macro_f1',
        'lr',
        'epoch_seconds',
        'elapsed_seconds',
    ]
    history_rows = [
        [1, 0.52, 0.49, 0.61, 0.70, 0.52, 0.78, 0.68, 0.86, 0.65, 0.82, 0.66, 0.73, 0.75, 0.74, 0.001, 1.2, 1.2],
        [2, 0.31, 0.35, 0.72, 0.80, 0.64, 0.86, 0.75, 0.91, 0.72, 0.88, 0.73, 0.805, 0.815, 0.805, 0.001, 1.1, 2.3],
        [3, 0.20, 0.27, 0.76, 0.84, 0.68, 0.89, 0.79, 0.93, 0.77, 0.91, 0.78, 0.84, 0.85, 0.845, 0.0005, 1.0, 3.3],
    ]
    with (run_dir / 'history.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(history_fields)
        writer.writerows(history_rows)

    (run_dir / 'config.json').write_text(
        json.dumps({'evaluation': {'class_names': ['Ground', 'Non-ground']}}),
        encoding='utf-8',
    )
    (run_dir / 'metadata.json').write_text(
        json.dumps({'status': 'complete', 'best_epoch': 3, 'best_val_miou': 0.76}),
        encoding='utf-8',
    )

    metrics = {
        'confusion_matrix': [[72, 8], [10, 50]],
        'support': [80, 60],
        'iou': [0.8, 0.735294],
        'miou': 0.767647,
        'precision': [0.878049, 0.862069],
        'recall': [0.9, 0.833333],
        'f1': [0.888889, 0.847458],
        'macro_precision': 0.870059,
        'macro_recall': 0.866667,
        'macro_f1': 0.868173,
        'num_points': 140,
        'split': 'test',
        'run_id': 'baseline_01',
        'best_epoch': 3,
        'best_val_miou': 0.76,
        'per_scene': [
            {
                'scene_id': '15 Холмы.npy',
                'num_points': 80,
                'miou': 0.78,
                'iou_ground': 0.82,
                'iou_non_ground': 0.74,
                'macro_f1': 0.87,
            },
            {
                'scene_id': '16 Равнина.npy',
                'num_points': 60,
                'miou': 0.71,
                'iou_ground': 0.76,
                'iou_non_ground': 0.66,
                'macro_f1': 0.82,
            },
        ],
    }
    (results_dir / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False),
        encoding='utf-8',
    )

    with (results_dir / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'class_id', 'class_name', 'support', 'iou', 'precision', 'recall', 'f1'])
        writer.writerow(['test', 0, 'Ground', 80, 0.8, 0.878049, 0.9, 0.888889])
        writer.writerow(['test', 1, 'Non-ground', 60, 0.735294, 0.862069, 0.833333, 0.847458])
        writer.writerow(['test', 'macro', 'macro', 140, 0.767647, 0.870059, 0.866667, 0.868173])

    with (results_dir / 'per_scene_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scene_id', 'num_points', 'miou', 'iou_ground', 'iou_non_ground', 'macro_f1'])
        writer.writerow(['15 Холмы.npy', 80, 0.78, 0.82, 0.74, 0.87])
        writer.writerow(['16 Равнина.npy', 60, 0.71, 0.76, 0.66, 0.82])

    return run_dir
