import csv
import json
from pathlib import Path

import pytest

from training_results import ResultsLoadError, load_experiment_results


def test_loads_completed_run(completed_run: Path) -> None:
    results = load_experiment_results(completed_run)

    assert results.run_id == 'baseline_01'
    assert results.best_epoch == 3
    assert results.best_val_miou == pytest.approx(0.76)
    assert len(results.history) == 3
    assert len(results.class_metrics) == 2
    assert len(results.scene_metrics) == 2
    assert results.confusion_matrix == ((72, 8), (10, 50))
    assert results.num_points == 140


def test_accepts_results_subdirectory(completed_run: Path) -> None:
    results = load_experiment_results(completed_run / 'results')

    assert results.run_dir == completed_run.resolve()
    assert results.results_dir == (completed_run / 'results').resolve()


def test_falls_back_to_json_when_metrics_csv_is_absent(completed_run: Path) -> None:
    (completed_run / 'results' / 'metrics.csv').unlink()

    results = load_experiment_results(completed_run)

    assert [metric.class_name for metric in results.class_metrics] == ['Ground', 'Non-ground']
    assert results.class_metrics[0].support == 80


def test_rejects_history_without_required_column(completed_run: Path) -> None:
    history_path = completed_run / 'history.csv'
    with history_path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    fieldnames = [name for name in rows[0] if name != 'lr']
    with history_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row.pop('lr')
            writer.writerow(row)

    with pytest.raises(ResultsLoadError, match='lr'):
        load_experiment_results(completed_run)


def test_rejects_inconsistent_point_count(completed_run: Path) -> None:
    metrics_path = completed_run / 'results' / 'metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    metrics['num_points'] = 999
    metrics_path.write_text(json.dumps(metrics), encoding='utf-8')

    with pytest.raises(ResultsLoadError, match='не совпадает'):
        load_experiment_results(completed_run)
