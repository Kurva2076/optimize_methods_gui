from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from screens.training import TrainingScreen


def test_training_screen_loads_run_and_switches_graphs(completed_run: Path) -> None:
    app = QApplication.instance() or QApplication([])
    with patch('screens.training.QSettings'):
        screen = TrainingScreen()

        assert screen.load_results(completed_run, show_errors=False)
        assert screen.results is not None
        assert screen.state_stack.currentWidget() is screen.results_stack
        assert screen.graph_title.text() == 'Функция потерь (Loss)'

        expected_graphs = (
            'Среднее пересечение (mIoU)',
            'IoU по классам',
            'Скорость обучения (Learning Rate)',
            'Функция потерь (Loss)',
        )
        for expected_title in expected_graphs:
            screen._change_graph(1)
            assert screen.graph_title.text() == expected_title
        assert screen.summary_cards[0].value_label.text() == '3'

        screen._switch_tab(1)
        assert screen.results_stack.currentIndex() == 1
        assert screen.tab_group.checkedId() == 1
        assert screen.class_table.rowCount() == 3
        assert screen.scene_table.rowCount() == 2

        screen.deleteLater()
    app.processEvents()
