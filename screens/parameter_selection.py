from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QPushButton, QVBoxLayout, QWidget

from card import create_card
from popups.hyperparameters import HyperparameterDialog
from screens.base import BaseScreen
from storage import save_search_config
from hyperparams import SearchConfig


class ParameterSelectionScreen(BaseScreen):
    step_title = 'Подбор параметров'
    title = 'Подбор параметров'
    subtitle = 'Подберите параметры для обучения модели.'

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.search_config: SearchConfig | None

    def init_state(self) -> None:
        self.search_config = None

    def build_content(self, layout: QVBoxLayout) -> None:
        card, card_layout = create_card(
            'Гиперпараметры',
            'Выберите модель, алгоритм перебора и значения гиперпараметров.',
            icon_name='chart-bars.svg',
        )
        card.setMaximumWidth(560)

        self.open_button = QPushButton('Подбор гиперпараметров')
        self.open_button.setObjectName('primaryButton')
        self.open_button.setMinimumHeight(50)
        self.open_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_button.clicked.connect(self.open_hyperparameter_dialog)
        card_layout.addWidget(self.open_button)

        layout.addWidget(card, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)

    def open_hyperparameter_dialog(self) -> None:
        dialog = HyperparameterDialog(self)
        if not dialog.exec() or dialog.config is None:
            return

        self.search_config = dialog.config
        try:
            path = save_search_config(dialog.config)
        except OSError as exc:
            QMessageBox.critical(self, 'Не удалось сохранить', f'Не удалось записать параметры подбора:\n{exc}')
            return

        QMessageBox.information(
            self,
            'Параметры сохранены',
            f'Модель: {dialog.config.model.name}\n'
            f'Алгоритм: {dialog.config.algorithm.name}\n'
            f'Комбинаций: {dialog.config.combinations_count}\n\n'
            f'Сам подбор пока не подключён — параметры записаны в файл:\n{path}',
        )
