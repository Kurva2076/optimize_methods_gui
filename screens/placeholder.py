from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout

from card import create_card
from screens.base import BaseScreen


class PlaceholderScreen(BaseScreen):
    """Заглушка для экранов, которые ещё не реализованы."""

    def build_content(self, layout: QVBoxLayout) -> None:
        card, card_layout = create_card(self.title, 'Этот раздел находится в разработке.')
        stub = QLabel('Скоро здесь появится функционал')
        stub.setObjectName('cardDescription')
        stub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addStretch(1)
        card_layout.addWidget(stub)
        card_layout.addStretch(1)
        layout.addWidget(card, 1)


class TrainingScreen(PlaceholderScreen):
    step_title = 'Обучение'
    title = 'Обучение модели'
    subtitle = 'Запустите обучение модели классификации.'


class AggregationScreen(PlaceholderScreen):
    step_title = 'Агрегация'
    title = 'Агрегация результатов'
    subtitle = 'Объедините результаты классификации.'


class SavingScreen(PlaceholderScreen):
    step_title = 'Сохранение'
    title = 'Сохранение результатов'
    subtitle = 'Сохраните итоговые данные.'
