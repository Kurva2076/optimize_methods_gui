from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from card import apply_shadow
from hyperparams import SearchConfig
from popups.hyperparameters import HyperparameterDialog
from resources import icon, image_path
from screens.base import BaseScreen
from storage import save_search_config


class ParameterSelectionScreen(BaseScreen):
    step_title = 'Настройка нейросети'
    title = 'Настройка нейросети'
    subtitle = ''

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.search_config: SearchConfig | None

    def init_state(self) -> None:
        self.search_config = None

    def build_content(self, layout: QVBoxLayout) -> None:
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(24)
        cards_layout.setContentsMargins(0, 0, 0, 0)

        for card_data in self._get_model_cards():
            cards_layout.addWidget(self._build_model_card(**card_data), 1)

        layout.addLayout(cards_layout)
        layout.addSpacing(22)
        layout.addWidget(self._build_info_panel())
        layout.addStretch(1)

    def _get_model_cards(self) -> list[dict[str, str]]:
        return [
            {
                'title': 'Грунт и растительность',
                'description': 'Поверхность земли, почва,\nдеревья, кустарники, трава и т.д.',
                'image_name': 'soil-vegetation.png',
            },
            {
                'title': 'Строения',
                'description': 'Здания, сооружения,\nискусственные объекты.',
                'image_name': 'structures.png',
            },
            {
                'title': 'Водная поверхность',
                'description': 'Реки, озера, водоёмы.',
                'image_name': 'water-surface.png',
            },
        ]

    def _build_model_card(self, title: str, description: str, image_name: str) -> QFrame:
        card = QFrame()
        card.setObjectName('modelCard')
        card.setMinimumHeight(350)
        apply_shadow(card, blur=18, y_offset=6, alpha=35)

        root = QVBoxLayout(card)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        image_label = QLabel()
        image_label.setObjectName('modelImage')
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setContentsMargins(0, 0, 0, 0)

        pixmap = QPixmap(str(image_path(image_name)))
        if not pixmap.isNull():
            image_label.setPixmap(pixmap)
            image_label.setFixedSize(pixmap.size())
        else:
            image_label.setFixedSize(220, 220)
            image_label.setPixmap(icon(image_name).pixmap(QSize(220, 220)))
        root.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel(title)
        title_label.setObjectName('modelTitle')
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(title_label)
        root.addSpacing(4)

        desc_label = QLabel(description)
        desc_label.setObjectName('modelDescription')
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(desc_label)

        root.addStretch(1)

        button = QPushButton('Настроить модель')
        button.setObjectName('modelButton')
        button.setMinimumHeight(44)
        button.setMinimumWidth(180)
        button.setMaximumWidth(180)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setIcon(icon('gear-white.svg'))
        button.setIconSize(QSize(24, 24))
        button.setStyleSheet('padding-left: 8px; padding-right: 8px;')
        button.clicked.connect(self.open_hyperparameter_dialog)
        root.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)

        return card

    def _build_info_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName('infoPanel')
        apply_shadow(panel, blur=16, y_offset=4, alpha=30)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        info_icon = QLabel()
        info_icon.setPixmap(icon('info-hint.svg').pixmap(QSize(28, 28)))
        info_icon.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(info_icon)

        message = QLabel(
            'Для выбранного класса откроется окно настройки гиперпараметров.\n'
            'После подбора параметров можно перейти к обучению модели.'
        )
        message.setObjectName('infoText')
        message.setStyleSheet('color: #6B7280;')
        message.setWordWrap(True)
        layout.addWidget(message, 1)

        return panel

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
