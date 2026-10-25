from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from card import apply_shadow, repolish
from hyperparams import ALGORITHMS, ALGORITHMS_BY_KEY, MODELS, MODELS_BY_KEY, ModelSpec, SearchConfig
from popups.base import BasePopup
from resources import icon, image_path


class ModelCard(QFrame):
    """Карточка модели. Клик — выбор; выбранная подсвечивается рамкой."""

    chosen = Signal(str)

    def __init__(self, model: ModelSpec) -> None:
        super().__init__()
        self._key = model.key
        self.setObjectName('modelCard')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(104)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setSpacing(6)

        title = QLabel(model.name)
        title.setObjectName('modelCardTitle')
        description = QLabel(model.description)
        description.setObjectName('modelCardDesc')
        description.setWordWrap(True)
        for label in (title, description):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch(1)

        apply_shadow(self, blur=18, y_offset=4, alpha=30)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        self.setProperty('selected', selected)
        repolish(self)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.chosen.emit(self._key)
        super().mouseReleaseEvent(event)


class HyperparameterDialog(BasePopup):
    """
    Окно «Подбор гиперпараметров»: модель → алгоритм → значения параметров.
    После accept() результат лежит в self.config (SearchConfig).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            title='Подбор гиперпараметров',
            subtitle='Выберите модель, алгоритм подбора и задайте значение параметров.',
            icon_name='chart-bars.svg',
            parent=parent,
        )

        self.config: SearchConfig | None
        self._model: ModelSpec | None
        self._cards: dict[str, ModelCard]
        self._inputs: dict[str, QLineEdit]
        self._saved_inputs: dict[str, dict[str, str]]

        self.params_scroll: QScrollArea

    def init_state(self) -> None:
        self.config = None
        self._model = None
        self._cards = {}
        self._inputs = {}
        self._saved_inputs = {}

    def build_body(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._section_title('1. Выбор модели'))
        layout.addLayout(self._build_model_cards())
        layout.addLayout(self._build_algorithm_section())
        layout.addWidget(self._section_title('3. Гиперпараметры'))

        self.params_scroll = QScrollArea()
        self.params_scroll.setObjectName('paramScroll')
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.params_scroll, 1)

        layout.addLayout(self._build_footer())

        self._select_model(MODELS[0].key)

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName('sectionTitle')
        return label

    def _build_model_cards(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(20)
        for model in MODELS:
            card = ModelCard(model)
            card.chosen.connect(self._select_model)
            self._cards[model.key] = card
            row.addWidget(card, 1)
        return row

    def _build_algorithm_section(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(24)

        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(self._section_title('2. Алгоритм подбора'))
        hint = QLabel('Выберите способ перебора гиперпараметров.')
        hint.setObjectName('sectionHint')
        left.addWidget(hint)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setObjectName('algorithmCombo')
        for algorithm in ALGORITHMS:
            self.algorithm_combo.addItem(algorithm.name, algorithm.key)
        arrow = image_path('chevron-down.svg').as_posix()
        self.algorithm_combo.setStyleSheet(
            f'QComboBox::down-arrow {{ image: url("{arrow}"); width: 16px; height: 16px; }}'
        )
        left.addWidget(self.algorithm_combo)
        row.addLayout(left, 2)

        row.addWidget(self._build_info_box(), 1, Qt.AlignmentFlag.AlignVCenter)

        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        self._on_algorithm_changed(self.algorithm_combo.currentIndex())
        return row

    def _build_info_box(self) -> QFrame:
        box = QFrame()
        box.setObjectName('infoBox')
        box.setMinimumWidth(320)
        layout = QHBoxLayout(box)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        icon_label = QLabel()
        icon_label.setPixmap(icon('info-hint.svg').pixmap(QSize(30, 30)))
        self.info_label = QLabel()
        self.info_label.setObjectName('infoText')
        self.info_label.setWordWrap(True)
        layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.info_label, 1)

        apply_shadow(box, blur=15, y_offset=3, alpha=25)
        return box

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)
        row.addStretch(1)

        cancel_button = QPushButton('Отмена')
        cancel_button.setObjectName('secondaryButton')
        cancel_button.setMinimumSize(170, 50)
        cancel_button.setAutoDefault(False)
        cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_button.clicked.connect(self.reject)

        run_button = QPushButton('Запустить подбор')
        run_button.setObjectName('primaryButton')
        run_button.setMinimumSize(280, 50)
        run_button.setIcon(icon('play.svg'))
        run_button.setIconSize(QSize(26, 26))
        run_button.setDefault(True)
        run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        run_button.clicked.connect(self._on_run)

        row.addWidget(cancel_button)
        row.addWidget(run_button)
        return row

    def _on_algorithm_changed(self, index: int) -> None:
        algorithm = ALGORITHMS_BY_KEY[self.algorithm_combo.itemData(index)]
        self.info_label.setText(algorithm.description)

    def _select_model(self, key: str) -> None:
        self._remember_inputs()
        self._model = MODELS_BY_KEY[key]
        for card_key, card in self._cards.items():
            card.set_selected(card_key == key)
        self._rebuild_params()

    def _remember_inputs(self) -> None:
        if self._model is not None:
            self._saved_inputs[self._model.key] = {
                key: edit.text() for key, edit in self._inputs.items()
            }

    def _rebuild_params(self) -> None:
        assert self._model is not None
        saved = self._saved_inputs.get(self._model.key, {})

        container = QWidget()
        container.setObjectName('paramContainer')
        grid = QGridLayout(container)
        grid.setContentsMargins(4, 4, 12, 4)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)
        grid.setColumnMinimumWidth(0, 300)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, 200)

        self._inputs = {}
        for row, spec in enumerate(self._model.params):
            label = QLabel(spec.label)
            label.setObjectName('paramLabel')

            edit = QLineEdit(saved.get(spec.key, spec.default))
            edit.setObjectName('paramInput')
            edit.textChanged.connect(lambda _text, e=edit: self._clear_invalid(e))

            hint = QLabel(f'Например: {spec.example}')
            hint.setObjectName('paramHint')

            grid.addWidget(label, row, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(edit, row, 1)
            grid.addWidget(hint, row, 2, alignment=Qt.AlignmentFlag.AlignVCenter)
            self._inputs[spec.key] = edit

        grid.setRowStretch(len(self._model.params), 1)
        self.params_scroll.setWidget(container)

    @staticmethod
    def _mark_invalid(edit: QLineEdit, message: str) -> None:
        edit.setProperty('invalid', True)
        edit.setToolTip(message)
        repolish(edit)

    @staticmethod
    def _clear_invalid(edit: QLineEdit) -> None:
        if edit.property('invalid'):
            edit.setProperty('invalid', False)
            edit.setToolTip('')
            repolish(edit)

    def _on_run(self) -> None:
        assert self._model is not None

        values: dict[str, list] = {}
        errors: list[str] = []
        first_invalid: QLineEdit | None = None

        for spec in self._model.params:
            edit = self._inputs[spec.key]
            try:
                values[spec.key] = spec.parse(edit.text())
            except ValueError as exc:
                self._mark_invalid(edit, str(exc))
                errors.append(f'• {spec.label}: {exc}')
                if first_invalid is None:
                    first_invalid = edit

        if errors:
            QMessageBox.warning(
                self,
                'Проверьте значения',
                'Исправьте значения в выделенных полях:\n\n' + '\n'.join(errors),
            )
            if first_invalid is not None:
                first_invalid.setFocus()
            return

        algorithm = ALGORITHMS_BY_KEY[self.algorithm_combo.currentData()]
        self.config = SearchConfig(self._model, algorithm, values)
        self.accept()
