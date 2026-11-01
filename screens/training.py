from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from card import apply_shadow, create_card
from resources import icon
from screens.base import BaseScreen
from training_results import (
    ExperimentResults,
    ResultsLoadError,
    load_experiment_results,
)

BLUE = QColor('#3867F5')
ORANGE = QColor('#F59E42')
GREEN = QColor('#2FA36B')
PURPLE = QColor('#7C5CE7')
TEXT = QColor('#171717')
MUTED = QColor('#737B89')
GRID = QColor('#E5E9F0')


def _display_class_name(name: str) -> str:
    normalized = name.strip().lower().replace('_', '-').replace(' ', '-')
    translations = {
        'ground': 'Грунт',
        'surface': 'Грунт',
        'non-ground': 'Растительность',
        'vegetation': 'Растительность',
        'high-vegetation': 'Растительность',
    }
    return translations.get(normalized, name)


def _format_metric(value: float) -> str:
    return f'{value:.4f}'


def _format_points(value: int) -> str:
    return f'{value:,}'.replace(',', ' ')


@dataclass(frozen=True)
class ChartSeries:
    name: str
    values: tuple[float, ...]
    color: QColor
    highlight_index: int | None = None


@dataclass(frozen=True)
class ChartSpec:
    title: str
    description: str
    y_label: str
    epochs: tuple[int, ...]
    series: tuple[ChartSeries, ...]
    y_min: float | None = None
    y_max: float | None = None
    scientific: bool = False


class LineChartWidget(QWidget):
    """Лёгкий Qt-график без зависимости от Matplotlib."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec: ChartSpec | None = None
        self._points: list[tuple[QPointF, str, int, float]] = []
        self.setMinimumSize(560, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAccessibleName('График результатов обучения')

    def set_spec(self, spec: ChartSpec) -> None:
        self._spec = spec
        series_description = '; '.join(
            f'{series.name}: {len(series.values)} значений' for series in spec.series
        )
        self.setAccessibleDescription(f'{spec.title}. {series_description}.')
        self.update()

    @staticmethod
    def _value_text(value: float, scientific: bool = False) -> str:
        if scientific:
            return f'{value:.1e}'
        if abs(value) >= 100:
            return f'{value:.0f}'
        if abs(value) >= 1:
            return f'{value:.2f}'
        return f'{value:.3f}'

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor('#FFFFFF'))
        self._points = []

        spec = self._spec
        if spec is None or not spec.epochs or not spec.series:
            painter.setPen(MUTED)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Нет данных для графика')
            return

        plot = QRectF(66, 24, max(1, self.width() - 92), max(1, self.height() - 82))
        values = [value for series in spec.series for value in series.values]
        data_min = min(values)
        data_max = max(values)
        y_min = spec.y_min if spec.y_min is not None else min(0.0, data_min)
        y_max = spec.y_max if spec.y_max is not None else data_max
        if y_max <= y_min:
            y_max = y_min + 1.0
        elif spec.y_max is None:
            y_max += (y_max - y_min) * 0.08

        painter.setFont(QFont('Segoe UI', 9))
        for tick in range(6):
            ratio = tick / 5
            y = plot.bottom() - ratio * plot.height()
            value = y_min + ratio * (y_max - y_min)
            painter.setPen(QPen(GRID, 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(MUTED)
            label_rect = QRectF(0, y - 10, plot.left() - 8, 20)
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._value_text(value, spec.scientific),
            )

        min_epoch, max_epoch = min(spec.epochs), max(spec.epochs)
        epoch_span = max(1, max_epoch - min_epoch)
        tick_count = min(6, len(spec.epochs))
        for tick in range(tick_count):
            ratio = tick / max(1, tick_count - 1)
            epoch = round(min_epoch + ratio * epoch_span)
            x = plot.left() + ratio * plot.width()
            painter.setPen(MUTED)
            painter.drawText(
                QRectF(x - 24, plot.bottom() + 7, 48, 20),
                Qt.AlignmentFlag.AlignCenter,
                str(epoch),
            )

        painter.setPen(QPen(QColor('#AEB7C6'), 1.2))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
        painter.setPen(MUTED)
        painter.drawText(
            QRectF(plot.center().x() - 60, self.height() - 28, 120, 20),
            Qt.AlignmentFlag.AlignCenter,
            'Эпоха',
        )
        painter.save()
        painter.translate(16, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-80, -10, 160, 20), Qt.AlignmentFlag.AlignCenter, spec.y_label)
        painter.restore()

        for series in spec.series:
            path = QPainterPath()
            series_points: list[QPointF] = []
            for index, (epoch, value) in enumerate(zip(spec.epochs, series.values)):
                x = plot.left() + (epoch - min_epoch) / epoch_span * plot.width()
                y = plot.bottom() - (value - y_min) / (y_max - y_min) * plot.height()
                point = QPointF(x, y)
                series_points.append(point)
                if index == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
                self._points.append((point, series.name, epoch, value))

            painter.setPen(QPen(series.color, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(path)
            if len(series_points) <= 40:
                painter.setBrush(QColor('#FFFFFF'))
                painter.setPen(QPen(series.color, 1.8))
                for point in series_points:
                    painter.drawEllipse(point, 3.2, 3.2)
            if series.highlight_index is not None and 0 <= series.highlight_index < len(series_points):
                point = series_points[series.highlight_index]
                painter.setBrush(series.color)
                painter.setPen(QPen(QColor('#FFFFFF'), 2))
                painter.drawEllipse(point, 6, 6)

        legend_width = min(260, max(130, max(len(item.name) for item in spec.series) * 8 + 46))
        legend_height = 12 + len(spec.series) * 24
        legend = QRectF(plot.right() - legend_width - 8, plot.top() + 8, legend_width, legend_height)
        painter.setBrush(QColor(255, 255, 255, 232))
        painter.setPen(QPen(GRID, 1))
        painter.drawRoundedRect(legend, 7, 7)
        for index, series in enumerate(spec.series):
            y = legend.top() + 18 + index * 24
            painter.setPen(QPen(series.color, 2.4))
            painter.drawLine(QPointF(legend.left() + 12, y), QPointF(legend.left() + 34, y))
            painter.setPen(TEXT)
            painter.drawText(
                QRectF(legend.left() + 42, y - 9, legend.width() - 50, 18),
                Qt.AlignmentFlag.AlignVCenter,
                series.name,
            )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._points:
            return super().mouseMoveEvent(event)
        cursor = event.position()
        point, name, epoch, value = min(
            self._points,
            key=lambda entry: (entry[0].x() - cursor.x()) ** 2 + (entry[0].y() - cursor.y()) ** 2,
        )
        distance = (point.x() - cursor.x()) ** 2 + (point.y() - cursor.y()) ** 2
        self.setToolTip(
            f'{name}\nЭпоха: {epoch}\nЗначение: {value:.6g}' if distance <= 18**2 else ''
        )
        super().mouseMoveEvent(event)


class ConfusionMatrixWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._matrix = ((0, 0), (0, 0))
        self._class_names = ('Грунт', 'Растительность')
        self.setMinimumSize(580, 340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName('Матрица ошибок')

    def set_data(
        self,
        matrix: tuple[tuple[int, int], tuple[int, int]],
        class_names: tuple[str, str],
    ) -> None:
        self._matrix = matrix
        self._class_names = tuple(_display_class_name(name) for name in class_names)
        self.setAccessibleDescription(
            'Строки — истинные классы, столбцы — предсказанные. '
            f'Значения: {matrix[0][0]}, {matrix[0][1]}, {matrix[1][0]}, {matrix[1][1]}.'
        )
        self.update()

    @staticmethod
    def _cell_color(value: int, maximum: int) -> QColor:
        ratio = value / maximum if maximum else 0
        start = QColor('#F3F7FC')
        end = QColor('#123F7A')
        return QColor(
            round(start.red() + (end.red() - start.red()) * ratio),
            round(start.green() + (end.green() - start.green()) * ratio),
            round(start.blue() + (end.blue() - start.blue()) * ratio),
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor('#FFFFFF'))

        maximum = max(value for row in self._matrix for value in row)
        size = min(self.height() - 100, self.width() - 330)
        size = max(220, size)
        left = (self.width() - size) / 2
        top = 28
        cell = size / 2

        for row in range(2):
            for column in range(2):
                value = self._matrix[row][column]
                rect = QRectF(left + column * cell, top + row * cell, cell, cell)
                color = self._cell_color(value, maximum)
                painter.fillRect(rect, color)
                painter.setPen(QPen(QColor('#D4DAE4'), 1))
                painter.drawRect(rect)
                painter.setPen(QColor('#FFFFFF') if value > maximum / 2 else TEXT)
                painter.setFont(QFont('Segoe UI', 11, QFont.Weight.DemiBold))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, _format_points(value))

        painter.setFont(QFont('Segoe UI', 10))
        painter.setPen(TEXT)
        for index, name in enumerate(self._class_names):
            painter.drawText(
                QRectF(left + index * cell, top + size + 7, cell, 24),
                Qt.AlignmentFlag.AlignCenter,
                name,
            )
            painter.drawText(
                QRectF(left - 145, top + index * cell, 132, cell),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                name,
            )

        painter.setPen(MUTED)
        painter.drawText(
            QRectF(left, top + size + 35, size, 24),
            Qt.AlignmentFlag.AlignCenter,
            'Предсказанный класс',
        )
        painter.save()
        painter.translate(left - 170, top + size / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(-80, -12, 160, 24), Qt.AlignmentFlag.AlignCenter, 'Истинный класс')
        painter.restore()

        bar = QRectF(left + size + 28, top, 18, size)
        steps = 80
        for index in range(steps):
            ratio = index / max(1, steps - 1)
            segment = QRectF(
                bar.left(),
                bar.bottom() - (index + 1) * bar.height() / steps,
                bar.width(),
                bar.height() / steps + 1,
            )
            painter.fillRect(segment, self._cell_color(round(ratio * maximum), maximum))
        painter.setPen(QPen(QColor('#AEB7C6'), 1))
        painter.drawRect(bar)
        painter.setFont(QFont('Segoe UI', 9))
        painter.setPen(MUTED)
        painter.drawText(
            QRectF(bar.right() + 8, bar.top() - 8, 92, 20),
            Qt.AlignmentFlag.AlignVCenter,
            _format_points(maximum),
        )
        painter.drawText(
            QRectF(bar.right() + 8, bar.bottom() - 12, 92, 20),
            Qt.AlignmentFlag.AlignVCenter,
            '0',
        )


class MetricValueCard(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('metricValueCard')
        self.setMinimumHeight(76)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName('metricValueTitle')
        self.value_label = QLabel('—')
        self.value_label.setObjectName('metricValue')
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        apply_shadow(self, blur=12, y_offset=3, alpha=24)

    def set_data(self, title: str, value: str) -> None:
        self.title_label.setText(title)
        self.value_label.setText(value)


class TrainingScreen(BaseScreen):
    step_title = 'Обучение'
    title = 'Обучение нейросети'
    subtitle = ''

    def init_state(self) -> None:
        self.results: ExperimentResults | None = None
        self._graph_index = 0
        self._restore_attempted = False

    def build_content(self, layout: QVBoxLayout) -> None:
        self.toolbar = self._build_toolbar()
        layout.addWidget(self.toolbar)

        self.state_stack = QStackedWidget()
        self.stage_placeholder_page = self._build_stage_placeholder()
        self.empty_page = self._build_empty_page()
        self.results_stack = QStackedWidget()
        self.results_stack.addWidget(self._build_graph_page())
        self.results_stack.addWidget(self._build_metrics_page())
        self.results_stack.addWidget(self._build_confusion_page())
        self.state_stack.addWidget(self.stage_placeholder_page)
        self.state_stack.addWidget(self.empty_page)
        self.state_stack.addWidget(self.results_stack)
        layout.addWidget(self.state_stack, 1)

    def _build_stage_placeholder(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        card, card_layout = create_card(self.title, 'Этот раздел находится в разработке.')
        stub = QLabel('Скоро здесь появится функционал')
        stub.setObjectName('cardDescription')
        stub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addStretch(1)
        card_layout.addWidget(stub)
        card_layout.addStretch(1)
        layout.addWidget(card, 1)
        return page

    def show_stage_placeholder(self) -> None:
        self.toolbar.hide()
        self.state_stack.setCurrentWidget(self.stage_placeholder_page)

    def show_results(self) -> None:
        self.toolbar.show()
        target = self.results_stack if self.results is not None else self.empty_page
        self.state_stack.setCurrentWidget(target)

    def _build_toolbar(self) -> QWidget:
        toolbar = QWidget()
        toolbar.setObjectName('trainingToolbar')
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        row.addStretch(1)
        self.tab_group = QButtonGroup(self)
        self.tab_group.setExclusive(True)
        tab_titles = ('Графики', 'Метрики', 'Матрица ошибок')
        for index, title in enumerate(tab_titles):
            button = QPushButton(title)
            button.setObjectName('trainingTab')
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumSize(170 if index != 2 else 190, 42)
            button.setAccessibleName(f'Открыть раздел «{title}»')
            button.clicked.connect(lambda _checked=False, i=index: self._switch_tab(i))
            self.tab_group.addButton(button, index)
            row.addWidget(button)
            if index == 0:
                button.setChecked(True)
        row.addStretch(1)

        self.path_label = QLabel('Результаты не выбраны')
        self.path_label.setObjectName('resultsPath')
        self.path_label.setMaximumWidth(220)
        self.path_label.setToolTip('Выберите папку завершённого запуска')
        row.addWidget(self.path_label)

        choose_button = QPushButton('Выбрать результаты')
        choose_button.setObjectName('secondaryAction')
        choose_button.setIcon(icon('folder.svg'))
        choose_button.setIconSize(QSize(20, 20))
        choose_button.setMinimumHeight(42)
        choose_button.setCursor(Qt.CursorShape.PointingHandCursor)
        choose_button.clicked.connect(self.choose_results_directory)
        row.addWidget(choose_button)
        return toolbar

    def _build_empty_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 12, 0, 0)

        card = QFrame()
        card.setObjectName('resultsEmptyCard')
        card.setMinimumHeight(360)
        apply_shadow(card)
        content = QVBoxLayout(card)
        content.setContentsMargins(32, 42, 32, 42)
        content.setSpacing(12)
        content.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_label = QLabel()
        icon_label.setPixmap(icon('chart-bars.svg').pixmap(QSize(58, 58)))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel('Выберите результаты завершённого обучения')
        title.setObjectName('emptyResultsTitle')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description = QLabel(
            'Укажите папку runs/<run_id> или вложенную папку results.\n'
            'Графики и таблицы будут построены из CSV и JSON-файлов ML-команды.'
        )
        description.setObjectName('emptyResultsDescription')
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        button = QPushButton('Открыть папку результатов')
        button.setObjectName('primaryButton')
        button.setIcon(icon('folder.svg'))
        button.setIconSize(QSize(22, 22))
        button.setMinimumSize(240, 48)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(self.choose_results_directory)
        content.addStretch(1)
        content.addWidget(icon_label)
        content.addWidget(title)
        content.addWidget(description)
        content.addSpacing(6)
        content.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)
        content.addStretch(1)
        outer.addWidget(card, 1)
        return page

    def _build_graph_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 10, 0, 0)
        row.setSpacing(14)

        self.previous_graph_button = self._arrow_button('chevron-left.svg', 'Предыдущий график')
        self.previous_graph_button.clicked.connect(lambda: self._change_graph(-1))
        row.addWidget(self.previous_graph_button, alignment=Qt.AlignmentFlag.AlignVCenter)

        card = QFrame()
        card.setObjectName('trainingContentCard')
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 20, 28, 20)
        card_layout.setSpacing(10)

        self.graph_title = QLabel('График')
        self.graph_title.setObjectName('trainingSectionTitle')
        self.graph_description = QLabel()
        self.graph_description.setObjectName('trainingSectionDescription')
        self.graph_description.setWordWrap(True)
        card_layout.addWidget(self.graph_title)
        card_layout.addWidget(self.graph_description)

        body = QHBoxLayout()
        body.setSpacing(24)
        self.chart = LineChartWidget()
        body.addWidget(self.chart, 1)

        summary = QVBoxLayout()
        summary.setSpacing(12)
        self.summary_cards = [MetricValueCard('—') for _ in range(4)]
        for metric_card in self.summary_cards:
            metric_card.setMinimumWidth(210)
            summary.addWidget(metric_card)
        summary.addStretch(1)
        body.addLayout(summary)
        card_layout.addLayout(body, 1)
        row.addWidget(card, 1)

        self.next_graph_button = self._arrow_button('chevron-right.svg', 'Следующий график')
        self.next_graph_button.clicked.connect(lambda: self._change_graph(1))
        row.addWidget(self.next_graph_button, alignment=Qt.AlignmentFlag.AlignVCenter)
        return page

    @staticmethod
    def _arrow_button(icon_name: str, accessible_name: str) -> QPushButton:
        button = QPushButton()
        button.setObjectName('graphArrow')
        button.setIcon(icon(icon_name))
        button.setIconSize(QSize(24, 24))
        button.setFixedSize(52, 52)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setAccessibleName(accessible_name)
        return button

    def _build_metrics_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 10, 0, 0)

        card = QFrame()
        card.setObjectName('trainingContentCard')
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(12)
        layout.addWidget(self._section_title('Основные метрики'))
        layout.addWidget(self._section_description(
            'Рассчитаны на тестовой выборке по общей матрице ошибок.'
        ))

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(18)
        self.main_metric_cards = [
            MetricValueCard('mIoU'),
            MetricValueCard('Precision (macro)'),
            MetricValueCard('Recall (macro)'),
            MetricValueCard('F1-score (macro)'),
        ]
        for metric_card in self.main_metric_cards:
            metrics_row.addWidget(metric_card, 1)
        layout.addLayout(metrics_row)
        layout.addSpacing(6)

        tables = QHBoxLayout()
        tables.setSpacing(24)
        class_column = QVBoxLayout()
        class_column.addWidget(self._section_title('Метрики по классам'))
        self.class_table = self._create_table(
            ('Класс', 'IoU', 'Precision', 'Recall', 'F1-score'),
        )
        class_column.addWidget(self.class_table, 1)
        tables.addLayout(class_column, 3)

        scene_column = QVBoxLayout()
        self.scene_title = self._section_title('Дополнительная информация')
        scene_column.addWidget(self.scene_title)
        self.scene_summary = QLabel()
        self.scene_summary.setObjectName('sceneSummary')
        scene_column.addWidget(self.scene_summary)
        self.scene_table = self._create_table(('Сцена', 'Точек', 'mIoU', 'F1'))
        scene_column.addWidget(self.scene_table, 1)
        tables.addLayout(scene_column, 2)
        layout.addLayout(tables, 1)
        outer.addWidget(card, 1)
        return page

    def _build_confusion_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 10, 0, 0)

        card = QFrame()
        card.setObjectName('trainingContentCard')
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(4)
        layout.addWidget(self._section_title('Матрица ошибок'))
        layout.addWidget(self._section_description(
            'Показывает, как модель классифицирует точки по каждому классу.'
        ))
        self.confusion_widget = ConfusionMatrixWidget()
        layout.addWidget(self.confusion_widget, 1)
        outer.addWidget(card, 1)
        return page

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName('trainingSectionTitle')
        return label

    @staticmethod
    def _section_description(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName('trainingSectionDescription')
        label.setWordWrap(True)
        return label

    @staticmethod
    def _create_table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setObjectName('metricsTable')
        table.setHorizontalHeaderLabels(list(headers))
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setAlternatingRowColors(False)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(headers)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        return table

    def _switch_tab(self, index: int) -> None:
        button = self.tab_group.button(index)
        if button is not None:
            button.setChecked(True)
        self.results_stack.setCurrentIndex(index)

    def _change_graph(self, offset: int) -> None:
        self._graph_index = (self._graph_index + offset) % 4
        self._refresh_graph()

    def choose_results_directory(self) -> None:
        start = QSettings('PointCloud', 'PointCloudGUI').value('training_results_dir', '')
        directory = QFileDialog.getExistingDirectory(
            self,
            'Выберите папку завершённого запуска',
            str(start or Path.home()),
        )
        if directory:
            self.load_results(directory)

    def load_results(
        self,
        directory: str | Path,
        show_errors: bool = True,
        show_results: bool = True,
    ) -> bool:
        """Загрузить результаты. Метод можно вызывать из карточки обученной модели."""
        try:
            results = load_experiment_results(directory)
        except ResultsLoadError as exc:
            if show_errors:
                QMessageBox.warning(
                    self,
                    'Не удалось открыть результаты',
                    f'{exc}\n\nВыберите папку завершённого запуска или её папку results.',
                )
            return False

        self.results = results
        self._graph_index = 0
        self.path_label.setText(results.run_id)
        self.path_label.setToolTip(str(results.run_dir))
        QSettings('PointCloud', 'PointCloudGUI').setValue(
            'training_results_dir', str(results.run_dir)
        )
        self._refresh_all()
        if show_results:
            self.show_results()
        return True

    def _refresh_all(self) -> None:
        assert self.results is not None
        self._refresh_graph()
        self._refresh_metrics()
        self.confusion_widget.set_data(
            self.results.confusion_matrix,
            self.results.class_names,
        )

    def _chart_specs(self) -> tuple[tuple[ChartSpec, tuple[tuple[str, str], ...]], ...]:
        assert self.results is not None
        history = self.results.history
        epochs = tuple(row.epoch for row in history)
        best_index = max(range(len(history)), key=lambda index: history[index].val_miou)
        final = history[-1]

        loss = ChartSpec(
            'Функция потерь (Loss)',
            'Изменение функции потерь на обучающей и валидационной выборках.',
            'Loss',
            epochs,
            (
                ChartSeries('Train', tuple(row.train_loss for row in history), BLUE),
                ChartSeries('Validation', tuple(row.val_loss for row in history), ORANGE),
            ),
            y_min=0,
        )
        miou = ChartSpec(
            'Среднее пересечение (mIoU)',
            'Качество сегментации на валидационной выборке по эпохам.',
            'mIoU',
            epochs,
            (
                ChartSeries(
                    'Validation mIoU',
                    tuple(row.val_miou for row in history),
                    GREEN,
                    highlight_index=best_index,
                ),
            ),
            y_min=0,
            y_max=1,
        )
        class_iou = ChartSpec(
            'IoU по классам',
            'Изменение IoU грунта и растительности на валидационной выборке.',
            'IoU',
            epochs,
            (
                ChartSeries('Грунт', tuple(row.val_iou_ground for row in history), BLUE),
                ChartSeries(
                    'Растительность',
                    tuple(row.val_iou_non_ground for row in history),
                    ORANGE,
                ),
            ),
            y_min=0,
            y_max=1,
        )
        learning_rate = ChartSpec(
            'Скорость обучения (Learning Rate)',
            'Изменение шага оптимизатора в процессе обучения.',
            'Learning rate',
            epochs,
            (ChartSeries('Learning rate', tuple(row.lr for row in history), PURPLE),),
            y_min=0,
            scientific=True,
        )

        return (
            (
                loss,
                (
                    ('Лучшая эпоха', str(self.results.best_epoch)),
                    ('Лучший mIoU (val)', _format_metric(self.results.best_val_miou)),
                    ('Финальный loss (train)', _format_metric(final.train_loss)),
                    ('Финальный loss (val)', _format_metric(final.val_loss)),
                ),
            ),
            (
                miou,
                (
                    ('Лучшая эпоха', str(self.results.best_epoch)),
                    ('Лучший mIoU (val)', _format_metric(self.results.best_val_miou)),
                    ('Финальный mIoU (val)', _format_metric(final.val_miou)),
                    ('Всего эпох', str(len(history))),
                ),
            ),
            (
                class_iou,
                (
                    ('IoU грунта (финал)', _format_metric(final.val_iou_ground)),
                    ('IoU растительности (финал)', _format_metric(final.val_iou_non_ground)),
                    ('Лучший IoU грунта', _format_metric(max(row.val_iou_ground for row in history))),
                    (
                        'Лучший IoU растительности',
                        _format_metric(max(row.val_iou_non_ground for row in history)),
                    ),
                ),
            ),
            (
                learning_rate,
                (
                    ('Начальный learning rate', f'{history[0].lr:.2e}'),
                    ('Финальный learning rate', f'{final.lr:.2e}'),
                    ('Минимальный learning rate', f'{min(row.lr for row in history):.2e}'),
                    ('Всего эпох', str(len(history))),
                ),
            ),
        )

    def _refresh_graph(self) -> None:
        if self.results is None:
            return
        spec, stats = self._chart_specs()[self._graph_index]
        self.graph_title.setText(spec.title)
        self.graph_description.setText(spec.description)
        self.chart.set_spec(spec)
        for card, (title, value) in zip(self.summary_cards, stats):
            card.set_data(title, value)
        self.previous_graph_button.setToolTip(
            f'Предыдущий график ({(self._graph_index - 1) % 4 + 1} из 4)'
        )
        self.next_graph_button.setToolTip(
            f'Следующий график ({(self._graph_index + 1) % 4 + 1} из 4)'
        )

    def _refresh_metrics(self) -> None:
        assert self.results is not None
        values = (
            self.results.miou,
            self.results.macro_precision,
            self.results.macro_recall,
            self.results.macro_f1,
        )
        titles = ('mIoU', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)')
        for card, title, value in zip(self.main_metric_cards, titles, values):
            card.set_data(title, _format_metric(value))

        rows = list(self.results.class_metrics)
        self.class_table.setRowCount(len(rows) + 1)
        for row_index, metric in enumerate(rows):
            values = (
                _display_class_name(metric.class_name),
                _format_metric(metric.iou),
                _format_metric(metric.precision),
                _format_metric(metric.recall),
                _format_metric(metric.f1),
            )
            self._set_table_row(self.class_table, row_index, values)
        macro_values = (
            'Среднее (macro)',
            _format_metric(self.results.miou),
            _format_metric(self.results.macro_precision),
            _format_metric(self.results.macro_recall),
            _format_metric(self.results.macro_f1),
        )
        self._set_table_row(self.class_table, len(rows), macro_values, bold=True)

        self.scene_summary.setText(
            f'Всего точек: {_format_points(self.results.num_points)}   •   '
            f'Сцен: {len(self.results.scene_metrics)}   •   '
            f'Выборка: {self.results.split}'
        )
        self.scene_table.setRowCount(len(self.results.scene_metrics))
        for row_index, scene in enumerate(self.results.scene_metrics):
            values = (
                Path(scene.scene_id).stem,
                _format_points(scene.num_points),
                _format_metric(scene.miou),
                _format_metric(scene.macro_f1),
            )
            self._set_table_row(self.scene_table, row_index, values)

    @staticmethod
    def _set_table_row(
        table: QTableWidget,
        row: int,
        values: tuple[str, ...],
        bold: bool = False,
    ) -> None:
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setForeground(TEXT)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                if column == 0
                else Qt.AlignmentFlag.AlignCenter
            )
            if bold:
                font = item.font()
                font.setWeight(QFont.Weight.DemiBold)
                item.setFont(font)
                item.setBackground(QColor('#F2F5FB'))
            table.setItem(row, column, item)

    def on_activated(self) -> None:
        if self.results is not None or self._restore_attempted:
            return
        self._restore_attempted = True
        saved = QSettings('PointCloud', 'PointCloudGUI').value('training_results_dir', '')
        if saved and Path(str(saved)).is_dir():
            self.load_results(str(saved), show_errors=False, show_results=False)
