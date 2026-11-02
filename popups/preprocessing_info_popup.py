from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from popups.base import BasePopup


class PreprocessingInfoPopup(BasePopup):
    """Подробная статистика предобработки одного или нескольких файлов."""

    max_size = BasePopup.max_size

    def __init__(
        self,
        files_info: list[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        self.files_info = files_info
        super().__init__(
            title="Дополнительная информация",
            subtitle="Статистика загрузки и результаты применения фильтров",
            parent=parent,
        )
        self._apply_popup_styles()

    def build_body(self, layout: QVBoxLayout) -> None:
        for index, file_info in enumerate(self.files_info):
            if index:
                separator = QFrame()
                separator.setObjectName("popupSeparator")
                separator.setFixedHeight(1)
                layout.addWidget(separator)
            self._add_file_info(layout, file_info)

        if not self.files_info:
            empty = QLabel("Нет данных о предобработке файлов.")
            empty.setObjectName("infoMuted")
            layout.addWidget(empty)

        layout.addStretch(1)

    def build_footer(self, layout: QHBoxLayout) -> None:
        layout.addStretch(1)
        close_button = QPushButton("Закрыть")
        close_button.setObjectName("secondaryButton")
        close_button.setMinimumWidth(120)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def _add_file_info(self, layout: QVBoxLayout, data: dict[str, Any]) -> None:
        file_name = QLabel(str(data.get("file_name", "Файл без названия")))
        file_name.setObjectName("infoFileName")
        layout.addWidget(file_name)

        summary = QFrame()
        summary.setObjectName("infoSummaryCard")
        summary_layout = QGridLayout(summary)
        summary_layout.setContentsMargins(16, 14, 16, 14)
        summary_layout.setHorizontalSpacing(24)
        summary_layout.setVerticalSpacing(10)

        self._add_metric(summary_layout, 0, 0, "Загружено строк", f'{data.get("rows_loaded", 0):,}'.replace(",", " "))
        unique_classes = data.get("unique_classes", [])
        self._add_metric(summary_layout, 0, 1, "Уникальные классы", "[" + "  ".join(map(str, unique_classes)) + "]")

        ranges = data.get("coordinate_range", {})
        range_text = "   ".join(
            f"{axis}[{bounds.get('min', 0):.2f}, {bounds.get('max', 0):.2f}]"
            for axis, bounds in ranges.items()
        )
        self._add_metric(summary_layout, 1, 0, "Диапазон координат", range_text or "—", span=2)

        self._add_metric(
            summary_layout, 2, 0, "Шаг облака до фильтров",
            f'{data.get("point_cloud_step_before_downsampling", 0):.3f} м',
        )
        self._add_metric(
            summary_layout, 2, 1, "Шаг облака после фильтров",
            f'{data.get("point_cloud_step_after_downsampling", 0):.3f} м',
        )
        summary_layout.setColumnStretch(1, 1)
        layout.addWidget(summary)

        filters_title = QLabel("Результаты фильтрации")
        filters_title.setObjectName("infoSectionTitle")
        layout.addWidget(filters_title)

        filters_frame = QFrame()
        filters_frame.setObjectName("infoFiltersCard")

        filters_layout = QGridLayout(filters_frame)
        filters_layout.setContentsMargins(14, 12, 14, 12)
        filters_layout.setSpacing(0)

        headers = (
            "Название фильтра",
            "Количество удалённых точек",
            "Процент удалённых точек",
            "Количество оставшихся после фильтра",
            "Время работы фильтра",
        )

        for col, title in enumerate(headers):
            label = QLabel(title)
            label.setObjectName("infoTableHeader")
            label.setWordWrap(True)
            if col > 0:
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            cell = QFrame()
            cell.setObjectName("infoCellHeader")
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(8, 8, 8, 8)
            cell_layout.addWidget(label)

            filters_layout.addWidget(cell, 0, col)

        filters = data.get("filters_result", {})
        for row_idx, (filter_name, filter_data) in enumerate(filters.items(), start=1):
            name = QLabel(str(filter_name))
            name.setObjectName("infoFilterName")

            removed = int(filter_data.get("deleted_points_count", 0))
            percent = float(filter_data.get("percent_deleted_points", 0))
            remaining = int(filter_data.get("number_of_remaining", 0))
            elapsed = float(filter_data.get("filter_worktime", 0))

            removed_label = QLabel(f"{removed:,}".replace(",", " "))
            removed_label.setObjectName("infoRemoved")
            percent_label = QLabel(f"{percent:.3f}%")
            percent_label.setObjectName("infoRemoved")
            remaining_label = QLabel(f"{remaining:,}".replace(",", " "))
            remaining_label.setObjectName("infoRemaining")
            elapsed_label = QLabel(f"{elapsed:.1f} с")
            elapsed_label.setObjectName("infoElapsed")

            values = (name, removed_label, percent_label, remaining_label, elapsed_label)

            for col, label in enumerate(values):
                label.setWordWrap(True)
                if col > 0:
                    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                else:
                    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

                cell = QFrame()
                cell.setObjectName("infoCell")
                cell_layout = QHBoxLayout(cell)
                cell_layout.setContentsMargins(8, 8, 8, 8)
                cell_layout.addWidget(label)

                filters_layout.addWidget(cell, row_idx, col)

        filters_layout.setColumnStretch(0, 1)
        filters_layout.setColumnStretch(1, 1)
        filters_layout.setColumnStretch(2, 1)
        filters_layout.setColumnStretch(3, 1)
        filters_layout.setColumnStretch(4, 1)

        filters_frame.setMinimumWidth(600)

        layout.addWidget(filters_frame)

        help_frame = QFrame()
        help_frame.setObjectName("infoHelpCard")
        help_layout = QVBoxLayout(help_frame)
        help_layout.setContentsMargins(14, 12, 14, 12)
        help_layout.setSpacing(5)

        help_title = QLabel("Справка по фильтрам")
        help_title.setObjectName("infoHelpTitle")
        help_layout.addWidget(help_title)

        descriptions = (
            (
                "HAG (Height Above Ground)",
                "фильтр по высоте точек относительно поверхности земли. "
                "Помогает выявлять точки ниже рельефа и аномально высокие выбросы.",
            ),
            (
                "column_gap",
                "Поиск вертикальных разрывов в колонках XY. "
                "Удаляет небольшие изолированные сегменты точек, отделённые от основной массы по высоте.",
            ),
            (
                "clusters",
                "Поиск небольших изолированных групп точек («островов»). "
                "Удаляет отдельные кластеры шума, не связанные с основным облаком.",
            ),
        )
        for term, description in descriptions:
            line = QLabel(f"<b>{term}</b> — {description}")
            line.setObjectName("infoHelpText")
            line.setWordWrap(True)
            help_layout.addWidget(line)

        layout.addWidget(help_frame)

    @staticmethod
    def _add_metric(
        layout: QGridLayout,
        row: int,
        column: int,
        title: str,
        value: str,
        span: int = 1,
    ) -> None:
        container = QWidget()
        metric_layout = QVBoxLayout(container)
        metric_layout.setContentsMargins(0, 0, 0, 0)
        metric_layout.setSpacing(3)

        title_label = QLabel(title)
        title_label.setObjectName("infoMetricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("infoMetricValue")
        value_label.setWordWrap(True)

        metric_layout.addWidget(title_label)
        metric_layout.addWidget(value_label)
        layout.addWidget(container, row, column, 1, span)

    def _apply_popup_styles(self) -> None:
        self.setStyleSheet(self.styleSheet() + """
            QLabel#infoFileName { color: #171717; font-size: 16px; font-weight: 700; }
            QFrame#infoSummaryCard, QFrame#infoFiltersCard {
                background: #F8FAFC; border: 1px solid #E7EBF0; border-radius: 10px;
            }
            QLabel#infoMetricTitle { color: #737B89; font-size: 11px; }
            QLabel#infoMetricValue { color: #202733; font-size: 13px; font-weight: 600; }
            QLabel#infoSectionTitle { color: #202733; font-size: 14px; font-weight: 700; margin-top: 5px; }
            QLabel#infoIndex { color: #9299A5; font-size: 12px; }
            QLabel#infoFilterName { color: #202733; font-size: 12px; font-weight: 700; }
            QLabel#infoRemoved { color: #394150; font-size: 12px; }
            QLabel#infoRemaining { color: #687282; font-size: 12px; }
            QLabel#infoElapsed { color: #687282; font-size: 12px; }
            QFrame#infoRowSeparator { background: #E9EDF2; border: none; }
            QFrame#infoHelpCard { background: #F4F6F9; border: none; border-radius: 9px; }
            QLabel#infoHelpTitle { color: #394150; font-size: 12px; font-weight: 700; }
            QLabel#infoHelpText { color: #687282; font-size: 11px; }
            QLabel#infoMuted { color: #737B89; font-size: 13px; }
            QPushButton#secondaryButton {
                background: #F2F4F7; color: #303846; border: 1px solid #E1E5EB;
                border-radius: 8px; padding: 9px 16px;
            }
            QPushButton#secondaryButton:hover { background: #E8ECF2; }
            QFrame#infoFiltersCard {
                background: #FFFFFF;
                border: 1px solid #E1E5EB;
                border-radius: 8px;
            }
            QFrame#infoCellHeader {
                background: #F8FAFC;
            }
        """)
