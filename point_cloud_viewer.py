from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QSettings,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
    QPainter,
    QPen,
    QPolygonF,
    QResizeEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from card import apply_shadow
from point_cloud_canvas import GRADIENTS, PointCloudCanvas
from point_cloud_data import (
    SUPPORTED_POINT_CLOUD_EXTENSIONS,
    PointCloudData,
    PointCloudLoadCancelled,
    class_style_for,
    load_point_cloud,
    save_point_cloud,
    select_profile_points,
)


class ThemeToggle(QAbstractButton):
    """Компактный переключатель светлой и тёмной темы."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('viewerThemeToggle')
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip('Переключить светлую и тёмную тему')
        self.setFixedSize(72, 34)

    def sizeHint(self) -> QSize:
        return QSize(72, 34)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(1.0, 1.0, 70.0, 32.0)
        painter.setPen(QPen(QColor('#CBD5E1' if not self.isChecked() else '#334155'), 1.0))
        painter.setBrush(QColor('#EAF0F8' if not self.isChecked() else '#172033'))
        painter.drawRoundedRect(track, 16.0, 16.0)

        painter.setPen(QColor('#F59E0B' if not self.isChecked() else '#64748B'))
        painter.drawText(QRectF(7, 4, 25, 26), Qt.AlignmentFlag.AlignCenter, '☀')
        painter.setPen(QColor('#94A3B8' if not self.isChecked() else '#F8FAFC'))
        painter.drawText(QRectF(40, 4, 25, 26), Qt.AlignmentFlag.AlignCenter, '☾')

        knob_x = 3.0 if not self.isChecked() else 39.0
        painter.setPen(QPen(QColor(255, 255, 255, 150), 1.0))
        painter.setBrush(QColor('#FFFFFF' if not self.isChecked() else '#3867F5'))
        painter.drawEllipse(QRectF(knob_x, 3.0, 28.0, 28.0))
        painter.setPen(QColor('#F59E0B' if not self.isChecked() else '#FFFFFF'))
        painter.drawText(
            QRectF(knob_x, 3.0, 28.0, 28.0),
            Qt.AlignmentFlag.AlignCenter,
            '☀' if not self.isChecked() else '☾',
        )
        painter.end()


class ViewerSection(QFrame):
    """Сворачиваемая секция панели управления просмотрщиком."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.setObjectName('viewerSection')

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.header = QPushButton()
        self.header.setObjectName('viewerSectionHeader')
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.toggled.connect(self.set_expanded)
        outer_layout.addWidget(self.header)

        self.content = QWidget()
        self.content.setObjectName('viewerSectionContent')
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(16, 0, 16, 16)
        self.content_layout.setSpacing(9)
        outer_layout.addWidget(self.content)

        self.set_expanded(True)

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(expanded)
        self.header.setText(f"{'▾' if expanded else '▸'}  {self.title}")
        self.content.setVisible(expanded)
        state = 'развёрнуто' if expanded else 'свёрнуто'
        self.header.setAccessibleName(f'{self.title}: {state}')

    def is_expanded(self) -> bool:
        return self.header.isChecked()


class PointCloudLoadWorker(QObject):
    progress = Signal(int)
    loaded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, path: Path, max_points: int | None = None) -> None:
        super().__init__()
        self.path = path
        self.max_points = max_points

    @Slot()
    def run(self) -> None:
        thread = QThread.currentThread()
        try:
            cloud = load_point_cloud(
                self.path,
                max_points=self.max_points,
                progress=self.progress.emit,
                cancelled=thread.isInterruptionRequested,
            )
        except PointCloudLoadCancelled:
            self.cancelled.emit()
        # Worker boundary: every loader/backend error must become a Qt signal.
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))
        else:
            self.loaded.emit(cloud)


class ProfilePlot(QWidget):
    def __init__(
        self,
        distance: np.ndarray,
        heights: np.ndarray,
        classes: np.ndarray,
        colors: dict[int, QColor],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.distance = distance
        self.heights = heights
        self.classes = classes
        self.colors = colors
        self.setMinimumSize(760, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = self.window().property('viewerTheme') == 'dark'
        background = QColor('#111827' if dark else '#FFFFFF')
        grid_color = QColor('#263247' if dark else '#DCE3ED')
        axis_color = QColor('#94A3B8' if dark else '#64748B')
        text_color = QColor('#CBD5E1' if dark else '#475569')
        painter.fillRect(self.rect(), background)

        plot = self.rect().adjusted(74, 28, -28, -62)
        painter.setPen(QPen(grid_color, 1.0))
        for division in range(6):
            y = plot.top() + division * plot.height() / 5
            painter.drawLine(plot.left(), round(y), plot.right(), round(y))

        painter.setPen(QPen(axis_color, 1.2))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())

        if not len(self.distance):
            painter.setPen(axis_color)
            painter.drawText(
                plot,
                Qt.AlignmentFlag.AlignCenter,
                'В выбранном коридоре нет точек',
            )
            painter.end()
            return

        min_x, max_x = float(self.distance.min()), float(self.distance.max())
        min_z, max_z = float(self.heights.min()), float(self.heights.max())
        span_x = max(max_x - min_x, 1e-9)
        span_z = max(max_z - min_z, 1e-9)
        screen_x = plot.left() + (self.distance - min_x) / span_x * plot.width()
        screen_y = plot.bottom() - (self.heights - min_z) / span_z * plot.height()

        for class_id in np.unique(self.classes):
            mask = self.classes == class_id
            color = self.colors.get(
                int(class_id),
                QColor(class_style_for(int(class_id)).hex_color),
            )
            color.setAlpha(155)
            painter.setPen(QPen(color, 2.0))
            polygon = QPolygonF(
                [
                    QPointF(float(x), float(y))
                    for x, y in zip(screen_x[mask], screen_y[mask], strict=True)
                ]
            )
            painter.drawPoints(polygon)

        painter.setPen(text_color)
        painter.drawText(
            plot.left(),
            plot.bottom() + 28,
            '0 м',
        )
        painter.drawText(
            plot.right() - 76,
            plot.bottom() + 28,
            f'{max_x:.2f} м',
        )
        painter.drawText(
            8,
            plot.top() + 5,
            62,
            24,
            Qt.AlignmentFlag.AlignRight,
            f'{max_z:.2f}',
        )
        painter.drawText(
            8,
            plot.bottom() - 12,
            62,
            24,
            Qt.AlignmentFlag.AlignRight,
            f'{min_z:.2f}',
        )
        painter.drawText(
            plot.left(),
            self.height() - 32,
            plot.width(),
            24,
            Qt.AlignmentFlag.AlignCenter,
            'Расстояние вдоль профиля, м',
        )
        painter.save()
        painter.translate(20, plot.center().y())
        painter.rotate(-90)
        painter.drawText(-90, 0, 180, 24, Qt.AlignmentFlag.AlignCenter, 'Высота Z, м')
        painter.restore()
        painter.end()


class ProfileDialog(QDialog):
    def __init__(
        self,
        cloud: PointCloudData,
        start: np.ndarray,
        end: np.ndarray,
        width: float,
        colors: dict[int, QColor],
        source_canvas: PointCloudCanvas,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName('pointCloudViewer')
        self.setProperty(
            'viewerTheme',
            parent.property('viewerTheme') if parent is not None else 'light',
        )
        self.setWindowTitle('Профиль облака точек')
        self.resize(1120, 720)
        self.setMinimumSize(880, 560)

        distance, heights, classes = select_profile_points(
            cloud,
            start,
            end,
            width,
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 20)
        root.setSpacing(14)

        title = QLabel('Профиль облака точек')
        title.setObjectName('viewerDialogTitle')
        subtitle = QLabel(
            f'Коридор шириной {width:.2f} м · точек в профиле: {len(distance):,}'.replace(',', ' ')
        )
        subtitle.setObjectName('viewerDialogSubtitle')
        root.addWidget(title)
        root.addWidget(subtitle)

        tabs = QTabWidget()
        tabs.setObjectName('viewerProfileTabs')

        interactive_card = QFrame()
        interactive_card.setObjectName('viewerSection')
        interactive_layout = QVBoxLayout(interactive_card)
        interactive_layout.setContentsMargins(8, 8, 8, 8)
        self.profile_canvas = PointCloudCanvas()
        self.profile_canvas.setMinimumSize(720, 430)
        self.profile_canvas.set_cloud(cloud)
        self.profile_canvas.visible_classes = set(source_canvas.visible_classes)
        self.profile_canvas.class_colors = {
            class_id: QColor(color)
            for class_id, color in source_canvas.class_colors.items()
        }
        self.profile_canvas.class_sizes = dict(source_canvas.class_sizes)
        self.profile_canvas.class_multipliers = dict(
            source_canvas.class_multipliers
        )
        self.profile_canvas.set_color_mode(source_canvas.color_mode)
        self.profile_canvas.set_gradient(source_canvas.gradient_name)
        self.profile_canvas.set_point_size(source_canvas.point_size)
        self.profile_canvas.set_round_points(source_canvas.round_points)
        self.profile_canvas.set_size_attenuation(source_canvas.size_attenuation)
        self.profile_canvas.set_field_of_view(source_canvas.field_of_view)
        self.profile_canvas.set_grid_visible(source_canvas.show_grid)
        self.profile_canvas.set_gradient_background(
            source_canvas.gradient_background
        )
        self.profile_canvas.set_profile_width(width)
        self.profile_canvas.set_profile_clip_mode('inside')
        self.profile_canvas.set_profile_clipping_enabled(True)
        self.profile_canvas.set_profile_helpers_visible(True)
        self.profile_canvas.set_profile_segment(start, end)
        self.profile_canvas.fit_profile_view(start, end)
        interactive_layout.addWidget(self.profile_canvas)
        tabs.addTab(interactive_card, 'Интерактивный профиль')

        plot_card = QFrame()
        plot_card.setObjectName('viewerSection')
        plot_layout = QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(8, 8, 8, 8)
        plot_layout.addWidget(ProfilePlot(distance, heights, classes, colors))
        tabs.addTab(plot_card, 'Развёртка 2D')
        root.addWidget(tabs, 1)

        close_button = QPushButton('Закрыть')
        close_button.setObjectName('viewerPrimaryButton')
        close_button.clicked.connect(self.close)
        root.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)


class PointCloudViewerDialog(QDialog):
    """Полноценный 3D-просмотрщик в стилистике основного приложения."""

    def __init__(
        self,
        initial_path: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName('pointCloudViewer')
        self.setProperty('viewerTheme', 'light')
        self.setWindowTitle('PointCloud — 3D-просмотр облака точек')
        self.resize(1440, 860)
        self.setMinimumSize(1120, 680)
        self.setAcceptDrops(True)

        self.cloud: PointCloudData | None = None
        self._thread: QThread | None = None
        self._worker: PointCloudLoadWorker | None = None
        self._close_when_finished = False
        self._profile_windows: set[ProfileDialog] = set()
        self._immersive_mode = False
        self._immersive_menu_visible = True
        self._restore_maximized = False
        self._normal_section_states: list[bool] = []
        self._class_rows: dict[
            int,
            tuple[QCheckBox, QPushButton, QDoubleSpinBox, QDoubleSpinBox],
        ] = {}
        self.settings = QSettings('PointCloud', 'PointCloudViewer')

        self._build_ui()
        self._restore_settings()
        if initial_path is not None:
            QTimer.singleShot(0, lambda: self.load_file(Path(initial_path)))

    def _build_ui(self) -> None:
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(20, 18, 20, 20)
        self.root_layout.setSpacing(14)
        self.header = self._build_header()
        self.root_layout.addWidget(self.header)

        self.content_layout = QHBoxLayout()
        self.content_layout.setSpacing(14)
        self.canvas_card = self._build_canvas_card()
        self.content_layout.addWidget(self.canvas_card, 1)
        self.controls_panel = self._build_control_panel()
        self.content_layout.addWidget(self.controls_panel)
        self.root_layout.addLayout(self.content_layout, 1)

        self.fullscreen_shortcut = QShortcut(QKeySequence('F11'), self)
        self.fullscreen_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.fullscreen_shortcut.activated.connect(self.toggle_immersive_mode)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName('viewerHeader')
        apply_shadow(header, blur=18, y_offset=4, alpha=28)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 14, 16, 14)
        layout.setSpacing(12)

        texts = QVBoxLayout()
        texts.setSpacing(2)
        title = QLabel('3D-просмотр облака точек')
        title.setObjectName('viewerTitle')
        subtitle = QLabel(
            'Классификация, измерения, профильные срезы и проверка результатов предобработки'
        )
        subtitle.setObjectName('viewerSubtitle')
        texts.addWidget(title)
        texts.addWidget(subtitle)
        layout.addLayout(texts, 1)

        self.open_button = QPushButton('Файл')
        self.open_button.setObjectName('viewerPrimaryButton')
        file_menu = QMenu(self.open_button)
        file_menu.setObjectName('viewerFileMenu')
        open_action = file_menu.addAction('Открыть файл…')
        open_action.triggered.connect(self.select_file)
        self.last_file_action = file_menu.addAction('Открыть последний файл')
        self.last_file_action.triggered.connect(self.open_last_file)
        last_path = self.settings.value('last_file', '', type=str)
        self.last_file_action.setEnabled(bool(last_path and Path(last_path).is_file()))
        file_menu.addSeparator()

        convert_menu = file_menu.addMenu('Конвертировать')
        self.convert_actions: list[QAction] = []
        for title_text, extension in (
            ('В XYZ…', '.xyz'),
            ('В TXT…', '.txt'),
            ('В LAS…', '.las'),
            ('В LAZ…', '.laz'),
        ):
            action = convert_menu.addAction(title_text)
            action.setEnabled(False)
            action.triggered.connect(
                lambda _checked=False, suffix=extension: self.export_cloud(suffix)
            )
            self.convert_actions.append(action)

        self.screenshot_action = file_menu.addAction('Сделать снимок…')
        self.screenshot_action.setEnabled(False)
        self.screenshot_action.triggered.connect(self.save_screenshot)
        self.open_button.setMenu(file_menu)
        layout.addWidget(self.open_button)

        self.theme_toggle = ThemeToggle()
        self.theme_toggle.toggled.connect(self._apply_theme)
        layout.addWidget(self.theme_toggle)

        self.immersive_button = QPushButton('На весь экран')
        self.immersive_button.setObjectName('viewerSecondaryButton')
        self.immersive_button.setToolTip('Полноэкранный режим (F11)')
        self.immersive_button.clicked.connect(self.toggle_immersive_mode)
        layout.addWidget(self.immersive_button)

        close_button = QPushButton('Закрыть')
        close_button.setObjectName('viewerSecondaryButton')
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        return header

    def _build_canvas_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName('viewerCanvasCard')
        apply_shadow(card, blur=20, y_offset=4, alpha=30)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 10)
        layout.setSpacing(7)

        self.canvas = PointCloudCanvas()
        self.canvas.measurement_changed.connect(self._set_tool_status)
        self.canvas.profile_ready.connect(self._show_profile)
        self.canvas.camera_changed.connect(self._set_camera_status)
        self.canvas.render_error.connect(self._show_render_error)
        layout.addWidget(self.canvas, 1)

        status = QHBoxLayout()
        status.setContentsMargins(8, 0, 8, 0)
        self.file_status = QLabel('Файл не выбран')
        self.file_status.setObjectName('viewerCanvasStatus')
        self.camera_status = QLabel('Камера: —')
        self.camera_status.setObjectName('viewerCanvasStatus')
        self.tool_status = QLabel('ЛКМ — вращение · ПКМ — перемещение · колесо — масштаб')
        self.tool_status.setObjectName('viewerCanvasStatusAccent')
        status.addWidget(self.file_status)
        status.addStretch(1)
        status.addWidget(self.tool_status)
        status.addStretch(1)
        status.addWidget(self.camera_status)
        layout.addLayout(status)

        self.loading_bar = QProgressBar()
        self.loading_bar.setObjectName('viewerLoadingBar')
        self.loading_bar.setRange(0, 100)
        self.loading_bar.hide()
        layout.addWidget(self.loading_bar)

        self.immersive_menu_button = QPushButton('☰', card)
        self.immersive_menu_button.setObjectName('viewerFloatingMenuButton')
        self.immersive_menu_button.setFixedSize(48, 44)
        self.immersive_menu_button.setToolTip('Показать меню')
        self.immersive_menu_button.clicked.connect(
            lambda: self._set_immersive_menu_visible(True)
        )
        self.immersive_menu_button.hide()
        return card

    def _build_control_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName('viewerControlsScroll')
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(330)

        container = QWidget()
        container.setObjectName('viewerControls')
        self.controls_layout = QVBoxLayout(container)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(12)

        self.immersive_menu_actions = QFrame()
        self.immersive_menu_actions.setObjectName('viewerImmersiveActions')
        immersive_actions_layout = QHBoxLayout(self.immersive_menu_actions)
        immersive_actions_layout.setContentsMargins(0, 0, 0, 0)
        immersive_actions_layout.setSpacing(8)

        hide_menu_button = QPushButton('Скрыть меню')
        hide_menu_button.setObjectName('viewerSecondaryButton')
        hide_menu_button.clicked.connect(
            lambda: self._set_immersive_menu_visible(False)
        )
        immersive_actions_layout.addWidget(hide_menu_button, 1)

        exit_immersive_button = QPushButton('Обычный вид')
        exit_immersive_button.setObjectName('viewerSecondaryButton')
        exit_immersive_button.clicked.connect(self.toggle_immersive_mode)
        immersive_actions_layout.addWidget(exit_immersive_button, 1)
        self.immersive_menu_actions.hide()
        self.controls_layout.addWidget(self.immersive_menu_actions)

        self._control_sections: list[ViewerSection] = []

        self.controls_layout.addWidget(self._build_file_section())
        self.controls_layout.addWidget(self._build_appearance_section())
        self.controls_layout.addWidget(self._build_camera_section())
        self.controls_layout.addWidget(self._build_tools_section())
        self.controls_layout.addWidget(self._build_clipping_section())
        self.controls_layout.addWidget(self._build_classes_section())
        self.controls_layout.addWidget(self._build_settings_section())
        self.controls_layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _section(self, title: str) -> tuple[ViewerSection, QVBoxLayout]:
        section = ViewerSection(title)
        self._control_sections.append(section)
        return section, section.content_layout

    @Slot()
    def toggle_immersive_mode(self) -> None:
        self.set_immersive_mode(not self._immersive_mode)

    def set_immersive_mode(self, enabled: bool) -> None:
        """Переключает обычную компоновку и полноэкранную сцену с меню поверх неё."""
        if enabled == self._immersive_mode:
            return

        self._immersive_mode = enabled
        self.setProperty('immersive', enabled)
        self.canvas_card.setProperty('immersive', enabled)

        if enabled:
            self._restore_maximized = self.isMaximized()
            self._normal_section_states = [
                section.is_expanded() for section in self._control_sections
            ]
            for section in self._control_sections:
                section.set_expanded(False)

            self.header.hide()
            self.root_layout.setContentsMargins(0, 0, 0, 0)
            self.root_layout.setSpacing(0)
            self.content_layout.setSpacing(0)
            self.canvas_card.layout().setContentsMargins(0, 0, 0, 6)

            self.content_layout.removeWidget(self.controls_panel)
            self.controls_panel.setParent(self.canvas_card)
            self.controls_panel.setFixedWidth(340)
            self.immersive_menu_actions.show()
            self.immersive_button.setText('Обычный вид')
            self.showFullScreen()
            self._set_immersive_menu_visible(True)
        else:
            self.controls_panel.hide()
            self.controls_panel.setParent(self)
            self.controls_panel.setFixedWidth(330)
            self.content_layout.addWidget(self.controls_panel)
            self.controls_panel.show()

            for index, section in enumerate(self._control_sections):
                expanded = (
                    self._normal_section_states[index]
                    if index < len(self._normal_section_states)
                    else True
                )
                section.set_expanded(expanded)

            self.immersive_menu_actions.hide()
            self.immersive_menu_button.hide()
            self.header.show()
            self.root_layout.setContentsMargins(20, 18, 20, 20)
            self.root_layout.setSpacing(14)
            self.content_layout.setSpacing(14)
            self.canvas_card.layout().setContentsMargins(8, 8, 8, 10)
            self.immersive_button.setText('На весь экран')
            if self._restore_maximized:
                self.showMaximized()
            else:
                self.showNormal()

        for widget in (self, self.canvas_card, self.controls_panel):
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        QTimer.singleShot(0, self._position_immersive_menu)

    def _set_immersive_menu_visible(self, visible: bool) -> None:
        if not self._immersive_mode:
            return
        self._immersive_menu_visible = visible
        self.controls_panel.setVisible(visible)
        self.immersive_menu_button.setVisible(not visible)
        self._position_immersive_menu()

    def _position_immersive_menu(self) -> None:
        if not self._immersive_mode:
            return
        margin = 16
        if self._immersive_menu_visible:
            width = min(340, max(260, self.canvas_card.width() - margin * 2))
            height = max(100, self.canvas_card.height() - margin * 2)
            self.controls_panel.setGeometry(
                self.canvas_card.width() - width - margin,
                margin,
                width,
                height,
            )
            self.controls_panel.raise_()
        else:
            self.immersive_menu_button.move(
                self.canvas_card.width() - self.immersive_menu_button.width() - margin,
                margin,
            )
            self.immersive_menu_button.raise_()

    def _build_file_section(self) -> QFrame:
        section, layout = self._section('Данные')
        self.filename_label = QLabel('Файл не выбран')
        self.filename_label.setObjectName('viewerFileName')
        self.filename_label.setWordWrap(True)
        self.points_label = QLabel('—')
        self.points_label.setObjectName('viewerMeta')
        self.range_label = QLabel('Перетащите XYZ, TXT, PTS, LAS или LAZ в окно')
        self.range_label.setObjectName('viewerMeta')
        self.range_label.setWordWrap(True)
        self.load_all_points = QCheckBox('Загружать все точки без ограничения')
        self.load_all_points.setChecked(True)
        self.load_all_points.setToolTip(
            'Для очень больших облаков потребуется больше памяти и времени. '
            'Снимите флажок, чтобы ограничить отображение 1 500 000 точек.'
        )
        layout.addWidget(self.filename_label)
        layout.addWidget(self.points_label)
        layout.addWidget(self.range_label)
        layout.addWidget(self.load_all_points)
        return section

    def _build_appearance_section(self) -> QFrame:
        section, layout = self._section('Отображение')

        layout.addWidget(self._control_label('Цветовая схема'))
        self.color_mode = QComboBox()
        self.color_mode.addItem('Классификация', 'classification')
        self.color_mode.addItem('Градиент по высоте', 'height')
        self.color_mode.addItem('Цвета RGB из файла', 'rgb')
        self.color_mode.currentIndexChanged.connect(self._change_color_mode)
        layout.addWidget(self.color_mode)

        layout.addWidget(self._control_label('Градиент высоты'))
        self.gradient_combo = QComboBox()
        self.gradient_combo.addItems(GRADIENTS)
        self.gradient_combo.setCurrentText('RAINBOW')
        self.gradient_combo.currentTextChanged.connect(self.canvas.set_gradient)
        layout.addWidget(self.gradient_combo)

        point_row = QHBoxLayout()
        point_row.addWidget(self._control_label('Размер точек'))
        self.point_size_value = QLabel('1.0')
        self.point_size_value.setObjectName('viewerValue')
        point_row.addStretch(1)
        point_row.addWidget(self.point_size_value)
        layout.addLayout(point_row)
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(10, 100)
        self.point_size_slider.setValue(10)
        self.point_size_slider.valueChanged.connect(self._change_point_size)
        layout.addWidget(self.point_size_slider)

        self.round_points = QCheckBox('Круглые точки')
        self.round_points.setChecked(True)
        self.round_points.toggled.connect(self.canvas.set_round_points)
        layout.addWidget(self.round_points)

        self.edge_detection = QCheckBox('Обнаружение краёв')
        self.edge_detection.toggled.connect(self.canvas.set_edge_detection)
        layout.addWidget(self.edge_detection)

        self.size_attenuation = QCheckBox('Перспективное ослабление')
        self.size_attenuation.setChecked(True)
        self.size_attenuation.toggled.connect(self.canvas.set_size_attenuation)
        layout.addWidget(self.size_attenuation)

        self.grid_visible = QCheckBox('Показывать координатную сетку')
        self.grid_visible.setChecked(True)
        self.grid_visible.toggled.connect(self.canvas.set_grid_visible)
        layout.addWidget(self.grid_visible)

        self.gradient_background = QCheckBox('Градиентный фон сцены')
        self.gradient_background.toggled.connect(
            self.canvas.set_gradient_background
        )
        layout.addWidget(self.gradient_background)
        return section

    def _build_camera_section(self) -> QFrame:
        section, layout = self._section('Камера')

        layout.addWidget(self._control_label('Режим навигации'))
        self.navigation_combo = QComboBox()
        self.navigation_combo.addItem('Орбитальная камера', 'orbit')
        self.navigation_combo.addItem('Свободный полёт', 'flight')
        self.navigation_combo.currentIndexChanged.connect(
            self._change_navigation_mode
        )
        layout.addWidget(self.navigation_combo)

        speed_row = QHBoxLayout()
        speed_row.addWidget(self._control_label('Скорость полёта'))
        self.flight_speed = QDoubleSpinBox()
        self.flight_speed.setDecimals(1)
        self.flight_speed.setRange(0.1, 100_000.0)
        self.flight_speed.setSingleStep(1.0)
        self.flight_speed.setSuffix(' м/с')
        self.flight_speed.setValue(10.0)
        self.flight_speed.setEnabled(False)
        self.flight_speed.valueChanged.connect(self.canvas.set_flight_speed)
        speed_row.addWidget(self.flight_speed, 1)
        layout.addLayout(speed_row)

        fov_row = QHBoxLayout()
        fov_row.addWidget(self._control_label('Угол обзора'))
        self.field_of_view = QDoubleSpinBox()
        self.field_of_view.setRange(20.0, 120.0)
        self.field_of_view.setSingleStep(5.0)
        self.field_of_view.setDecimals(0)
        self.field_of_view.setSuffix('°')
        self.field_of_view.setValue(75.0)
        self.field_of_view.setToolTip(
            'Угол обзора перспективной камеры. В оригинальном SmartLaser — 75°.'
        )
        self.field_of_view.valueChanged.connect(self.canvas.set_field_of_view)
        fov_row.addWidget(self.field_of_view, 1)
        layout.addLayout(fov_row)

        self.projection_combo = QComboBox()
        self.projection_combo.addItem('Перспектива', True)
        self.projection_combo.addItem('Ортографическая', False)
        self.projection_combo.currentIndexChanged.connect(
            lambda index: self.canvas.set_projection(
                bool(self.projection_combo.itemData(index))
            )
        )
        layout.addWidget(self.projection_combo)

        grid = QGridLayout()
        grid.setSpacing(6)
        views = (
            ('Сверху', 'top', 0, 0),
            ('Снизу', 'bottom', 0, 1),
            ('Спереди', 'front', 1, 0),
            ('Сзади', 'back', 1, 1),
            ('Слева', 'left', 2, 0),
            ('Справа', 'right', 2, 1),
        )
        for text, view, row, column in views:
            button = QPushButton(text)
            button.setObjectName('viewerMiniButton')
            button.clicked.connect(lambda _checked=False, value=view: self.canvas.set_view(value))
            grid.addWidget(button, row, column)
        layout.addLayout(grid)

        self.camera_hint = QLabel(
            'Стрелки — сдвиг · W/S — масштаб · Home — сброс'
        )
        self.camera_hint.setObjectName('viewerMeta')
        self.camera_hint.setWordWrap(True)
        layout.addWidget(self.camera_hint)
        reset_button = QPushButton('Сбросить камеру')
        reset_button.setObjectName('viewerSecondaryButton')
        reset_button.clicked.connect(self.canvas.reset_camera)
        layout.addWidget(reset_button)
        return section

    def _build_tools_section(self) -> QFrame:
        section, layout = self._section('Измерения')
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        tools = (
            ('Камера', 'camera'),
            ('Точка', 'info'),
            ('Рулетка', 'ruler'),
            ('Профиль', 'profile'),
        )
        grid = QGridLayout()
        grid.setSpacing(6)
        for index, (text, mode) in enumerate(tools):
            button = QPushButton(text)
            button.setObjectName('viewerToolButton')
            button.setCheckable(True)
            button.setProperty('mode', mode)
            if mode == 'camera':
                button.setChecked(True)
            button.clicked.connect(
                lambda _checked=False, value=mode: self.canvas.set_tool_mode(value)
            )
            self.tool_group.addButton(button)
            grid.addWidget(button, index // 2, index % 2)
        layout.addLayout(grid)

        tool_hint = QLabel(
            'Ctrl — маркер под курсором · Shift+ЛКМ — новая линия рулетки'
        )
        tool_hint.setObjectName('viewerMeta')
        tool_hint.setWordWrap(True)
        layout.addWidget(tool_hint)

        width_row = QHBoxLayout()
        width_row.addWidget(self._control_label('Ширина профиля'))
        self.profile_width = QDoubleSpinBox()
        self.profile_width.setDecimals(2)
        self.profile_width.setRange(0.01, 100_000.0)
        self.profile_width.setValue(2.0)
        self.profile_width.setSuffix(' м')
        self.profile_width.valueChanged.connect(self.canvas.set_profile_width)
        width_row.addWidget(self.profile_width, 1)
        layout.addLayout(width_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self._control_label('Срез профиля'))
        self.profile_clip_mode = QComboBox()
        self.profile_clip_mode.addItem('Оставить внутри', 'inside')
        self.profile_clip_mode.addItem('Исключить внутри', 'outside')
        self.profile_clip_mode.addItem('Выключен', 'none')
        self.profile_clip_mode.currentIndexChanged.connect(
            lambda index: self.canvas.set_profile_clip_mode(
                str(self.profile_clip_mode.itemData(index))
            )
        )
        self.canvas.set_profile_clip_mode('inside')
        mode_row.addWidget(self.profile_clip_mode, 1)
        layout.addLayout(mode_row)

        self.profile_main_clipping = QCheckBox(
            'Применять профильный срез в основном окне'
        )
        self.profile_main_clipping.toggled.connect(
            self.canvas.set_profile_clipping_enabled
        )
        layout.addWidget(self.profile_main_clipping)

        self.profile_helpers = QCheckBox('Показывать границы профиля')
        self.profile_helpers.toggled.connect(
            self.canvas.set_profile_helpers_visible
        )
        layout.addWidget(self.profile_helpers)

        color_row = QHBoxLayout()
        color_row.addWidget(self._control_label('Маркер'))
        self.marker_color_button = QPushButton()
        self.marker_color_button.setObjectName('viewerColorButton')
        self.marker_color_button.setFixedSize(28, 28)
        self.marker_color_button.clicked.connect(
            lambda: self._select_measurement_color('marker')
        )
        color_row.addWidget(self.marker_color_button)
        color_row.addSpacing(8)
        color_row.addWidget(self._control_label('Линия'))
        self.line_color_button = QPushButton()
        self.line_color_button.setObjectName('viewerColorButton')
        self.line_color_button.setFixedSize(28, 28)
        self.line_color_button.clicked.connect(
            lambda: self._select_measurement_color('line')
        )
        color_row.addWidget(self.line_color_button)
        color_row.addStretch(1)
        layout.addLayout(color_row)
        self._set_color_button(self.marker_color_button, self.canvas.marker_color)
        self._set_color_button(self.line_color_button, self.canvas.line_color)

        clear_button = QPushButton('Очистить измерения')
        clear_button.setObjectName('viewerSecondaryButton')
        clear_button.clicked.connect(self.canvas.clear_tools)
        layout.addWidget(clear_button)
        return section

    def _build_clipping_section(self) -> QFrame:
        section, layout = self._section('Срез по высоте')
        self.clip_enabled = QCheckBox('Включить срез')
        self.clip_enabled.toggled.connect(self._apply_clip)
        layout.addWidget(self.clip_enabled)

        min_row = QHBoxLayout()
        min_row.addWidget(self._control_label('Минимум Z'))
        self.clip_minimum = QDoubleSpinBox()
        self.clip_minimum.setDecimals(3)
        self.clip_minimum.valueChanged.connect(self._apply_clip)
        min_row.addWidget(self.clip_minimum, 1)
        layout.addLayout(min_row)

        max_row = QHBoxLayout()
        max_row.addWidget(self._control_label('Максимум Z'))
        self.clip_maximum = QDoubleSpinBox()
        self.clip_maximum.setDecimals(3)
        self.clip_maximum.valueChanged.connect(self._apply_clip)
        max_row.addWidget(self.clip_maximum, 1)
        layout.addLayout(max_row)
        return section

    def _build_classes_section(self) -> QFrame:
        section, layout = self._section('Классы')
        self.classes_container = QWidget()
        self.classes_layout = QVBoxLayout(self.classes_container)
        self.classes_layout.setContentsMargins(0, 0, 0, 0)
        self.classes_layout.setSpacing(7)
        empty = QLabel('Классы появятся после загрузки файла')
        empty.setObjectName('viewerMeta')
        empty.setWordWrap(True)
        self.classes_layout.addWidget(empty)
        layout.addWidget(self.classes_container)
        return section

    def _build_settings_section(self) -> QFrame:
        section, layout = self._section('Настройки')
        self.preserve_settings_checkbox = QCheckBox(
            'Сохранять настройки при загрузке нового файла'
        )
        self.preserve_settings_checkbox.setToolTip(
            'Сохранить выбранные классы, цвета и размеры при смене облака.'
        )
        layout.addWidget(self.preserve_settings_checkbox)
        self.save_settings_checkbox = QCheckBox('Сохранять при выходе')
        self.save_settings_checkbox.setChecked(True)
        layout.addWidget(self.save_settings_checkbox)
        clear_button = QPushButton('Сбросить сохранённые настройки')
        clear_button.setObjectName('viewerSecondaryButton')
        clear_button.clicked.connect(self._clear_saved_settings)
        layout.addWidget(clear_button)
        return section

    @staticmethod
    def _control_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName('viewerControlLabel')
        return label

    def select_file(self) -> None:
        start_directory = self.settings.value('last_file', '', type=str)
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            'Выберите облако точек',
            start_directory,
            'Облака точек (*.xyz *.txt *.pts *.las *.laz);;Все файлы (*)',
        )
        if selected:
            self.load_file(Path(selected))

    def open_last_file(self) -> None:
        last_path = Path(self.settings.value('last_file', '', type=str))
        if not last_path.is_file():
            self.last_file_action.setEnabled(False)
            QMessageBox.information(
                self,
                'Последний файл недоступен',
                'Последний открытый файл был перемещён или удалён.',
            )
            return
        self.load_file(last_path)

    def load_file(self, path: Path) -> None:
        if self._thread is not None:
            QMessageBox.information(
                self,
                'Загрузка уже идёт',
                'Дождитесь окончания текущей загрузки.',
            )
            return
        if path.suffix.lower() not in SUPPORTED_POINT_CLOUD_EXTENSIONS:
            QMessageBox.warning(
                self,
                'Неподдерживаемый формат',
                'Выберите XYZ, TXT, PTS, LAS или LAZ.',
            )
            return

        self.open_button.setEnabled(False)
        self.load_all_points.setEnabled(False)
        self.loading_bar.setValue(0)
        if self.load_all_points.isChecked():
            self.loading_bar.setFormat('Загрузка всех точек: %p%')
            max_points = None
        else:
            self.loading_bar.setFormat('Загрузка выборки: %p%')
            max_points = 1_500_000
        self.loading_bar.show()
        self.filename_label.setText(path.name)
        self.points_label.setText('Чтение облака точек…')
        self.file_status.setText(path.name)

        thread = QThread(self)
        worker = PointCloudLoadWorker(path, max_points=max_points)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.loading_bar.setValue)
        worker.loaded.connect(self._cloud_loaded)
        worker.failed.connect(self._load_failed)
        worker.cancelled.connect(self._load_cancelled)
        worker.loaded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._load_thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(object)
    def _cloud_loaded(self, cloud: PointCloudData) -> None:
        preserved_classes = (
            self._capture_class_settings()
            if self.cloud is not None and self.preserve_settings_checkbox.isChecked()
            else {}
        )
        self.cloud = cloud
        self.canvas.set_cloud(cloud)
        if preserved_classes:
            self._apply_class_settings(preserved_classes, cloud)
        elif self.save_settings_checkbox.isChecked():
            self._apply_saved_class_settings(cloud)
        self.screenshot_action.setEnabled(True)
        for action in self.convert_actions:
            action.setEnabled(True)
        self.settings.setValue('last_file', str(cloud.path))
        self.settings.sync()
        self.last_file_action.setEnabled(True)
        self.filename_label.setText(cloud.path.name)

        rendered = f'{cloud.rendered_count:,}'.replace(',', ' ')
        source = f'{cloud.source_count:,}'.replace(',', ' ')
        if cloud.sampled:
            self.points_label.setText(
                f'Отображается {rendered} из {source} точек · равномерная выборка'
            )
        else:
            self.points_label.setText(f'{source} точек')
        if cloud.invalid_count:
            self.points_label.setText(
                f'{self.points_label.text()} · пропущено строк: {cloud.invalid_count}'
            )

        self.range_label.setText(
            f'X [{cloud.bounds_min[0]:.2f}; {cloud.bounds_max[0]:.2f}]  ·  Y [{cloud.bounds_min[1]:.2f}; {cloud.bounds_max[1]:.2f}]  ·  Z [{cloud.bounds_min[2]:.2f}; {cloud.bounds_max[2]:.2f}]'
        )
        self.file_status.setText(f'{cloud.path.name} · {rendered} точек')

        self.color_mode.model().item(2).setEnabled(cloud.rgb is not None)
        if self.color_mode.currentData() == 'rgb' and cloud.rgb is None:
            self.color_mode.setCurrentIndex(0)
        self._configure_clip(cloud)
        self._rebuild_classes(cloud)

    def _capture_class_settings(
        self,
    ) -> dict[int, tuple[bool, QColor, float, float]]:
        return {
            class_id: (
                class_id in self.canvas.visible_classes,
                QColor(self.canvas.class_colors[class_id]),
                self.canvas.class_sizes.get(class_id, 0.15),
                self.canvas.class_multipliers.get(class_id, 1.0),
            )
            for class_id in self.canvas.class_colors
        }

    def _apply_class_settings(
        self,
        values: dict[int, tuple[bool, QColor, float, float]],
        cloud: PointCloudData,
    ) -> None:
        for class_id in (int(value) for value in cloud.unique_classes):
            if class_id not in values:
                continue
            visible, color, size, multiplier = values[class_id]
            self.canvas.set_class_visible(class_id, visible)
            self.canvas.set_class_color(class_id, color)
            self.canvas.set_class_size(class_id, size)
            self.canvas.set_class_multiplier(class_id, multiplier)

    def _configure_clip(self, cloud: PointCloudData) -> None:
        minimum = float(cloud.bounds_min[2])
        maximum = float(cloud.bounds_max[2])
        margin = max((maximum - minimum) * 0.1, 1.0)
        for spinbox in (self.clip_minimum, self.clip_maximum):
            spinbox.blockSignals(True)
            spinbox.setRange(minimum - margin, maximum + margin)
        self.clip_minimum.setValue(minimum)
        self.clip_maximum.setValue(maximum)
        self.clip_minimum.blockSignals(False)
        self.clip_maximum.blockSignals(False)
        self.clip_enabled.setChecked(False)

        xy_range = max(
            float(cloud.bounds_max[0] - cloud.bounds_min[0]),
            float(cloud.bounds_max[1] - cloud.bounds_min[1]),
        )
        self.profile_width.setValue(max(0.1, xy_range / 60.0))

    def _rebuild_classes(self, cloud: PointCloudData) -> None:
        while self.classes_layout.count():
            item = self.classes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._class_rows.clear()

        counts = cloud.class_counts()
        for class_id in sorted(counts):
            style = class_style_for(class_id)
            row = QFrame()
            row.setObjectName('viewerClassRow')
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(9, 8, 9, 8)
            row_layout.setSpacing(7)
            top_row = QHBoxLayout()
            top_row.setSpacing(7)

            checkbox = QCheckBox()
            checkbox.setChecked(class_id in self.canvas.visible_classes)
            checkbox.setToolTip('Показать или скрыть класс')
            checkbox.toggled.connect(
                lambda visible, value=class_id: self.canvas.set_class_visible(
                    value,
                    visible,
                )
            )
            color_button = QPushButton()
            color_button.setObjectName('viewerColorButton')
            color_button.setFixedSize(24, 24)
            class_color = self.canvas.class_colors.get(
                class_id,
                QColor(style.hex_color),
            )
            color_button.setProperty('color', class_color.name())
            self._set_color_button(color_button, class_color)
            color_button.clicked.connect(
                lambda _checked=False, value=class_id, button=color_button: self._select_class_color(
                    value,
                    button,
                )
            )
            text = QLabel(f'{class_id} · {style.title}')
            text.setObjectName('viewerClassName')
            count = QLabel(f'{counts[class_id]:,}'.replace(',', ' '))
            count.setObjectName('viewerClassCount')
            size = QDoubleSpinBox()
            size.setObjectName('viewerClassSize')
            size.setRange(0.0, 10.0)
            size.setSingleStep(0.05)
            size.setDecimals(2)
            size.setValue(self.canvas.class_sizes.get(class_id, 1.0))
            size.setToolTip('Абсолютный размер точек класса от 0 до 10')
            size.valueChanged.connect(
                lambda value, selected_class=class_id: self.canvas.set_class_size(
                    selected_class,
                    value,
                )
            )
            multiplier = QDoubleSpinBox()
            multiplier.setObjectName('viewerClassMultiplier')
            multiplier.setRange(0.1, 10.0)
            multiplier.setSingleStep(0.1)
            multiplier.setDecimals(1)
            multiplier.setSuffix('×')
            multiplier.setValue(
                self.canvas.class_multipliers.get(class_id, 1.0)
            )
            multiplier.setToolTip(
                'Дополнительный множитель абсолютного размера класса'
            )
            multiplier.valueChanged.connect(
                lambda value, selected_class=class_id: self.canvas.set_class_multiplier(
                    selected_class,
                    value,
                )
            )

            top_row.addWidget(checkbox)
            top_row.addWidget(color_button)
            top_row.addWidget(text, 1)
            top_row.addWidget(count)
            row_layout.addLayout(top_row)

            controls_row = QHBoxLayout()
            controls_row.setSpacing(6)
            size_label = QLabel('Размер')
            size_label.setObjectName('viewerClassControlLabel')
            multiplier_label = QLabel('Множитель')
            multiplier_label.setObjectName('viewerClassControlLabel')
            controls_row.addWidget(size_label)
            controls_row.addWidget(size, 1)
            controls_row.addSpacing(4)
            controls_row.addWidget(multiplier_label)
            controls_row.addWidget(multiplier, 1)
            row_layout.addLayout(controls_row)
            self.classes_layout.addWidget(row)
            self._class_rows[class_id] = (
                checkbox,
                color_button,
                size,
                multiplier,
            )

    def _select_measurement_color(self, target: str) -> None:
        initial = (
            self.canvas.marker_color
            if target == 'marker'
            else self.canvas.line_color
        )
        color = QColorDialog.getColor(initial, self, 'Цвет измерения')
        if not color.isValid():
            return
        marker = color if target == 'marker' else self.canvas.marker_color
        line = color if target == 'line' else self.canvas.line_color
        self.canvas.set_measurement_colors(marker, line)
        self._set_color_button(
            self.marker_color_button,
            self.canvas.marker_color,
        )
        self._set_color_button(self.line_color_button, self.canvas.line_color)

    def _change_navigation_mode(self, index: int) -> None:
        mode = str(self.navigation_combo.itemData(index))
        flight_enabled = mode == 'flight'
        self.canvas.set_navigation_mode(mode)
        self.flight_speed.setEnabled(flight_enabled)
        if flight_enabled:
            self.camera_hint.setText(
                'Кликните по сцене · WASD — полёт · Space — вверх · '
                'Shift — вниз · ЛКМ — направление взгляда · Home — сброс'
            )
            self.tool_status.setText(
                'Свободный полёт: WASD · Space/Shift · ЛКМ — обзор'
            )
        else:
            self.camera_hint.setText(
                'Стрелки — сдвиг · W/S — масштаб · Home — сброс'
            )
            self.tool_status.setText(
                'ЛКМ — вращение · ПКМ — перемещение · колесо — масштаб'
            )
        QTimer.singleShot(0, self.canvas.setFocus)

    def _restore_settings(self) -> None:
        self.save_settings_checkbox.setChecked(
            self.settings.value('save_on_exit', True, type=bool)
        )
        self.preserve_settings_checkbox.setChecked(
            self.settings.value('preserve_on_load', False, type=bool)
        )
        self.load_all_points.setChecked(
            self.settings.value('load_all_points', True, type=bool)
        )
        dark_theme = self.settings.value('dark_theme', False, type=bool)
        self.theme_toggle.setChecked(dark_theme)
        self._apply_theme(dark_theme)
        mode = self.settings.value('color_mode', 'classification', type=str)
        mode_index = self.color_mode.findData(mode)
        self.color_mode.setCurrentIndex(max(0, mode_index))
        gradient = self.settings.value('gradient', 'RAINBOW', type=str)
        if gradient in GRADIENTS:
            self.gradient_combo.setCurrentText(gradient)
        self.point_size_slider.setValue(
            self.settings.value('point_size', 10, type=int)
        )
        self.round_points.setChecked(
            self.settings.value('round_points', True, type=bool)
        )
        self.edge_detection.setChecked(
            self.settings.value('edge_detection', False, type=bool)
        )
        self.size_attenuation.setChecked(
            self.settings.value('size_attenuation', True, type=bool)
        )
        self.grid_visible.setChecked(
            self.settings.value('grid_visible', True, type=bool)
        )
        self.gradient_background.setChecked(
            self.settings.value('gradient_background', False, type=bool)
        )
        self.field_of_view.setValue(
            self.settings.value('field_of_view', 75.0, type=float)
        )
        perspective = self.settings.value('perspective', True, type=bool)
        self.projection_combo.setCurrentIndex(0 if perspective else 1)
        self.flight_speed.setValue(
            self.settings.value('flight_speed', 10.0, type=float)
        )
        navigation_mode = self.settings.value(
            'navigation_mode',
            'orbit',
            type=str,
        )
        navigation_index = self.navigation_combo.findData(navigation_mode)
        self.navigation_combo.setCurrentIndex(max(0, navigation_index))
        self._change_navigation_mode(self.navigation_combo.currentIndex())
        profile_mode = self.settings.value('profile_clip_mode', 'inside', type=str)
        profile_mode_index = self.profile_clip_mode.findData(profile_mode)
        self.profile_clip_mode.setCurrentIndex(max(0, profile_mode_index))
        self.profile_main_clipping.setChecked(
            self.settings.value('profile_main_clipping', False, type=bool)
        )
        self.profile_helpers.setChecked(
            self.settings.value('profile_helpers', False, type=bool)
        )

        marker = QColor(self.settings.value('marker_color', '#FFFF00', type=str))
        line = QColor(self.settings.value('line_color', '#3B82F6', type=str))
        self.canvas.set_measurement_colors(marker, line)
        self._set_color_button(self.marker_color_button, marker)
        self._set_color_button(self.line_color_button, line)

    def _apply_saved_class_settings(self, cloud: PointCloudData) -> None:
        for class_id in (int(value) for value in cloud.unique_classes):
            visible = self.settings.value(
                f'classes/{class_id}/visible',
                True,
                type=bool,
            )
            color_name = self.settings.value(
                f'classes/{class_id}/color',
                class_style_for(class_id).hex_color,
                type=str,
            )
            default_size = self.canvas.class_sizes.get(class_id, 1.0)
            size = self.settings.value(
                f'classes/{class_id}/size',
                default_size,
                type=float,
            )
            multiplier = self.settings.value(
                f'classes/{class_id}/multiplier',
                1.0,
                type=float,
            )
            self.canvas.set_class_visible(class_id, visible)
            self.canvas.set_class_color(class_id, QColor(color_name))
            self.canvas.set_class_size(class_id, size)
            self.canvas.set_class_multiplier(class_id, multiplier)

    @Slot(bool)
    def _apply_theme(self, dark: bool) -> None:
        theme = 'dark' if dark else 'light'
        self.setProperty('viewerTheme', theme)
        self.theme_toggle.setToolTip(
            'Включить светлую тему' if dark else 'Включить тёмную тему'
        )
        for dialog in self._profile_windows:
            dialog.setProperty('viewerTheme', theme)
            for widget in (dialog, *dialog.findChildren(QWidget)):
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()
        for widget in (self, *self.findChildren(QWidget)):
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()
        self.update()

    def _save_settings(self) -> None:
        self.settings.setValue('save_on_exit', self.save_settings_checkbox.isChecked())
        self.settings.setValue(
            'preserve_on_load',
            self.preserve_settings_checkbox.isChecked(),
        )
        self.settings.setValue('load_all_points', self.load_all_points.isChecked())
        self.settings.setValue('dark_theme', self.theme_toggle.isChecked())
        if not self.save_settings_checkbox.isChecked():
            self.settings.sync()
            return
        self.settings.setValue('color_mode', self.color_mode.currentData())
        self.settings.setValue('gradient', self.gradient_combo.currentText())
        self.settings.setValue('point_size', self.point_size_slider.value())
        self.settings.setValue('round_points', self.round_points.isChecked())
        self.settings.setValue('edge_detection', self.edge_detection.isChecked())
        self.settings.setValue('size_attenuation', self.size_attenuation.isChecked())
        self.settings.setValue('grid_visible', self.grid_visible.isChecked())
        self.settings.setValue(
            'gradient_background',
            self.gradient_background.isChecked(),
        )
        self.settings.setValue('field_of_view', self.field_of_view.value())
        self.settings.setValue('perspective', self.projection_combo.currentData())
        self.settings.setValue(
            'navigation_mode',
            self.navigation_combo.currentData(),
        )
        self.settings.setValue('flight_speed', self.flight_speed.value())
        self.settings.setValue('profile_clip_mode', self.profile_clip_mode.currentData())
        self.settings.setValue(
            'profile_main_clipping',
            self.profile_main_clipping.isChecked(),
        )
        self.settings.setValue('profile_helpers', self.profile_helpers.isChecked())
        self.settings.setValue('marker_color', self.canvas.marker_color.name())
        self.settings.setValue('line_color', self.canvas.line_color.name())
        for class_id, (
            checkbox,
            _color_button,
            size,
            multiplier,
        ) in self._class_rows.items():
            self.settings.setValue(
                f'classes/{class_id}/visible',
                checkbox.isChecked(),
            )
            self.settings.setValue(
                f'classes/{class_id}/color',
                self.canvas.class_colors[class_id].name(),
            )
            self.settings.setValue(f'classes/{class_id}/size', size.value())
            self.settings.setValue(
                f'classes/{class_id}/multiplier',
                multiplier.value(),
            )
        self.settings.sync()

    def _clear_saved_settings(self) -> None:
        self.settings.clear()
        self.settings.sync()
        self.tool_status.setText('Сохранённые настройки сброшены')

    def _select_class_color(self, class_id: int, button: QPushButton) -> None:
        initial = self.canvas.class_colors.get(class_id, QColor('#FFFFFF'))
        color = QColorDialog.getColor(initial, self, 'Цвет класса')
        if not color.isValid():
            return
        self.canvas.set_class_color(class_id, color)
        self._set_color_button(button, color)

    @staticmethod
    def _set_color_button(button: QPushButton, color: QColor) -> None:
        button.setStyleSheet(
            'QPushButton {'
            f'background-color: {color.name()};'
            'border: 2px solid #FFFFFF; border-radius: 7px;'
            '}'
            'QPushButton:hover { border-color: #3867F5; }'
        )

    @Slot(str)
    def _load_failed(self, error: str) -> None:
        self.points_label.setText('Не удалось загрузить файл')
        QMessageBox.critical(self, 'Ошибка загрузки облака точек', error)

    @Slot()
    def _load_cancelled(self) -> None:
        self.points_label.setText('Загрузка отменена')

    @Slot()
    def _load_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.open_button.setEnabled(True)
        self.load_all_points.setEnabled(True)
        self.loading_bar.hide()
        if self._close_when_finished:
            self._close_when_finished = False
            self.close()

    def _change_color_mode(self, _index: int) -> None:
        mode = self.color_mode.currentData()
        if mode:
            self.canvas.set_color_mode(str(mode))

    def _change_point_size(self, value: int) -> None:
        size = value / 10.0
        self.point_size_value.setText(f'{size:.1f}')
        self.canvas.set_point_size(size)

    def _apply_clip(self, _value=None) -> None:
        self.canvas.set_clip(
            self.clip_enabled.isChecked(),
            self.clip_minimum.value(),
            self.clip_maximum.value(),
        )

    @Slot(object, object)
    def _show_profile(self, start: np.ndarray, end: np.ndarray) -> None:
        if self.cloud is None:
            return
        dialog = ProfileDialog(
            self.cloud,
            start,
            end,
            self.profile_width.value(),
            self.canvas.class_colors,
            self.canvas,
            self,
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._profile_windows.add(dialog)
        dialog.destroyed.connect(
            lambda _object=None, window=dialog: self._profile_windows.discard(window)
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def save_screenshot(self) -> None:
        if self.cloud is None:
            return
        default_path = self.cloud.path.with_name(f'{self.cloud.path.stem}_3d.png')
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            'Сохранить снимок',
            str(default_path),
            'PNG (*.png);;JPEG (*.jpg *.jpeg)',
        )
        if not selected:
            return
        image = self.canvas.grabFramebuffer()
        if not image.save(selected):
            QMessageBox.critical(
                self,
                'Ошибка сохранения',
                'Не удалось сохранить снимок.',
            )

    def export_cloud(self, extension: str | None = None) -> None:
        if self.cloud is None:
            return
        if self.cloud.sampled:
            answer = QMessageBox.warning(
                self,
                'Облако отображается частично',
                'Файл слишком большой, поэтому просмотрщик использует '
                'равномерную выборку. В экспорт попадут только отображаемые '
                'точки. Продолжить?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        suffix = extension or '.las'
        filter_by_suffix = {
            '.xyz': 'XYZ (*.xyz)',
            '.txt': 'TXT (*.txt)',
            '.las': 'LAS (*.las)',
            '.laz': 'Сжатый LAZ (*.laz)',
        }
        default_path = self.cloud.path.with_name(
            f'{self.cloud.path.stem}_converted{suffix}'
        )
        selected_filter = filter_by_suffix.get(suffix, 'LAS (*.las)')
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            'Конвертировать облако точек',
            str(default_path),
            'LAS (*.las);;Сжатый LAZ (*.laz);;XYZ (*.xyz);;TXT (*.txt)',
            selected_filter,
        )
        if not selected:
            return
        target = Path(selected)
        if not target.suffix:
            target = target.with_suffix(suffix)
        try:
            saved = save_point_cloud(self.cloud, target)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, 'Ошибка конвертации', str(error))
            return
        QMessageBox.information(
            self,
            'Конвертация завершена',
            f'Файл сохранён:\n{saved}',
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if any(
            url.isLocalFile()
            and Path(url.toLocalFile()).suffix.lower() in SUPPORTED_POINT_CLOUD_EXTENSIONS
            for url in urls
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() in SUPPORTED_POINT_CLOUD_EXTENSIONS:
                self.load_file(path)
                event.acceptProposedAction()
                return

    @Slot(str)
    def _set_tool_status(self, text: str) -> None:
        self.tool_status.setText(text)

    @Slot(str)
    def _set_camera_status(self, text: str) -> None:
        self.camera_status.setText(text)

    @Slot(str)
    def _show_render_error(self, error: str) -> None:
        QMessageBox.critical(
            self,
            'Ошибка 3D-отрисовки',
            'Не удалось инициализировать OpenGL. Обновите драйвер видеокарты.\n\n'
            f'{error}',
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, 'canvas_card'):
            self._position_immersive_menu()

    def reject(self) -> None:
        if self._immersive_mode:
            self.set_immersive_mode(False)
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_settings()
        if self._thread is not None and self._thread.isRunning():
            self._close_when_finished = True
            self._thread.requestInterruption()
            self.points_label.setText('Отмена загрузки…')
            event.ignore()
            return
        event.accept()
