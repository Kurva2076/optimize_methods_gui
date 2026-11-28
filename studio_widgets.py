"""Переиспользуемые виджеты единой оболочки PointCloud / ML Studio."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from resources import image_path, studio_icon_path
from studio_context import StudioContext
from studio_theme import apply_studio_theme

NAVIGATION = (
    ('projects', 'Проекты', 'projects.svg'),
    ('preprocessing', 'Предобработка', 'datasets.svg'),
    ('datasets', 'Датасеты', 'datasets.svg'),
    ('runs', 'Запуски', 'runs.svg'),
    ('live', 'Live', 'live.svg'),
    ('testing', 'Тестирование', 'testing.svg'),
    ('scenarios', 'Сценарии', 'scenarios.svg'),
    ('artifacts', 'Артефакты', 'artifacts.svg'),
    ('environment', 'Окружение', 'environment.svg'),
)


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def open_in_file_manager(path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


def tinted_icon(path: Path, color: str, size: QSize | None = None) -> QIcon:
    """Окрасить SVG-иконку в цвет текущей темы.

    Qt не поддерживает CSS ``currentColor`` в SVG так же, как
    браузер, поэтому на macOS и Windows такие иконки иначе
    остаются чёрными в тёмной теме.
    """
    icon_size = size or QSize(24, 24)
    pixmap = QIcon(str(path)).pixmap(icon_size)
    painter = QPainter(pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color))
    painter.end()
    return QIcon(pixmap)


class StudioSidebar(QFrame):
    section_requested = Signal(str)

    def __init__(
        self,
        context: StudioContext | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.context = context
        self.setObjectName('studioSidebar')
        self.setFixedWidth(220)
        self.buttons: dict[str, QPushButton] = {}
        self.icon_paths: dict[str, Path] = {}
        self.nav_titles = {key: title for key, title, _icon in NAVIGATION}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 18, 14, 16)
        layout.setSpacing(6)
        layout.addLayout(self._build_brand())
        layout.addSpacing(22)

        for key, title, icon_name in NAVIGATION:
            button = QPushButton(title)
            button.setObjectName('studioNavButton')
            button.setProperty('active', False)
            button.setIconSize(QSize(19, 19))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda _checked=False, section=key: self.section_requested.emit(section)
            )
            layout.addWidget(button)
            self.buttons[key] = button
            self.icon_paths[key] = studio_icon_path(icon_name)

        self.process_panel = self._build_processes()
        layout.addWidget(self.process_panel)
        layout.addStretch(1)
        layout.addWidget(self._build_resources())
        layout.addWidget(self._build_footer())

        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(2000)
        self.metrics_timer.timeout.connect(self.update_dynamic_state)
        self.metrics_timer.start()
        if self.context is not None:
            self.context.project_changed.connect(lambda _root: self.update_context())
            self.context.recent_projects_changed.connect(self.update_context)
            self.context.favorites_changed.connect(self.update_context)
            self.context.theme_changed.connect(self._theme_changed)
            self.context.processes.changed.connect(self.update_processes)
            apply_studio_theme(self.context.theme)
        self.update_dynamic_state()

    def _build_brand(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        logo_path = image_path('logo.svg')
        if logo_path.is_file():
            self.brand_logo = QLabel()
            row.addWidget(self.brand_logo)
        texts = QVBoxLayout()
        texts.setSpacing(0)
        title = QLabel('PointCloud')
        title.setObjectName('studioBrandTitle')
        subtitle = QLabel('ML Studio для LiDAR')
        subtitle.setObjectName('studioBrandSubtitle')
        texts.addWidget(title)
        texts.addWidget(subtitle)
        row.addLayout(texts, 1)
        return row

    def _build_resources(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName('resourcePanel')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.cpu_label, self.cpu_bar = self._add_meter(layout, 'CPU')
        self.ram_label, self.ram_bar = self._add_meter(layout, 'RAM')
        self.gpu_label, self.gpu_bar = self._add_meter(layout, 'GPU')
        return panel

    def _build_processes(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName('processPanel')
        panel.setMaximumHeight(150)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        title = QLabel('ПРОЦЕССЫ')
        title.setObjectName('processPanelTitle')
        layout.addWidget(title)
        self.process_list_layout = QVBoxLayout()
        self.process_list_layout.setSpacing(3)
        layout.addLayout(self.process_list_layout)
        panel.hide()
        return panel

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName('sidebarFooter')
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(7)
        self.project_path = QLabel('Проект не выбран')
        self.project_path.setObjectName('projectFooterPath')
        self.project_path.setWordWrap(True)
        self.project_path.setMaximumHeight(34)
        self.project_path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.project_path)
        self.theme_button = QPushButton('Тема: светлая')
        self.theme_button.setObjectName('themeButton')
        self.theme_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_button.clicked.connect(self.toggle_theme)
        layout.addWidget(self.theme_button)
        return footer

    @staticmethod
    def _add_meter(
        layout: QVBoxLayout,
        title: str,
    ) -> tuple[QLabel, QProgressBar]:
        row = QHBoxLayout()
        name = QLabel(title)
        name.setObjectName('resourceName')
        value = QLabel('—')
        value.setObjectName('resourceValue')
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(value)
        bar = QProgressBar()
        bar.setObjectName('resourceBar')
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(5)
        layout.addLayout(row)
        layout.addWidget(bar)
        return value, bar

    def set_active(self, section: str | None) -> None:
        for key, button in self.buttons.items():
            button.setProperty('active', key == section)
            repolish(button)
        self._refresh_icons()

    def _refresh_icons(self) -> None:
        dark = self.context is not None and self.context.theme == 'dark'
        for key, button in self.buttons.items():
            active = bool(button.property('active'))
            color = '#73A2FF' if dark and active else '#D8E2F0' if dark else '#245FE5' if active else '#253047'
            button.setIcon(tinted_icon(self.icon_paths[key], color, QSize(19, 19)))
        if hasattr(self, 'brand_logo'):
            logo_color = '#E6EDF7' if dark else '#111827'
            logo_path = image_path('logo.svg')
            self.brand_logo.setPixmap(
                tinted_icon(logo_path, logo_color, QSize(44, 44)).pixmap(QSize(44, 44))
            )

    def update_dynamic_state(self) -> None:
        self.update_metrics()
        self.update_context()
        self.update_processes()

    def update_context(self) -> None:
        if self.context is None:
            self.project_path.setText('Проект не выбран')
            return
        root = self.context.project_root
        self.project_path.setText(str(root) if root is not None else 'Проект не выбран')
        self.project_path.setToolTip(str(root) if root is not None else '')
        datasets = _dataset_count(root)
        runs = _run_count(root)
        self.buttons['datasets'].setText(f'Датасеты · {datasets}')
        self.buttons['runs'].setText(f'Запуски · {runs}')
        python = self.context.python_executable()
        environment_ready = (
            python is not None and '.venv311' in {part.casefold() for part in python.parts}
        )
        suffix = '' if environment_ready else ' · !'
        self.buttons['environment'].setText(f'Окружение{suffix}')
        theme_title = 'тёмная' if self.context.theme == 'dark' else 'светлая'
        self.theme_button.setText(f'Тема: {theme_title}')

    def update_processes(self) -> None:
        while self.process_list_layout.count():
            item = self.process_list_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        if self.context is None:
            self.process_panel.hide()
            return
        processes = self.context.processes.running()
        self.process_panel.setVisible(bool(processes))
        for process in processes[:3]:
            snapshot = process.snapshot()
            row = QFrame()
            row.setObjectName('processRow')
            layout = QHBoxLayout(row)
            layout.setContentsMargins(3, 1, 3, 1)
            layout.setSpacing(6)
            dot = QFrame()
            dot.setObjectName('processStatusDot')
            dot.setProperty('state', _process_state(snapshot.status))
            dot.setFixedSize(8, 8)
            button = QPushButton(
                f'{snapshot.name}\n{_format_duration(snapshot.seconds)}'
            )
            button.setObjectName('processRowButton')
            button.setToolTip(f'{snapshot.status}\n{" ".join(snapshot.command)}')
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda _checked=False: self.section_requested.emit('live')
            )
            layout.addWidget(dot)
            layout.addWidget(button, 1)
            self.process_list_layout.addWidget(row)

    def toggle_theme(self) -> None:
        if self.context is None:
            return
        self.context.toggle_theme()

    def _theme_changed(self, theme: str) -> None:
        apply_studio_theme(theme)
        self._refresh_icons()
        self.update_context()

    def update_metrics(self) -> None:
        cpu = _cpu_percent()
        ram = _memory_percent()
        gpu = _gpu_percent()
        self._set_meter(self.cpu_label, self.cpu_bar, cpu)
        self._set_meter(self.ram_label, self.ram_bar, ram)
        self._set_meter(self.gpu_label, self.gpu_bar, gpu)

    @staticmethod
    def _set_meter(label: QLabel, bar: QProgressBar, value: float | None) -> None:
        if value is None:
            label.setText('—')
            bar.setValue(0)
            return
        normalized = max(0, min(100, round(value)))
        label.setText(f'{normalized}%')
        bar.setValue(normalized)


class ProjectRequired(QFrame):
    choose_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('projectRequired')
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        title = QLabel('Сначала откройте проект ML Studio')
        title.setObjectName('emptyStateTitle')
        text = QLabel(
            'Выберите папку, в которой находится hpo_pointnet.py. '
            'После этого станут доступны реальные датасеты, запуски и артефакты.'
        )
        text.setObjectName('emptyStateText')
        text.setWordWrap(True)
        text.setMaximumWidth(520)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button = QPushButton('Выбрать проект')
        button.setObjectName('primaryButton')
        button.clicked.connect(self.choose_requested)
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)


def _cpu_percent() -> float | None:
    try:
        load = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        return min(100.0, load / cores * 100.0)
    except (AttributeError, OSError):
        return None


def _memory_percent() -> float | None:
    if os.name == 'nt':
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ('length', ctypes.c_ulong),
                    ('memory_load', ctypes.c_ulong),
                    ('total_phys', ctypes.c_ulonglong),
                    ('avail_phys', ctypes.c_ulonglong),
                    ('total_page', ctypes.c_ulonglong),
                    ('avail_page', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong),
                    ('avail_virtual', ctypes.c_ulonglong),
                    ('avail_extended_virtual', ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return float(status.memory_load)
        except (AttributeError, OSError):
            return None
    try:
        total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        available = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
        return (1.0 - available / total) * 100.0
    except (AttributeError, OSError, ValueError, ZeroDivisionError):
        return None


def _gpu_percent() -> float | None:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=0.8,
            check=False,
            # A windowed PyInstaller build has no parent console. Without this
            # flag Windows briefly creates one every time the GPU meter polls.
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def _dataset_count(root: Path | None) -> int:
    if root is None or not root.is_dir():
        return 0
    try:
        return sum(
            1
            for path in root.glob('exports*')
            if path.is_dir() and next(path.glob('scene_*.npy'), None) is not None
        )
    except OSError:
        return 0


def _run_count(root: Path | None) -> int:
    runs = root / 'runs' if root is not None else None
    if runs is None or not runs.is_dir():
        return 0
    try:
        return sum(
            1 for path in runs.iterdir() if path.is_dir() and (path / 'best.json').is_file()
        )
    except OSError:
        return 0


def _format_duration(seconds: int) -> str:
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def _process_state(status: str) -> str:
    normalized = status.casefold()
    if 'ошиб' in normalized:
        return 'error'
    if 'заверш' in normalized:
        return 'complete'
    if 'остан' in normalized:
        return 'stopped'
    return 'running'

