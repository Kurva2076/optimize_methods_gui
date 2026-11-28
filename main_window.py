from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from screens import SCREENS, BaseScreen
from screens.preprocessing import PreprocessingScreen
from screens.studio import STUDIO_PAGE_CLASSES, ProjectsPage, StudioPage
from screens.studio_advanced import ADVANCED_STUDIO_PAGES
from studio_context import StudioContext
from studio_widgets import StudioSidebar


class MainWindow(QMainWindow):
    """Единая оболочка PointCloud с навигацией только в боковом меню."""

    def __init__(self, screen_classes: list[type[BaseScreen]] = SCREENS) -> None:
        super().__init__()
        self.setWindowTitle('PointCloud — ML Studio')
        self.resize(1500, 900)
        self.setMinimumSize(1280, 720)
        self._center_on_screen()

        self.studio_context = StudioContext(self)
        preprocessing_classes = [
            cls for cls in screen_classes if issubclass(cls, PreprocessingScreen)
        ]
        legacy_screens: list[BaseScreen] = [cls() for cls in preprocessing_classes]
        self.preprocessing_screen = next(
            (
                screen
                for screen in legacy_screens
                if isinstance(screen, PreprocessingScreen)
            ),
            None,
        )
        if self.preprocessing_screen is None:
            raise ValueError('В наборе экранов отсутствует PreprocessingScreen')
        # Старые этапы 2–5 больше не входят в пользовательский маршрут. Их
        # функциональность представлена разделами ML Studio слева.
        self.screens: list[BaseScreen] = [self.preprocessing_screen]
        page_classes = {**STUDIO_PAGE_CLASSES, **ADVANCED_STUDIO_PAGES}
        self.studio_pages: dict[str, StudioPage] = {
            key: page_class(self.studio_context)
            for key, page_class in page_classes.items()
        }
        for page in self.studio_pages.values():
            page.project_selection_requested.connect(
                lambda: self.show_studio_section('projects')
            )
        projects = self.studio_pages.get('projects')
        if isinstance(projects, ProjectsPage):
            projects.project_opened.connect(
                lambda _root: self.show_studio_section('datasets')
            )
        self._init_ui()
        self.show_preprocessing()

    def _center_on_screen(self) -> None:
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    def _init_ui(self) -> None:
        central = QWidget()
        central.setObjectName('centralWidget')
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = StudioSidebar(self.studio_context)
        self.sidebar.section_requested.connect(self.show_studio_section)
        root.addWidget(self.sidebar)

        workspace = QWidget()
        workspace.setObjectName('studioWorkspace')
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(20, 18, 20, 20)
        workspace_layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.preprocessing_screen)
        self.preprocessing_screen.busy_changed.connect(self._on_busy_changed)
        for page in self.studio_pages.values():
            self.stack.addWidget(page)
        workspace_layout.addWidget(self.stack, 1)
        root.addWidget(workspace, 1)

    def go_to(self, index: int) -> None:
        """Совместимость со старым API: доступен только первый этап."""
        if index == 0:
            self.show_preprocessing()

    def show_preprocessing(self) -> None:
        self.stack.setCurrentWidget(self.preprocessing_screen)
        self.sidebar.set_active('preprocessing')
        self.preprocessing_screen.on_activated()

    def show_studio_section(self, section: str) -> None:
        if section == 'preprocessing':
            self.show_preprocessing()
            return
        page = self.studio_pages.get(section)
        if page is None:
            return
        if self.preprocessing_screen.is_busy():
            QMessageBox.warning(
                self,
                'Идёт обработка',
                'Дождитесь завершения обработки перед переходом на другой экран.',
            )
            return
        page.refresh()
        self.stack.setCurrentWidget(page)
        self.sidebar.set_active(section)

    def _on_busy_changed(self, _busy: bool) -> None:
        self.sidebar.setEnabled(not self.preprocessing_screen.is_busy())

    def closeEvent(self, event: QCloseEvent) -> None:
        if any(s.is_busy() for s in self.screens):
            QMessageBox.warning(self, 'Идёт обработка', 'Дождитесь завершения оптимизации перед закрытием окна.')
            event.ignore()
            return
        running = self.studio_context.processes.running()
        if running:
            answer = QMessageBox.question(
                self,
                'Остановить процессы?',
                'При закрытии приложения будут остановлены все запущенные '
                'обучения и тесты. Закрыть приложение?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.studio_context.processes.stop_all()
        event.accept()
