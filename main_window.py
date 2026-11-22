from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from screens import SCREENS, BaseScreen
from screens.parameter_selection import ParameterSelectionScreen
from screens.studio import STUDIO_PAGE_CLASSES, ProjectsPage, StudioPage
from screens.studio_advanced import ADVANCED_STUDIO_PAGES
from screens.training import TrainingScreen
from stepper import Stepper
from studio_context import StudioContext
from studio_widgets import StudioSidebar


class MainWindow(QMainWindow):
    """Единая оболочка PointCloud: рабочий процесс + разделы ML Studio."""

    def __init__(self, screen_classes: list[type[BaseScreen]] = SCREENS) -> None:
        super().__init__()
        self.setWindowTitle('PointCloud — ML Studio')
        self.resize(1500, 900)
        self.setMinimumSize(1280, 720)
        self._center_on_screen()

        self.studio_context = StudioContext(self)
        self.screens: list[BaseScreen] = [cls() for cls in screen_classes]
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
        self.current_index = 0
        self.training_screen = next(
            (screen for screen in self.screens if isinstance(screen, TrainingScreen)),
            None,
        )
        for screen in self.screens:
            if isinstance(screen, ParameterSelectionScreen):
                screen.training_results_requested.connect(self.show_training_results)

        self._init_ui()
        self.go_to(0)

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
        workspace_layout.setSpacing(16)
        workspace_layout.addWidget(self._build_header())

        self.stack = QStackedWidget()
        for screen in self.screens:
            self.stack.addWidget(screen)
            screen.busy_changed.connect(self._on_busy_changed)
        for page in self.studio_pages.values():
            self.stack.addWidget(page)
        workspace_layout.addWidget(self.stack, 1)
        root.addWidget(workspace, 1)

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName('workflowHeader')
        header = QHBoxLayout(frame)
        header.setContentsMargins(22, 10, 22, 10)
        header.setSpacing(16)
        self.stepper = Stepper([s.step_title for s in self.screens])
        self.stepper.step_clicked.connect(self.go_to)
        header.addWidget(self.stepper, 1)
        return frame

    def go_to(self, index: int) -> None:
        if not 0 <= index < len(self.screens):
            return

        if index != self.current_index and self.screens[self.current_index].is_busy():
            QMessageBox.warning(
                self, 'Идёт обработка',
                'Дождитесь завершения обработки перед переходом на другой экран.',
            )
            return

        self.current_index = index
        self.stack.setCurrentIndex(index)
        self.sidebar.set_active(None)
        self.stepper.set_current(index)
        if self.screens[index] is self.training_screen:
            self.training_screen.show_stage_placeholder()
        self.screens[index].on_activated()

    def show_studio_section(self, section: str) -> None:
        page = self.studio_pages.get(section)
        if page is None:
            return
        if self.screens[self.current_index].is_busy():
            QMessageBox.warning(
                self,
                'Идёт обработка',
                'Дождитесь завершения обработки перед переходом на другой экран.',
            )
            return
        page.refresh()
        self.stack.setCurrentWidget(page)
        self.sidebar.set_active(section)
        stage_by_section = {
            # Каталог подготовленных NPY относится к настройке нейросети,
            # а не к первичной обработке исходных XYZ.
            'datasets': 1,
            'runs': 2,
            'live': 2,
            'scenarios': 2,
            'testing': 3,
            'artifacts': 4,
        }
        stage = stage_by_section.get(section)
        if stage is not None and stage < len(self.screens):
            self.stepper.set_current(stage)

    def show_training_results(self) -> None:
        if self.training_screen is None:
            return
        training_index = self.screens.index(self.training_screen)
        self.go_to(training_index)
        if self.current_index == training_index:
            self.training_screen.show_results()

    def _on_busy_changed(self, _busy: bool) -> None:
        self.stepper.set_locked(any(s.is_busy() for s in self.screens))

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
