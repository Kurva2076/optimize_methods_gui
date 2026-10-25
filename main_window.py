from PySide6.QtGui import QCloseEvent
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QStackedWidget, QVBoxLayout, QWidget,
)

from resources import image_path
from screens import SCREENS, BaseScreen
from stepper import Stepper


class MainWindow(QMainWindow):
    """Оболочка приложения: шапка со степпером + стек экранов."""

    def __init__(self, screen_classes: list[type[BaseScreen]] = SCREENS) -> None:
        super().__init__()
        self.setWindowTitle('PointCloud — классификация облаков точек')
        self.resize(1320, 720)
        self.setMinimumSize(1200, 640)
        self._center_on_screen()

        self.screens: list[BaseScreen] = [cls() for cls in screen_classes]
        self.current_index = 0

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

        root = QVBoxLayout(central)
        root.setContentsMargins(26, 20, 26, 26)
        root.setSpacing(20)

        root.addLayout(self._build_header())

        self.stack = QStackedWidget()
        for screen in self.screens:
            self.stack.addWidget(screen)
            screen.busy_changed.connect(self._on_busy_changed)
        root.addWidget(self.stack, 1)

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(30)

        header.addLayout(self._build_brand())

        self.stepper = Stepper([s.step_title for s in self.screens])
        self.stepper.step_clicked.connect(self.go_to)
        header.addWidget(self.stepper, 1)
        return header

    @staticmethod
    def _build_brand() -> QHBoxLayout:
        brand = QHBoxLayout()
        brand.setSpacing(12)

        logo_file = image_path('logo.svg')
        if logo_file.exists():
            from PySide6.QtCore import QSize
            from PySide6.QtGui import QIcon

            logo = QLabel()
            logo.setPixmap(QIcon(str(logo_file)).pixmap(QSize(56, 56)))
            brand.addWidget(logo)

        texts = QVBoxLayout()
        texts.setSpacing(0)
        texts.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        name = QLabel('PointCloud')
        name.setObjectName('brandTitle')
        tagline = QLabel('Классификация облаков точек')
        tagline.setObjectName('brandSubtitle')
        texts.addWidget(name)
        texts.addWidget(tagline)
        brand.addLayout(texts)
        return brand

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
        self.stepper.set_current(index)
        self.screens[index].on_activated()

    def _on_busy_changed(self, _busy: bool) -> None:
        self.stepper.set_locked(any(s.is_busy() for s in self.screens))

    def closeEvent(self, event: QCloseEvent) -> None:
        if any(s.is_busy() for s in self.screens):
            QMessageBox.warning(self, 'Идёт обработка', 'Дождитесь завершения оптимизации перед закрытием окна.')
            event.ignore()
            return
        event.accept()
