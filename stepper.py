from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QWidget


class StepItem(QWidget):
    """Один шаг: кружок с номером + подпись. Клик по любой части — переход."""

    clicked = Signal(int)

    def __init__(self, index: int, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._locked = False

        self.circle = QLabel(str(index + 1))
        self.circle.setObjectName('stepCircle')
        self.circle.setFixedSize(40, 40)
        self.circle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel(title)
        self.label.setObjectName('stepLabel')

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self.circle)
        layout.addWidget(self.label)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        for widget in (self.circle, self.label):
            widget.setProperty('active', active)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        self.setCursor(
            Qt.CursorShape.ForbiddenCursor if locked else Qt.CursorShape.PointingHandCursor
        )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._locked
            and self.rect().contains(event.position().toPoint())
        ):
            self.clicked.emit(self._index)
        super().mouseReleaseEvent(event)


class Stepper(QWidget):
    """Горизонтальный индикатор шагов с кликабельными заголовками."""

    step_clicked = Signal(int)

    def __init__(self, titles: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[StepItem] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        for index, title in enumerate(titles):
            if index:
                layout.addWidget(self._create_line(), 1)
            item = StepItem(index, title)
            item.clicked.connect(self.step_clicked)
            self._items.append(item)
            layout.addWidget(item)

    @staticmethod
    def _create_line() -> QFrame:
        line = QFrame()
        line.setObjectName('stepLine')
        line.setFixedHeight(2)
        line.setMinimumWidth(20)
        line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return line

    def set_current(self, index: int) -> None:
        for i, item in enumerate(self._items):
            item.set_active(i == index)

    def set_locked(self, locked: bool) -> None:
        """Блокирует переключение шагов (например, во время обработки)."""
        for item in self._items:
            item.set_locked(locked)
