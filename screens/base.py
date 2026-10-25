from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class BaseScreen(QWidget):
    """Базовый класс экрана.

    Чтобы добавить новый экран:
      1. Унаследуйтесь от BaseScreen.
      2. Задайте step_title (подпись в степпере), title и subtitle (заголовок экрана).
      3. Реализуйте build_content(layout) — там строится интерфейс.
      4. Добавьте класс в список SCREENS в screens/__init__.py.
    """

    step_title: str = ''
    title: str = ''
    subtitle: str = ''

    busy_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._busy = False
        self.init_state()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        title_label = QLabel(self.title)
        title_label.setObjectName('screenTitle')
        subtitle_label = QLabel(self.subtitle)
        subtitle_label.setObjectName('screenSubtitle')
        root.addWidget(title_label)
        root.addWidget(subtitle_label)
        root.addSpacing(16)

        self.build_content(root)

    def init_state(self) -> None:
        """Инициализация данных экрана. Вызывается до построения интерфейса."""

    def build_content(self, layout: QVBoxLayout) -> None:
        """Построение интерфейса. Виджеты добавляются в переданный layout."""
        raise NotImplementedError

    def on_activated(self) -> None:
        """Вызывается каждый раз, когда экран становится видимым."""

    def is_busy(self) -> bool:
        return self._busy

    def set_busy(self, busy: bool) -> None:
        if self._busy != busy:
            self._busy = busy
            self.busy_changed.emit(busy)
