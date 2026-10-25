from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from card import apply_shadow
from resources import icon

OVERLAY_COLOR = QColor(0x66, 0x69, 0x6F, 0x99)
SHADOW_MARGIN = 16


class ModalOverlay(QWidget):
    """Полупрозрачная вуаль поверх главного окна. Следит за его размером."""

    def __init__(self, host: QWidget) -> None:
        super().__init__(host)
        self._host = host
        self.hide()

    def show_over_host(self) -> None:
        self._host.installEventFilter(self)
        self.setGeometry(self._host.rect())
        self.raise_()
        self.show()

    def dismiss(self) -> None:
        self._host.removeEventFilter(self)
        self.hide()
        self.deleteLater()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._host and event.type() == QEvent.Type.Resize:
            self.setGeometry(self._host.rect())
            self.raise_()
        return super().eventFilter(watched, event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), OVERLAY_COLOR)


class BasePopup(QDialog):
    """
    Базовый класс попапа: белая карточка с тенью, шапка с крестиком, затемнение фона.

    Чтобы сделать новый попап:
      1. Унаследуйтесь от BasePopup и передайте в super().__init__ заголовок, подзаголовок
         и (необязательно) имя иконки из папки images.
      2. Реализуйте build_body(layout) — содержимое под шапкой.
      3. Откройте через popup.exec(): окно затемнится само и вернёт управление после закрытия.
    """

    max_size = QSize(1000, 820)

    def __init__(
        self,
        title: str,
        subtitle: str = '',
        icon_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        host = parent.window() if parent is not None else None
        super().__init__(host)
        self._host = host

        self.setObjectName('popupDialog')
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setModal(True)

        self.init_state()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SHADOW_MARGIN, SHADOW_MARGIN, SHADOW_MARGIN, SHADOW_MARGIN)

        card = QFrame()
        card.setObjectName('popupCard')
        apply_shadow(card, blur=36, y_offset=8, alpha=70)
        outer.addWidget(card)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(32, 26, 32, 24)
        layout.setSpacing(14)
        layout.addLayout(self._build_header(title, subtitle, icon_name))
        layout.addWidget(self._build_separator())
        self.build_body(layout)

    def init_state(self) -> None:
        """Инициализация данных попапа. Вызывается до построения интерфейса."""

    def build_body(self, layout: QVBoxLayout) -> None:
        """Содержимое попапа под шапкой."""
        raise NotImplementedError

    def _build_header(self, title: str, subtitle: str, icon_name: str | None) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(16)

        if icon_name:
            icon_label = QLabel()
            icon_label.setPixmap(icon(icon_name).pixmap(QSize(52, 52)))
            header.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        texts = QVBoxLayout()
        texts.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName('popupTitle')
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName('popupSubtitle')
        texts.addWidget(title_label)
        texts.addWidget(subtitle_label)
        header.addLayout(texts, 1)

        close_button = QPushButton()
        close_button.setObjectName('popupClose')
        close_button.setFixedSize(40, 40)
        close_button.setIcon(icon('close.svg'))
        close_button.setIconSize(QSize(22, 22))
        close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_button.setAutoDefault(False)
        close_button.clicked.connect(self.reject)
        header.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignTop)
        return header

    @staticmethod
    def _build_separator() -> QFrame:
        line = QFrame()
        line.setObjectName('popupSeparator')
        line.setFixedHeight(1)
        return line

    def _place_over_host(self) -> None:
        host = self._host
        if host is None:
            return
        width = min(self.max_size.width(), host.width() - 40)
        height = min(self.max_size.height(), host.height() - 40)
        self.resize(width, height)
        center = host.mapToGlobal(host.rect().center())
        self.move(center.x() - width // 2, center.y() - height // 2)

    def exec(self) -> int:
        overlay: ModalOverlay | None = None
        if self._host is not None:
            overlay = ModalOverlay(self._host)
            overlay.show_over_host()
        self._place_over_host()
        try:
            return super().exec()
        finally:
            if overlay is not None:
                overlay.dismiss()
