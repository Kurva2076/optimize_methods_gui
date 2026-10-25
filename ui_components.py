import os

from PySide6.QtWidgets import QProgressBar, QStyle, QStyleOptionProgressBar, QWidget, QFrame
from PySide6.QtGui import QPainter, QColor, QPixmap
from PySide6.QtCore import Qt


class RotatingIcon(QWidget):
    """Вращающаяся иконка фиксированного размера."""

    def __init__(
        self,
        pixmap: QPixmap,
        parent=None
    ) -> None:
        super().__init__(parent)

        self.pixmap = pixmap
        self.angle = 0

        self.setFixedSize(32, 32)

    def set_angle(self, angle: int) -> None:
        self.angle = angle % 360
        self.update()

    def paintEvent(self, event) -> None:
        if self.pixmap is None or self.pixmap.isNull():
            return

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            True
        )

        painter.translate(
            self.width() / 2,
            self.height() / 2
        )

        painter.rotate(self.angle)

        painter.drawPixmap(
            -(self.pixmap.width() // 2),
            -(self.pixmap.height() // 2),
            self.pixmap
        )


class DropZone(QFrame):
    """Кастомный фрейм, который принимает перетаскивание папок и файлов."""

    def __init__(self, parent=None, callback=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.callback = callback

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty('drag_active', 'true')
            self.style().unpolish(self)
            self.style().polish(self)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty('drag_active', 'false')
        self.style().unpolish(self)
        self.style().polish(self)
        event.accept()

    def dropEvent(self, event):
        self.setProperty('drag_active', 'false')
        self.style().unpolish(self)
        self.style().polish(self)

        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            target_dir = path if os.path.isdir(path) else os.path.dirname(path)

            if self.callback:
                self.callback(target_dir)
        event.acceptProposedAction()


class StyledProgressBar(QProgressBar):
    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        option = QStyleOptionProgressBar()
        self.initStyleOption(option)

        self.style().drawControl(
            QStyle.ControlElement.CE_ProgressBarGroove,
            option,
            painter,
            self,
        )

        self.style().drawControl(
            QStyle.ControlElement.CE_ProgressBarContents,
            option,
            painter,
            self,
        )

        text_rect = self.rect().adjusted(8, 0, -8, 0)
        text = self.text()

        painter.setFont(self.font())
        painter.setPen(QColor("#343A46"))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignCenter,
            text,
        )

        if self.maximum() > self.minimum():
            progress = (
                (self.value() - self.minimum())
                / (self.maximum() - self.minimum())
            )
        else:
            progress = 0

        filled_width = round(self.width() * progress)

        if filled_width > 0:
            painter.save()
            painter.setClipRect(
                0, 0, filled_width, self.height()
            )
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignCenter,
                text,
            )
            painter.restore()

        painter.end()
