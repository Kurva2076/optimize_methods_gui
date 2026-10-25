from PySide6.QtWidgets import QProgressBar, QStyle, QStyleOptionProgressBar
from PySide6.QtGui import QPainter, QColor
from PySide6.QtCore import Qt


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
