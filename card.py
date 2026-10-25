from PySide6.QtCore import QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from resources import icon


def apply_shadow(widget: QWidget, blur: int = 22, y_offset: int = 5, alpha: int = 38) -> None:
    """Добавляет мягкую тень виджету."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setXOffset(0)
    shadow.setYOffset(y_offset)
    shadow.setColor(QColor(25, 40, 65, alpha))
    widget.setGraphicsEffect(shadow)


def repolish(widget: QWidget) -> None:
    """Заставляет Qt перечитать стили после смены dynamic-property (например, selected)."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def create_card(
    title: str,
    description: str | None = None,
    icon_name: str | None = None,
) -> tuple[QFrame, QVBoxLayout]:
    """
    Создаёт карточку с заголовком, (необязательной) иконкой и описанием.
    Возвращает саму карточку и её layout, в который экран добавляет содержимое.
    """
    card = QFrame()
    card.setObjectName('card')
    layout = QVBoxLayout(card)
    layout.setContentsMargins(22, 22, 22, 22)
    layout.setSpacing(10)

    title_row = QHBoxLayout()
    if icon_name:
        icon_label = QLabel()
        icon_label.setPixmap(icon(icon_name).pixmap(QSize(32, 32)))
        title_row.addWidget(icon_label)
    title_label = QLabel(title)
    title_label.setObjectName('cardTitle')
    title_row.addWidget(title_label)
    title_row.addStretch()
    layout.addLayout(title_row)

    if description:
        desc_label = QLabel(description)
        desc_label.setObjectName('cardDescription')
        layout.addWidget(desc_label)
        layout.addSpacing(5)

    apply_shadow(card)
    return card, layout
