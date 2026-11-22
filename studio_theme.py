"""Переключение светлой/тёмной темы без перезапуска приложения."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from resources import load_styles

DARK_STYLES = r"""
QMainWindow, QWidget#centralWidget, QWidget#studioWorkspace, QDialog {
    background: #0B1220;
    color: #E6EDF7;
}
QFrame#studioSidebar, QFrame#workflowHeader,
QFrame#studioCard, QFrame#projectRequired,
QFrame#card, QFrame#statusBox, QFrame#dropZone,
QFrame#resultsEmptyCard, QFrame#trainingContentCard,
QFrame#metricValueCard, #popupCard, #popupContent, #popupFooter {
    background: #111B2E;
    border-color: #263752;
}
QLabel, #studioBrandTitle, #panelTitle, #dialogTitle, #emptyStateTitle,
#studioPageTitle, #statValue, QLabel#trainingSectionTitle,
QLabel#metricValue, QLabel#emptyResultsTitle {
    color: #E6EDF7;
}
#stepLabel, #stepLabel[active="true"] {
    color: #E6EDF7;
}
#stepCircle {
    background: #263752;
    color: #D8E2F0;
}
#stepCircle[active="true"] {
    background: #356DF3;
    color: #FFFFFF;
}
#stepLine {
    background: #405573;
}
#studioBrandSubtitle, #studioPageSubtitle, #dialogSubtitle, #mutedText,
#emptyStateText, #statName, #legendText, QLabel#trainingSectionDescription,
QLabel#metricValueTitle, QLabel#emptyResultsDescription {
    color: #9BAAC0;
}
QPushButton#studioNavButton, QPushButton#secondaryButton,
QPushButton#secondaryAction, QPushButton#segmentButton,
QPushButton#themeButton, QPushButton#processRowButton {
    background: transparent;
    color: #D8E2F0;
    border-color: #344762;
}
QPushButton#dangerButton {
    background: #111B2E;
    color: #FF7B84;
    border-color: #6B313D;
}
QPushButton#dangerButton:hover {
    background: #2B1822;
}
QPushButton#studioNavButton:hover, QPushButton#secondaryButton:hover,
QPushButton#secondaryAction:hover, QPushButton#themeButton:hover,
QPushButton#processRowButton:hover {
    background: #18263C;
}
QPushButton#studioNavButton[active="true"] {
    background: #18325F;
    color: #73A2FF;
    border-left-color: #5B8EFF;
}
QFrame#resourcePanel, QFrame#sidebarFooter, QFrame#processPanel,
#projectPath, #infoNote, #runSummary {
    background: #0F192A;
    color: #C5D0DF;
    border-color: #263752;
}
#resourceName, #resourceValue, #projectFooterPath, #processPanelTitle {
    color: #B9C6D8;
}
QProgressBar#resourceBar, QProgressBar#sceneSizeBar {
    background: #263752;
}
QFrame#studioCard QListWidget, QListWidget#datasetList, QListWidget#runList,
QFrame#studioCard QTableWidget, QTableWidget#metricsTable,
QPlainTextEdit#jsonPreview, QPlainTextEdit#artifactPreview,
QPlainTextEdit#commandPreview,
QFrame#studioCard QLineEdit, QFrame#studioCard QComboBox,
QFrame#studioCard QSpinBox, QFrame#studioCard QDoubleSpinBox,
QDialog QLineEdit, QDialog QComboBox, QDialog QSpinBox,
QDialog QDoubleSpinBox, QComboBox QAbstractItemView,
QTextBrowser#artifactMarkdown {
    background: #0F192A;
    color: #D8E2F0;
    border-color: #2C3E59;
    selection-background-color: #244A87;
    selection-color: #FFFFFF;
}
QFrame#studioCard QTableWidget::item, QTableWidget#metricsTable::item {
    background: #0F192A;
    color: #D8E2F0;
    border-bottom-color: #24344D;
}
QFrame#studioCard QTableWidget::item:selected,
QListWidget#datasetList::item:selected, QListWidget#runList::item:selected,
QFrame#studioCard QListWidget::item:selected {
    background: #18325F;
    color: #73A2FF;
}
QHeaderView::section, QFrame#studioCard QTableWidget QHeaderView::section,
QTableWidget#metricsTable QHeaderView::section {
    background: #162238;
    color: #AEBBD0;
    border-color: #2C3E59;
}
QFrame#studioCard QHeaderView {
    background: #162238;
}
QFrame#studioCard QTableCornerButton::section {
    background: #162238;
    border-color: #2C3E59;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #0F192A;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #405573;
    border-radius: 4px;
}
QToolTip {
    background: #18263C;
    color: #F4F7FB;
    border: 1px solid #405573;
}
"""


def stylesheet_for(theme: str) -> str:
    if theme not in {'light', 'dark'}:
        raise ValueError(f'Неизвестная тема: {theme}')
    return load_styles() + ('\n' + DARK_STYLES if theme == 'dark' else '')


def apply_studio_theme(theme: str, app: QApplication | None = None) -> None:
    application = app or QApplication.instance()
    if application is None:
        return
    application.setProperty('studioTheme', theme)
    application.setStyleSheet(stylesheet_for(theme))
