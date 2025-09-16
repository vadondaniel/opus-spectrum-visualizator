from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QVBoxLayout, QPushButton, QDialog, QTextBrowser
)


class InfoWindow(QDialog):
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(500, 350)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowCloseButtonHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # QTextBrowser for rich text display
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(text)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setReadOnly(True)
        self.text_browser.setFrameStyle(
            QTextBrowser.Shape.NoFrame)  # remove borders
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                border: none;
                font-size: 12pt;
                background: transparent;
            }
        """)
        layout.addWidget(self.text_browser)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.setLayout(layout)
