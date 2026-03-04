from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QVBoxLayout, QPushButton, QDialog, QTextBrowser
)


class InfoDialog(QDialog):
    def __init__(self, title, text, parent=None, min_width=500, min_height=350):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(min_width, min_height)
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
