import os
from PyQt6.QtWidgets import QDialog, QVBoxLayout
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl


class ThreeDPlotDialog(QDialog):
    def __init__(self, html_file, plot_type, parent=None):
        super().__init__(parent)

        self.setWindowTitle(f"3D Absorbance {plot_type} Plot")
        self.resize(1200, 800)

        # Layout
        layout = QVBoxLayout(self)

        # Web view
        self.view = QWebEngineView()
        self.view.setUrl(QUrl.fromLocalFile(os.path.abspath(html_file)))

        # Force reload when finished loading
        def on_load_finished(ok):
            if ok:
                self.view.resize(self.view.width() + 1, self.view.height() + 1)
                self.view.resize(self.view.width() - 1, self.view.height() - 1)

        self.view.loadFinished.connect(on_load_finished)

        layout.addWidget(self.view)
