from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QHBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from utils.plot import plot_spectra


class SpectraPlotDialog(QDialog):
    def __init__(self, spectra_data, start_index=0, end_index=10,
                 normalize=False, index_offset=False, save_callback=None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Spectra Plot")
        self.resize(900, 600)

        self.spectra_data = spectra_data
        self.start_index = start_index
        self.end_index = end_index
        self.normalize = normalize
        self.index_offset = index_offset
        self.save_callback = save_callback

        # --- Matplotlib figure/canvas ---
        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.canvas = FigureCanvas(self.fig)

        # Optional toolbar (zoom/pan/reset)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Button row (Qt buttons, not matplotlib!)
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("Insert Selected Range into Peak Analysis")
        self.close_btn = QPushButton("Close")

        self.save_btn.clicked.connect(self._on_save)
        self.close_btn.clicked.connect(self.close)

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        # Plot spectra
        self._plot_spectra()

    def _plot_spectra(self):
        self.ax.clear()
        plot_spectra(self.ax, self.spectra_data,
                     start_index=self.start_index,
                     end_index=self.end_index,
                     normalize=self.normalize)
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_save(self):
        xmin, xmax = self.ax.get_xlim()
        if self.save_callback:
            self.save_callback((xmin, xmax))
        else:
            print(f"Saved range: {xmin:.2f} – {xmax:.2f}")
