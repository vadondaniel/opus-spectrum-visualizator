import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from utils.plot import plot_peak_analysis


class PeakAnalysisDialog(QDialog):
    def __init__(self, combined_list, wavelength_ranges,
                 display_type="both", smoothing="gaussian",
                 smoothing_param=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Peak Analysis")
        self.resize(900, 600)

        # Matplotlib Figure + Canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        plot_peak_analysis(
            combined_list=combined_list,
            wavelength_ranges=wavelength_ranges,
            display_type=display_type,
            smoothing=smoothing,
            smoothing_param=smoothing_param,
            eval_on_grid=True,
            grid_points=300,
            ax=self.ax
        )
        self.canvas.draw()
