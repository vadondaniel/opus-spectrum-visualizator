from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QHBoxLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from utils.plot import plot_peak_analysis


class PeakAnalysisDialog(QDialog):
    def __init__(self, combined_list, wavelength_ranges,
                 display_type="both", smoothing="gaussian",
                 smoothing_param=5.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Peak Analysis")
        self.resize(900, 600)
        
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinMaxButtonsHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        self.combined_list = combined_list
        self.wavelength_ranges = wavelength_ranges

        # Store settings
        self.display_type = display_type
        self.smoothing_method = smoothing
        self.smoothing_param = smoothing_param
        self.baseline_method = "none"

        # --- Matplotlib figure ---
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # --- Layout ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        # --- Settings row ---
        settings_group = QGroupBox("Plot Settings")
        settings_layout = QHBoxLayout()

        # Display type dropdown
        self.display_type_dropdown = QComboBox()
        self.display_type_dropdown.addItems(["Both", "Raw", "Smoothed"])
        self.display_type_dropdown.setCurrentText(self.display_type.capitalize())
        self.display_type_dropdown.setToolTip("Select how to plot")
        self.display_type_dropdown.setFixedWidth(100)
        self.display_type_dropdown.currentTextChanged.connect(self._on_display_changed)

        # Smoothing method dropdown
        self.smoothing_method_dropdown = QComboBox()
        self.smoothing_method_dropdown.addItems(["gaussian", "loess", "spline", "boxcar"])
        self.smoothing_method_dropdown.setCurrentText(self.smoothing_method)
        self.smoothing_method_dropdown.setToolTip("Select the method of smoothing")
        self.smoothing_method_dropdown.setFixedWidth(110)
        self.smoothing_method_dropdown.currentTextChanged.connect(self._update_smoothing_param_ui)

        # Smoothing parameter spinbox
        self.smoothing_param_label = QLabel()
        self.smoothing_param_spin = QDoubleSpinBox()
        self.smoothing_param_spin.setDecimals(3)
        self.smoothing_param_spin.setRange(0.0, 1e6)
        self.smoothing_param_spin.setSingleStep(0.1)
        self.smoothing_param_spin.setValue(self.smoothing_param)
        self.smoothing_param_spin.setFixedWidth(110)
        self.smoothing_param_spin.valueChanged.connect(self._update_smoothing_param_value)
        
        # Baseline method dropdown
        self.baseline_method_dropdown = QComboBox()
        self.baseline_method_dropdown.addItems(["none", "equal", "zero"])
        self.baseline_method_dropdown.setCurrentText(self.baseline_method)
        self.baseline_method_dropdown.setToolTip("Select the method of baseline correction")
        self.baseline_method_dropdown.setFixedWidth(110)
        self.baseline_method_dropdown.currentTextChanged.connect(self._on_baseline_changed)

        # Assemble settings layout
        settings_layout.addWidget(QLabel("Display:"))
        settings_layout.addWidget(self.display_type_dropdown)
        settings_layout.addSpacing(10)
        settings_layout.addWidget(QLabel("Smoothing Method:"))
        settings_layout.addWidget(self.smoothing_method_dropdown)
        settings_layout.addSpacing(10)
        settings_layout.addWidget(self.smoothing_param_label)
        settings_layout.addWidget(self.smoothing_param_spin)
        self._update_smoothing_param_ui(self.smoothing_method)
        settings_layout.addSpacing(10)
        settings_layout.addWidget(QLabel("Baseline Correction:"))
        settings_layout.addWidget(self.baseline_method_dropdown)
        settings_layout.addStretch()

        settings_group.setLayout(settings_layout)
        settings_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(settings_group)

        self.setLayout(main_layout)

        # --- Initial plot ---
        QTimer.singleShot(0, self._plot)

    def _on_display_changed(self, text):
        self.display_type = text.lower()
        self._plot()

    def _update_smoothing_param_ui(self, method: str):
        """Adjust label, ranges, and defaults depending on smoothing method."""
        method = method.lower()
        self.smoothing_method = method

        if method == "gaussian":
            self.smoothing_param_label.setText("Bandwidth (K):")
            self.smoothing_param_spin.setRange(0.01, 1e5)
            self.smoothing_param_spin.setValue(5.0)
            self.smoothing_param_spin.setSingleStep(0.1)
            self.smoothing_param_spin.setToolTip("Set the level of smoothing (bandwidth in K)")
        elif method == "loess":
            self.smoothing_param_label.setText("Frac (0–1):")
            self.smoothing_param_spin.setRange(0.01, 1.0)
            self.smoothing_param_spin.setValue(0.25)
            self.smoothing_param_spin.setSingleStep(0.01)
            self.smoothing_param_spin.setToolTip("Set the level of smoothing (fraction of points)")
        elif method == "spline":
            self.smoothing_param_label.setText("Strength (0–1):")
            self.smoothing_param_spin.setRange(0.0, 1.0)
            self.smoothing_param_spin.setValue(0.2)
            self.smoothing_param_spin.setSingleStep(0.01)
            self.smoothing_param_spin.setToolTip("Set the level of smoothing (strength 0-1)")
        elif method == "boxcar":
            self.smoothing_param_label.setText("Window (K):")
            self.smoothing_param_spin.setRange(0.01, 1e5)
            self.smoothing_param_spin.setValue(5.0)
            self.smoothing_param_spin.setSingleStep(0.5)
            self.smoothing_param_spin.setToolTip("Set the window size for moving average")

        self._plot()

    def _update_smoothing_param_value(self, value):
        self.smoothing_param = value
        self._plot()
        
    def _on_baseline_changed(self, text):
        self.baseline_method = text.lower()
        self._plot()

    def _plot(self):
        """Call the plotting function safely, clearing axes and handling smoothed-only legend."""
        self.ax.clear()

        plot_peak_analysis(
            combined_list=self.combined_list,
            wavelength_ranges=self.wavelength_ranges,
            display_type=self.display_type,
            smoothing=self.smoothing_method,
            smoothing_param=self.smoothing_param,
            baseline_correction_mode=self.baseline_method,
            eval_on_grid=True,
            grid_points=300,
            ax=self.ax
        )

        self.canvas.draw()
