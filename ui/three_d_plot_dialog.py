from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout,
    QSizePolicy, QComboBox, QLineEdit, QMessageBox, QCheckBox
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QUrl
import numpy as np
import os
from ui.q_range_slider import QRangeSlider
from utils.plot import plot_3d


class ThreeDPlotDialog(QDialog):
    def __init__(self, combined_data, plot_type, cmap,
                 smoothing_factor=None,
                 baseline_enabled=False,
                 baseline_min=None,
                 baseline_max=None,
                 parent=None):
        super().__init__(parent)

        self.setWindowTitle(plot_type + " Plot")
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinMaxButtonsHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        self.combined_data = combined_data
        self.plot_type = plot_type
        self.cmap = cmap
        self.smoothing_factor = smoothing_factor

        # Baseline correction state
        self.baseline_enabled = bool(baseline_enabled)
        self.baseline_min = baseline_min
        self.baseline_max = baseline_max

        self.wavenumbers = np.array(combined_data[0]["wavenumbers"])
        self.temperatures = np.array([e["temperature"] for e in combined_data])

        main_layout = QVBoxLayout(self)

        # --- Web view ---
        self.view = QWebEngineView()
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.view, stretch=1)

        # Force repaint after load to avoid blank view
        def on_load_finished(ok):
            if ok:
                self.view.resize(self.view.width() + 1, self.view.height() + 1)
                self.view.resize(self.view.width() - 1, self.view.height() - 1)
        self.view.loadFinished.connect(on_load_finished)

        # --- Wavenumber range group ---
        wn_group = QGroupBox("Wavenumber range (cm⁻¹)")
        wn_layout = QHBoxLayout()

        self.wn_slider = QRangeSlider(float(self.wavenumbers.min()),
                                      float(self.wavenumbers.max()))

        self.wn_min_input = QLineEdit()
        self.wn_max_input = QLineEdit()
        for inp in (self.wn_min_input, self.wn_max_input):
            inp.setFixedWidth(70)
            inp.setAlignment(Qt.AlignmentFlag.AlignRight)

        def update_wn_inputs():
            self.wn_min_input.setText(f"{self.wn_slider.lowerValue():.1f}")
            self.wn_max_input.setText(f"{self.wn_slider.upperValue():.1f}")

        def update_wn_slider():
            try:
                min_val = float(self.wn_min_input.text())
                max_val = float(self.wn_max_input.text())
                if min_val <= max_val:
                    self.wn_slider.setLowerValue(min_val)
                    self.wn_slider.setUpperValue(max_val)
                    self.update_plot()
            except ValueError:
                pass

        self.wn_slider.rangeChanged.connect(
            lambda *_: (update_wn_inputs(), self.update_plot()))
        self.wn_min_input.editingFinished.connect(update_wn_slider)
        self.wn_max_input.editingFinished.connect(update_wn_slider)

        wn_layout.addWidget(self.wn_slider, stretch=1)
        wn_layout.addWidget(self.wn_min_input)
        wn_layout.addWidget(QLabel("–"))
        wn_layout.addWidget(self.wn_max_input)
        wn_group.setLayout(wn_layout)
        main_layout.addWidget(wn_group)
        update_wn_inputs()

        # --- Temperature range group ---
        t_group = QGroupBox("Temperature range (K)")
        t_layout = QHBoxLayout()

        self.t_slider = QRangeSlider(float(self.temperatures.min()),
                                     float(self.temperatures.max()))

        self.t_min_input = QLineEdit()
        self.t_max_input = QLineEdit()
        for inp in (self.t_min_input, self.t_max_input):
            inp.setFixedWidth(70)
            inp.setAlignment(Qt.AlignmentFlag.AlignRight)

        def update_t_inputs():
            self.t_min_input.setText(f"{self.t_slider.lowerValue():.1f}")
            self.t_max_input.setText(f"{self.t_slider.upperValue():.1f}")

        def update_t_slider():
            try:
                min_val = float(self.t_min_input.text())
                max_val = float(self.t_max_input.text())
                if min_val <= max_val:
                    self.t_slider.setLowerValue(min_val)
                    self.t_slider.setUpperValue(max_val)
                    self.update_plot()
            except ValueError:
                pass

        self.t_slider.rangeChanged.connect(
            lambda *_: (update_t_inputs(), self.update_plot()))
        self.t_min_input.editingFinished.connect(update_t_slider)
        self.t_max_input.editingFinished.connect(update_t_slider)

        t_layout.addWidget(self.t_slider, stretch=1)
        t_layout.addWidget(self.t_min_input)
        t_layout.addWidget(QLabel("–"))
        t_layout.addWidget(self.t_max_input)
        t_group.setLayout(t_layout)
        main_layout.addWidget(t_group)
        update_t_inputs()

        # --- Plot settings row ---
        settings_group = QGroupBox("Plot Settings")
        settings_layout = QHBoxLayout()

        self.plot_type_dropdown = QComboBox()
        self.plot_type_dropdown.addItems(
            ["Surface", "Scatter", "Contour", "Heatmap"])
        self.plot_type_dropdown.setCurrentText(plot_type)
        self.plot_type_dropdown.currentTextChanged.connect(
            self._on_plot_type_changed)

        self.cmap_dropdown = QComboBox()
        self.cmap_dropdown.addItems(
            ["plasma", "viridis", "inferno", "cividis", "magma"])
        self.cmap_dropdown.setCurrentText(cmap)
        self.cmap_dropdown.currentTextChanged.connect(self._on_cmap_changed)

        self.smoothing_input = QLineEdit()
        self.smoothing_input.setPlaceholderText("Smoothing factor")
        self.smoothing_input.setFixedWidth(110)
        self.smoothing_input.setToolTip("Enter smoothing factor (optional)")
        self.smoothing_input.textChanged.connect(self.update_plot)

        # Pre-fill smoothing input if provided
        if self.smoothing_factor:
            self.smoothing_input.setText(str(self.smoothing_factor))

        # Baseline correction controls (inline with plot settings)
        self.baseline_checkbox = QCheckBox("Enable")
        self.baseline_checkbox.setChecked(self.baseline_enabled)
        self.baseline_checkbox.toggled.connect(self.update_plot)

        self.bl_min_input = QLineEdit()
        self.bl_min_input.setPlaceholderText("Min wn")
        self.bl_min_input.setFixedWidth(90)
        self.bl_min_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.bl_max_input = QLineEdit()
        self.bl_max_input.setPlaceholderText("Max wn")
        self.bl_max_input.setFixedWidth(90)
        self.bl_max_input.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Prefill with passed values or defaults from global wn range
        if self.baseline_min is None:
            self.baseline_min = float(self.wavenumbers.min())
        if self.baseline_max is None:
            self.baseline_max = float(self.wavenumbers.max())

        try:
            self.bl_min_input.setText(f"{float(self.baseline_min):.1f}")
        except Exception:
            self.bl_min_input.setText("")
        try:
            self.bl_max_input.setText(f"{float(self.baseline_max):.1f}")
        except Exception:
            self.bl_max_input.setText("")

        def _on_bl_inputs_changed():
            self.update_plot()

        self.bl_min_input.editingFinished.connect(_on_bl_inputs_changed)
        self.bl_max_input.editingFinished.connect(_on_bl_inputs_changed)

        settings_layout.addWidget(QLabel("Type:"))
        settings_layout.addWidget(self.plot_type_dropdown)
        settings_layout.addSpacing(10)
        settings_layout.addWidget(QLabel("Colormap:"))
        settings_layout.addWidget(self.cmap_dropdown)
        settings_layout.addSpacing(10)
        settings_layout.addWidget(QLabel("Smoothing:"))
        settings_layout.addWidget(self.smoothing_input)
        settings_layout.addStretch()
        settings_group.setLayout(settings_layout)

        # Baseline correction in its own group, placed next to Plot Settings
        bl_group = QGroupBox("Baseline Correction (3D)")
        bl_layout = QHBoxLayout()
        bl_layout.addWidget(self.baseline_checkbox)
        bl_layout.addSpacing(8)
        bl_layout.addWidget(QLabel("Wn min:"))
        bl_layout.addWidget(self.bl_min_input)
        bl_layout.addWidget(QLabel("Wn max:"))
        bl_layout.addWidget(self.bl_max_input)
        bl_layout.addStretch()
        bl_group.setLayout(bl_layout)

        settings_row = QHBoxLayout()
        settings_row.addWidget(settings_group)
        settings_row.addWidget(bl_group)
        main_layout.addLayout(settings_row)

        self.update_plot()

    def _on_plot_type_changed(self, value):
        self.plot_type = value
        self.update_plot()

    def _on_cmap_changed(self, value):
        self.cmap = value
        self.update_plot()

    def update_plot(self, *_):
        wn_min, wn_max = self.wn_slider.lowerValue(), self.wn_slider.upperValue()
        t_min, t_max = self.t_slider.lowerValue(), self.t_slider.upperValue()

        wn = self.wavenumbers
        wn_mask = (wn >= wn_min) & (wn <= wn_max)

        # Determine whether to apply baseline correction and get min/max
        apply_baseline = self.baseline_checkbox.isChecked()
        bl_min_val = None
        bl_max_val = None
        if apply_baseline:
            try:
                bl_min_val = float(self.bl_min_input.text())
                bl_max_val = float(self.bl_max_input.text())
                if not (np.isfinite(bl_min_val) and np.isfinite(bl_max_val)):
                    apply_baseline = False
                elif bl_min_val == bl_max_val:
                    QMessageBox.warning(self, "Baseline",
                                        "Baseline min and max must differ.")
                    apply_baseline = False
            except Exception:
                apply_baseline = False

        # Prepare data (optionally baseline-corrected) then filter by ranges
        source_data = self.combined_data
        if apply_baseline:
            source_data = self._baseline_correct_dataset(self.combined_data, bl_min_val, bl_max_val)

        filtered_data = []
        for entry in source_data:
            if t_min <= entry["temperature"] <= t_max:
                filtered_data.append({
                    "temperature": entry["temperature"],
                    "wavenumbers": wn[wn_mask],
                    "absorbance": entry["absorbance"][wn_mask],
                })

        if not filtered_data:
            return

        # Parse smoothing factor
        smoothing_val = None
        if self.smoothing_input.text().strip():
            try:
                smoothing_val = int(self.smoothing_input.text())
                if smoothing_val <= 0:
                    smoothing_val = None
            except ValueError:
                QMessageBox.warning(self, "Smoothing",
                                    "Invalid smoothing value.")
                return

        tmp_html = plot_3d(filtered_data,
                           self.plot_type,
                           self.cmap,
                           smoothing_factor=smoothing_val)
        self.view.setUrl(QUrl.fromLocalFile(os.path.abspath(tmp_html)))

    # --- helpers ---
    def _baseline_correct_dataset(self, combined_data, wn_min, wn_max):
        """Return a new combined_data list where each spectrum is corrected by a linear baseline
        defined by absorbance values at wn_min and wn_max for that spectrum, subtracting the
        baseline from all absorbance points.
        """
        if wn_min == wn_max:
            return combined_data

        corrected = []
        for entry in combined_data:
            wn = np.asarray(entry["wavenumbers"], dtype=float)
            ab = np.asarray(entry["absorbance"], dtype=float)

            # Interpolate absorbance at the two anchor wavenumbers
            y1 = self._interp_at(wn, ab, wn_min)
            y2 = self._interp_at(wn, ab, wn_max)

            # Compute baseline line and subtract
            slope = (y2 - y1) / (wn_max - wn_min)
            baseline = y1 + slope * (wn - wn_min)
            ab_corr = ab - baseline

            corrected.append({
                "temperature": entry["temperature"],
                "wavenumbers": wn,
                "absorbance": ab_corr,
            })

        return corrected

    @staticmethod
    def _interp_at(x, y, x0):
        """Interpolate y(x) at x0 for monotonic x that may be ascending or descending.
        If x0 is out of bounds, returns the nearest endpoint value (np.interp behavior).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0:
            return float('nan')
        if x.size == 1:
            return float(y[0])
        if x[0] <= x[-1]:
            return float(np.interp(x0, x, y))
        # descending axis: reverse for interpolation
        return float(np.interp(x0, x[::-1], y[::-1]))
