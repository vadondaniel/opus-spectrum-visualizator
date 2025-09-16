from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout, QSizePolicy
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QUrl
import numpy as np
import os
from ui.q_range_slider import QRangeSlider
from utils.plot import plot_3d


class ThreeDPlotDialog(QDialog):
    def __init__(self, combined_data, plot_type, cmap, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(plot_type + " Plot")

        self.combined_data = combined_data
        self.plot_type = plot_type
        self.cmap = cmap

        self.wavenumbers = np.array(combined_data[0]["wavenumbers"])
        self.temperatures = np.array([e["temperature"] for e in combined_data])

        main_layout = QVBoxLayout(self)

        # Web view (plot first)
        self.view = QWebEngineView()
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.view, stretch=1)

        # --- Wavenumber range group ---
        wn_group = QGroupBox("Wavenumber range (cm⁻¹)")
        wn_layout = QHBoxLayout()
        wn_layout.setContentsMargins(4, 4, 4, 4)
        wn_layout.setSpacing(6)

        self.wn_slider = QRangeSlider(float(self.wavenumbers.min()),
                                      float(self.wavenumbers.max()))
        self.wn_label = QLabel("—")
        self.wn_label.setFixedWidth(120)  # keeps it tight
        self.wn_slider.rangeChanged.connect(self.update_plot)

        wn_layout.addWidget(self.wn_slider, stretch=1)
        wn_layout.addWidget(self.wn_label, stretch=0)
        wn_group.setLayout(wn_layout)

        main_layout.addWidget(wn_group)

        # --- Temperature range group ---
        t_group = QGroupBox("Temperature range (K)")
        t_layout = QHBoxLayout()
        t_layout.setContentsMargins(4, 4, 4, 4)
        t_layout.setSpacing(6)

        self.t_slider = QRangeSlider(float(self.temperatures.min()),
                                     float(self.temperatures.max()))
        self.t_label = QLabel("—")
        self.t_label.setFixedWidth(120)
        self.t_slider.rangeChanged.connect(self.update_plot)

        t_layout.addWidget(self.t_slider, stretch=1)
        t_layout.addWidget(self.t_label, stretch=0)
        t_group.setLayout(t_layout)

        main_layout.addWidget(t_group)

        self.update_plot()

    def update_plot(self, *_):
        wn_min, wn_max = self.wn_slider.lowerValue(), self.wn_slider.upperValue()
        t_min, t_max = self.t_slider.lowerValue(), self.t_slider.upperValue()

        self.wn_label.setText(f"{wn_min:.1f} – {wn_max:.1f} cm⁻¹")
        self.t_label.setText(f"{t_min:.1f} – {t_max:.1f} K")

        wn = self.wavenumbers
        wn_mask = (wn >= wn_min) & (wn <= wn_max)

        filtered_data = []
        for entry in self.combined_data:
            if t_min <= entry["temperature"] <= t_max:
                filtered_data.append({
                    "temperature": entry["temperature"],
                    "wavenumbers": wn[wn_mask],
                    "absorbance": entry["absorbance"][wn_mask],
                })

        if not filtered_data:
            return

        tmp_html = plot_3d(filtered_data, self.plot_type, self.cmap)
        self.view.setUrl(QUrl.fromLocalFile(os.path.abspath(tmp_html)))
