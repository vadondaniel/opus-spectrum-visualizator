#!/usr/bin/env python3
from pathlib import Path
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QLineEdit, QMessageBox,
    QComboBox, QGroupBox, QDoubleSpinBox, QGridLayout, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon
import numpy as np

from ui.info_window import InfoWindow
from ui.spectra_plot_dialog import SpectraPlotDialog
from ui.three_d_plot_dialog import ThreeDPlotDialog
from ui.peak_analysis_dialog import PeakAnalysisDialog
from data_processing.spectrum import process_spectrum_data
from data_processing.temperature import process_temperature_data
from data_processing.combined_data import combine_temperature_and_spectrum_data
from utils.processing_thread import ProcessingThread
from utils.smoothing import estimate_bandwidth
from utils.export_csv import (
    export_peak_analysis_csv, export_combined_data_csv, export_spectra_csv
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._init_state()
        self._setup_window()
        self._setup_ui()

    # ------------------------
    # State / Window Setup
    # ------------------------
    def _init_state(self):
        """Initialize application state."""
        self.spectrum_list = []
        self.temperature_list_filtered = []
        self.combined_list = []

        self.spectrum_path = None
        self.temp_file = None

    def _setup_window(self):
        """Basic window properties."""
        self.setWindowTitle("Spectrum & Temperature Data Processor")
        self.setGeometry(150, 150, 720, 460)

    # ------------------------
    # UI Setup
    # ------------------------
    def _setup_ui(self):
        """Create the full UI layout."""
        self._create_menu_bar()

        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Add each group box section
        main_layout.addWidget(self._create_file_selection_group())
        main_layout.addWidget(self._create_processing_group())

        abs_spec_layout = QHBoxLayout()
        abs_spec_layout.addWidget(self._create_absorbance_group())
        abs_spec_layout.addWidget(self._create_spectra_group())
        main_layout.addLayout(abs_spec_layout)

        main_layout.addWidget(self._create_visualization_group())
        main_layout.addWidget(self._create_peak_analysis_group())

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # ------------------------
    # Menu Bar
    # ------------------------
    def _create_menu_bar(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")

        # How To Use
        howto_action = help_menu.addAction("How To Use")
        howto_action.setToolTip(
            "Show instructions on how to use the application")
        howto_action.triggered.connect(self.show_howto)

        # Credits
        credits_action = help_menu.addAction("Credits")
        credits_action.setToolTip("Show credits and acknowledgments")
        credits_action.triggered.connect(self.show_credits)

        return menubar

    # ------------------------
    # File Selection
    # ------------------------
    def _create_file_selection_group(self):
        group = QGroupBox("Data Selection")
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(12, 8, 12, 8)

        # Spectrum row
        spec_layout = QHBoxLayout()
        self.spectrum_label = QLabel("No folder selected")
        self.spectrum_label.setStyleSheet("color: gray;")
        self.spectrum_btn = QPushButton("Browse")
        self.spectrum_btn.setFixedWidth(100)
        self.spectrum_btn.setToolTip(
            "Select the folder containing spectrum files")
        self.spectrum_btn.clicked.connect(self.select_spectrum_folder)
        spec_layout.addWidget(QLabel("Spectrum:"))
        spec_layout.addWidget(self.spectrum_label)
        spec_layout.addStretch()
        spec_layout.addWidget(self.spectrum_btn)

        # Temperature row
        temp_layout = QHBoxLayout()
        self.temp_label = QLabel("No file selected")
        self.temp_label.setStyleSheet("color: gray;")
        self.temp_btn = QPushButton("Browse")
        self.temp_btn.setFixedWidth(100)
        self.temp_btn.setToolTip("Select the temperature data file")
        self.temp_btn.clicked.connect(self.select_temp_file)
        temp_layout.addWidget(QLabel("Temperature:"))
        temp_layout.addWidget(self.temp_label)
        temp_layout.addStretch()
        temp_layout.addWidget(self.temp_btn)

        layout.addLayout(spec_layout)
        layout.addLayout(temp_layout)
        group.setLayout(layout)
        return group

    # ------------------------
    # Processing
    # ------------------------
    def _create_processing_group(self):
        group = QGroupBox("Processing")
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(12, 8, 12, 8)

        # Start button
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setFixedWidth(140)
        self.process_btn.setToolTip(
            "Start processing spectrum and temperature data")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn,
                         alignment=Qt.AlignmentFlag.AlignCenter)

        # Progress rows
        spec_row, self.spec_progress = self._create_progress_row("Spectrum:")
        temp_row, self.temp_progress = self._create_progress_row(
            "Temperature:")
        comb_row, self.combine_progress = self._create_progress_row("Combine:")
        layout.addLayout(spec_row)
        layout.addLayout(temp_row)
        layout.addLayout(comb_row)

        group.setLayout(layout)
        return group

    def _create_progress_row(self, label_text):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(100)
        progress = QProgressBar()
        progress.setFormat("%p%")
        row.addWidget(label)
        row.addWidget(progress)
        return row, progress

    # ------------------------
    # Absorbance
    # ------------------------
    def _create_absorbance_group(self):
        group = QGroupBox("Absorbance Range Filter")
        layout = QHBoxLayout()

        self.abs_min_input = QLineEdit()
        self.abs_min_input.setPlaceholderText("Min (e.g., 0)")
        self.abs_min_input.setFixedWidth(80)

        self.abs_max_input = QLineEdit()
        self.abs_max_input.setPlaceholderText("Max (e.g., 1)")
        self.abs_max_input.setFixedWidth(80)

        layout.addWidget(QLabel("Min:"))
        layout.addWidget(self.abs_min_input)
        layout.addWidget(QLabel("Max:"))
        layout.addWidget(self.abs_max_input)
        layout.addStretch()
        group.setLayout(layout)
        return group

    # ------------------------
    # Spectra
    # ------------------------
    def _create_spectra_group(self):
        group = QGroupBox("Spectra")
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 6, 10, 6)

        # Export button
        self.export_csv_btn = QPushButton("Export as CSV")
        self.export_csv_btn.setFixedWidth(110)
        self.export_csv_btn.clicked.connect(
            self._on_export_spectra_csv_clicked)
        self.export_csv_btn.setEnabled(False)

        # Plot button
        self.plot_spectra_btn = QPushButton("Plot Spectra")
        self.plot_spectra_btn.setFixedWidth(110)
        self.plot_spectra_btn.clicked.connect(self._on_plot_spectra)
        self.plot_spectra_btn.setEnabled(False)

        # Assemble
        layout.addStretch()
        layout.addWidget(self.export_csv_btn)
        layout.addWidget(self.plot_spectra_btn)

        group.setLayout(layout)
        return group

    # ------------------------
    # Visualization
    # ------------------------
    def _create_visualization_group(self):
        group = QGroupBox("3D Plotting")
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 6, 10, 6)

        # Plot type
        self.plot_type_dropdown = QComboBox()
        self.plot_type_dropdown.addItems(
            ["Surface", "Scatter", "Contour", "Heatmap"])
        self.plot_type_dropdown.setFixedWidth(100)
        self.plot_type_dropdown.setToolTip("Select the 3D plot type")

        # Colormap
        self.cmap_dropdown = QComboBox()
        self.cmap_dropdown.addItems(
            ["plasma", "viridis", "inferno", "cividis", "magma"])
        self.cmap_dropdown.setFixedWidth(80)
        self.cmap_dropdown.setToolTip("Select the color map for the plot")

        # Smoothing
        self.smoothing_input = QLineEdit()
        self.smoothing_input.setPlaceholderText("Smoothing factor")
        self.smoothing_input.setFixedWidth(110)
        self.smoothing_input.setToolTip("Enter smoothing factor (optional)")

        # Buttons
        self.combined_export_btn = QPushButton("Export as CSV")
        self.combined_export_btn.setFixedWidth(100)
        self.combined_export_btn.clicked.connect(
            self._on_export_combined_csv_clicked)
        self.combined_export_btn.setEnabled(False)

        self.plot_btn = QPushButton("Plot 3D")
        self.plot_btn.setFixedWidth(90)
        self.plot_btn.clicked.connect(self._plot_3d)
        self.plot_btn.setEnabled(False)

        # Assemble
        layout.addWidget(QLabel("Type:"))
        layout.addWidget(self.plot_type_dropdown)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Colormap:"))
        layout.addWidget(self.cmap_dropdown)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Smoothing:"))
        layout.addWidget(self.smoothing_input)
        layout.addStretch()
        layout.addWidget(self.combined_export_btn)
        layout.addWidget(self.plot_btn)

        group.setLayout(layout)
        return group

    # ------------------------
    # Peak Analysis
    # ------------------------
    def _create_peak_analysis_group(self):
        group = QGroupBox("Peak Analysis")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.setContentsMargins(10, 6, 10, 6)

        # Row 0: Range(s)
        self.wavelength_range_input = QLineEdit()
        self.wavelength_range_input.setPlaceholderText(
            "950-1050, 2450-2550, 3050-3150, ...")
        self.wavelength_range_input.setToolTip(
            "Enter range(s) separated by commas. Example: 950-1050, 2450-2550")

        self.peak_export_btn = QPushButton("Export as CSV")
        self.peak_export_btn.setFixedWidth(100)
        self.peak_export_btn.clicked.connect(self._on_export_peak_csv_clicked)
        self.peak_export_btn.setEnabled(False)

        grid.addWidget(QLabel("Range(s):"), 0, 0)
        grid.addWidget(self.wavelength_range_input, 0, 1, 1, 6)
        grid.addWidget(self.peak_export_btn, 0, 7)

        # Row 1: Display / Method / Param / Button
        self.display_type_dropdown = QComboBox()
        self.display_type_dropdown.addItems(["Both", "Raw", "Smoothed"])
        self.display_type_dropdown.setToolTip("Select how to plot")
        self.display_type_dropdown.setFixedWidth(100)

        self.smoothing_method_dropdown = QComboBox()
        self.smoothing_method_dropdown.addItems(
            ["gaussian", "loess", "spline", "boxcar"])
        self.smoothing_method_dropdown.setToolTip(
            "Select the method of smoothing")
        self.smoothing_method_dropdown.setFixedWidth(110)

        self.smoothing_param_spin = QDoubleSpinBox()
        self.smoothing_param_spin.setDecimals(3)
        self.smoothing_param_spin.setRange(0.0, 1e6)
        self.smoothing_param_spin.setSingleStep(0.1)
        self.smoothing_param_spin.setValue(5.0)
        self.smoothing_param_spin.setFixedWidth(110)

        self.peak_btn = QPushButton("Peak Analysis")
        self.peak_btn.setFixedWidth(100)
        self.peak_btn.clicked.connect(self._plot_absorption)
        self.peak_btn.setEnabled(False)

        # Place widgets
        grid.addWidget(QLabel("Display:"), 1, 0)
        grid.addWidget(self.display_type_dropdown, 1, 1)
        self.smoothing_method_label = QLabel("Smoothing Method:")
        grid.addWidget(self.smoothing_method_label, 1, 2)
        grid.addWidget(self.smoothing_method_dropdown, 1, 3)
        self.smoothing_param_label = QLabel("Param:")
        grid.addWidget(self.smoothing_param_label, 1, 4)
        grid.addWidget(self.smoothing_param_spin, 1, 5)

        spacer = QSpacerItem(
            1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        grid.addItem(spacer, 1, 6)
        grid.addWidget(self.peak_btn, 1, 7)

        group.setLayout(grid)

        # Connect UI changes
        self.smoothing_method_dropdown.currentTextChanged.connect(
            self._update_smoothing_param_ui)
        self.display_type_dropdown.currentTextChanged.connect(
            self._on_display_changed)

        # Initialize
        self._update_smoothing_param_ui(
            self.smoothing_method_dropdown.currentText())
        self._on_display_changed(self.display_type_dropdown.currentText())

        return group

    # ------------------------
    # Info Windows
    # ------------------------
    def show_info(self, title, text):
        self.info_win = InfoWindow(title, text)
        self.info_win.show()

    def show_howto(self):
        self.show_info("How To Use", """
            <h2>How To Use</h2>
            <ol>
                <li>Select a <b>folder</b> containing <b>OPUS files</b> and a <b>Temperature file</b>.</li>
                <li>Click <b>Start Processing</b> — Spectrum and Temperature will run in parallel, then Combine.</li>
                <li>Use <b>Visualization</b> to plot 3D (Surface or Scatter, choose colormap, optional smoothing).</li>
                <li>Use <b>Range Comparison</b> to enter ranges (e.g. <code>950-1050, 3050-3150</code>) and compare absorption.</li>
            </ol>
        """)

    def show_credits(self):
        self.show_info("Credits", """
            <h2>Credits</h2>
            <p>This application was built using the following libraries:
            <ul>
                <li><a href="https://www.riverbankcomputing.com/software/pyqt/intro">PyQt6</a> (LGPL): GUI framework</li>
                <li>Numpy (BSD): Numerical computations</li>
                <li>Pandas (BSD): Data handling and CSV processing</li>
                <li>Plotly (MIT): Interactive 3D plots</li>
                <li><a href="https://github.com/spectrochempy/spectrochempy">SpectroChemPy</a> (CeCILL-B/C): OPUS file handling and spectroscopic data processing</li>
            </ul>
            <b>Developed by:</b> Dániel Vadon & Dr. Bálint Rubovszky</p>
            <p><b>Version:</b> 1.4 <a href="https://github.com/vadondaniel/opus-spectrum-visualizator">GitHub</a></p>
        """)

    # === Slots ===
    def select_spectrum_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Spectrum Folder")
        if folder:
            self.spectrum_path = folder
            self.spectrum_label.setText(f"{folder}")
        self._update_start_enabled()

    def select_temp_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Temperature File", "", "CSV/TXT Files (*.csv *.txt);;All Files (*)")
        if file:
            self.temp_file = file
            self.temp_label.setText(f"{file}")
        self._update_start_enabled()

    def _update_start_enabled(self):
        self.process_btn.setEnabled(
            bool(self.spectrum_path and self.temp_file))

    def start_processing(self):
        self.spectrum_list = []
        self.temperature_list_filtered = []
        self.combined_list = []

        # Reset progress bars
        self.spec_progress.setValue(0)
        self.temp_progress.setValue(0)
        self.combine_progress.setValue(0)

        # Disable buttons until finished
        self.export_csv_btn.setEnabled(False)
        self.plot_spectra_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.plot_btn.setEnabled(False)
        self.combined_export_btn.setEnabled(False)
        self.peak_btn.setEnabled(False)
        self.peak_export_btn.setEnabled(False)

        # Spectrum
        self.spectrum_thread = ProcessingThread(
            process_spectrum_data, self.spectrum_path)
        self.spectrum_thread.progress_update.connect(
            self.spec_progress.setValue)
        self.spectrum_thread.result_ready.connect(self._on_spectrum_done)
        self.spectrum_thread.error.connect(self._on_error)
        self.spectrum_thread.start()

        # Temperature
        self.temperature_thread = ProcessingThread(
            process_temperature_data, self.temp_file)
        self.temperature_thread.progress_update.connect(
            self.temp_progress.setValue)
        self.temperature_thread.result_ready.connect(self._on_temperature_done)
        self.temperature_thread.error.connect(self._on_error)
        self.temperature_thread.start()

    def _on_error(self, message: str):
        QMessageBox.critical(self, "Processing error", message)

    def _on_spectrum_done(self, result):
        self.spectrum_list = result
        self.spec_progress.setValue(100)
        if self.spectrum_list:
            self.export_csv_btn.setEnabled(True)
            self.plot_spectra_btn.setEnabled(True)
        self._combine_data()

    def _on_temperature_done(self, result):
        self.temperature_list_filtered = result
        self.temp_progress.setValue(100)
        self._combine_data()

    def _combine_data(self):
        if not self.spectrum_list or not self.temperature_list_filtered:
            return

        try:
            self.combined_list = combine_temperature_and_spectrum_data(
                self.temperature_list_filtered,
                self.spectrum_list,
                time_buffer=10
            ) or []
            self.combine_progress.setValue(100)
        except Exception as e:
            QMessageBox.critical(self, "Combine error", str(e))
            self.combined_list = []

        if not self.combined_list:
            QMessageBox.information(self, "Combine", "No combined entries.")
        else:
            # Enable visualization buttons
            self.plot_btn.setEnabled(True)
            self.combined_export_btn.setEnabled(True)
            self.peak_btn.setEnabled(True)
            self.peak_export_btn.setEnabled(True)

        # Re-enable processing start if inputs are still valid
        self._update_start_enabled()

    def get_filtered_spectrum_list(self):
        """Return spectra_list filtered by min/max absorbance if inputs are provided."""
        if not self.spectrum_list:
            return []

        try:
            min_val = float(self.abs_min_input.text()
                            ) if self.abs_min_input.text() else None
        except ValueError:
            min_val = None

        try:
            max_val = float(self.abs_max_input.text()
                            ) if self.abs_max_input.text() else None
        except ValueError:
            max_val = None

        if min_val is None and max_val is None:
            return self.spectrum_list

        filtered = []
        for entry in self.spectrum_list:
            new_entry = entry.copy()
            absorbance = np.asarray(new_entry["absorbance"])

            if min_val is not None:
                absorbance = np.maximum(absorbance, min_val)
            if max_val is not None:
                absorbance = np.minimum(absorbance, max_val)

            new_entry["absorbance"] = absorbance
            filtered.append(new_entry)

        return filtered

    def get_filtered_combined_list(self):
        """Return combined_list filtered by min/max absorbance if inputs are provided."""
        if not self.combined_list:
            return []

        try:
            min_val = float(self.abs_min_input.text()
                            ) if self.abs_min_input.text() else None
        except ValueError:
            min_val = None

        try:
            max_val = float(self.abs_max_input.text()
                            ) if self.abs_max_input.text() else None
        except ValueError:
            max_val = None

        if min_val is None and max_val is None:
            return self.combined_list

        filtered = []
        for entry in self.combined_list:
            new_entry = entry.copy()
            absorbance = new_entry["absorbance"]

            if min_val is not None:
                absorbance = np.maximum(absorbance, min_val)
            if max_val is not None:
                absorbance = np.minimum(absorbance, max_val)

            new_entry["absorbance"] = absorbance
            filtered.append(new_entry)

        return filtered

    def _update_smoothing_param_ui(self, method: str):
        """Adjust label, ranges and defaults for the smoothing parameter depending on method."""
        method = method.lower()

        if method == "gaussian":
            self.smoothing_param_label.setText("Bandwidth (K):")
            self.smoothing_param_spin.setRange(0.01, 1e5)
            self.smoothing_param_spin.setValue(estimate_bandwidth([e["temperature"] for e in self.get_filtered_combined_list()])
                                               if self.get_filtered_combined_list() else 5.0)
            self.smoothing_param_spin.setToolTip("Set the level of smoothing")

        elif method == "loess":
            self.smoothing_param_label.setText("Frac (0–1):")
            self.smoothing_param_spin.setRange(0.01, 1.0)
            self.smoothing_param_spin.setSingleStep(0.01)
            self.smoothing_param_spin.setToolTip("Set the level of smoothing")
            self.smoothing_param_spin.setValue(0.25)

        elif method == "spline":
            self.smoothing_param_label.setText("Strength (0–1):")
            self.smoothing_param_spin.setRange(0.0, 1.0)
            self.smoothing_param_spin.setSingleStep(0.01)
            self.smoothing_param_spin.setToolTip(
                "Set the level of smoothing (scale can be inconsistent between datasets of varying variances)")
            self.smoothing_param_spin.setValue(0.2)

        elif method == "boxcar":
            self.smoothing_param_label.setText("Window (K):")
            self.smoothing_param_spin.setRange(0.01, 1e5)
            self.smoothing_param_spin.setSingleStep(0.5)
            self.smoothing_param_spin.setToolTip(
                "Set the range the moving average should use for smoothing")
            self.smoothing_param_spin.setValue(5.0)

    def _on_display_changed(self, display_text: str):
        """If user chooses 'Raw' hide smoothing param since it won't be applied."""
        if display_text == "Raw":
            self.smoothing_param_spin.setVisible(False)
            self.smoothing_param_label.setVisible(False)
            self.smoothing_method_dropdown.setVisible(False)
            self.smoothing_method_label.setVisible(False)
        else:
            self.smoothing_param_spin.setVisible(True)
            self.smoothing_param_label.setVisible(True)
            self.smoothing_method_dropdown.setVisible(True)
            self.smoothing_method_label.setVisible(True)

    def _on_plot_spectra(self):
        if not self.get_filtered_spectrum_list():
            QMessageBox.warning(self, "Plot Spectra",
                                "No spectrum data to plot.")
            return

        def save_range_callback(x_range):
            xmin, xmax = sorted([round(x_range[0]), round(x_range[1])])
            new_range = f"{xmin}-{xmax}"

            current_text = self.wavelength_range_input.text().strip()
            if current_text:
                updated_text = f"{current_text}, {new_range}"
            else:
                updated_text = new_range

            self.wavelength_range_input.setText(updated_text)

        # Non-blocking popup (show instead of exec)
        self.spectra_dialog = SpectraPlotDialog(
            self.get_filtered_spectrum_list(),
            start_index=0,
            end_index=len(self.get_filtered_spectrum_list()) - 1,
            save_callback=save_range_callback,
            parent=self
        )
        self.spectra_dialog.show()

    def _plot_3d(self):
        data_to_plot = self.get_filtered_combined_list()
        if not data_to_plot:
            QMessageBox.warning(self, "Plot 3D", "No combined data to plot.")
            return

        cmap = self.cmap_dropdown.currentText()
        plot_type = self.plot_type_dropdown.currentText()

        smoothing_factor = self.smoothing_input.text() or None

        self.three_d_dialog = ThreeDPlotDialog(
            data_to_plot,
            plot_type,
            cmap,
            smoothing_factor,
            self
        )
        self.three_d_dialog.show()

    def _plot_absorption(self):
        if not self.get_filtered_combined_list():
            QMessageBox.warning(self, "Peak Analysis",
                                "No combined data to plot.")
            return

        s = self.wavelength_range_input.text().strip()
        if not s:
            QMessageBox.warning(self, "Peak Analysis",
                                "Please enter one or more wavelength ranges.")
            return

        try:
            ranges = []
            for part in s.split(","):
                part = part.strip()
                if not part:
                    continue
                start_str, end_str = part.replace(" ", "").split("-")
                start, end = float(start_str), float(end_str)
                ranges.append((start, end))
        except Exception:
            QMessageBox.warning(self, "Peak Analysis",
                                "Invalid format. Use e.g. 950-1050, 3050-3150")
            return

        display_type = self.display_type_dropdown.currentText().lower()
        smoothing_method = self.smoothing_method_dropdown.currentText().lower()
        smoothing_param = None
        if display_type != "raw":
            smoothing_param = float(self.smoothing_param_spin.value())

        self.peak_dialog = PeakAnalysisDialog(
            combined_list=self.get_filtered_combined_list(),
            wavelength_ranges=ranges,
            display_type=display_type,
            smoothing=smoothing_method,
            smoothing_param=smoothing_param,
            parent=self
        )
        self.peak_dialog.show()

    def _on_export_spectra_csv_clicked(self):
        if not self.get_filtered_spectrum_list():
            QMessageBox.warning(self, "Spectrum Data Export",
                                "No spectrum data to export.")
            return

        filepath = export_spectra_csv(self.get_filtered_spectrum_list(), self)

        msg = QMessageBox(self)
        msg.setWindowTitle("Export Result")

        if filepath:
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText(f"CSV successfully saved:\n{filepath}")
        else:
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Export cancelled.")

        msg.exec()

    def _on_export_combined_csv_clicked(self):
        if not self.get_filtered_combined_list():
            QMessageBox.warning(self, "Combined Data Export",
                                "No combined data to export.")
            return

        filepath = export_combined_data_csv(
            self.get_filtered_combined_list(), parent=self, format="matrix")

        msg = QMessageBox(self)
        msg.setWindowTitle("Export Result")

        if filepath:
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText(f"CSV successfully saved:\n{filepath}")
        else:
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Export cancelled.")

        msg.exec()

    def _on_export_peak_csv_clicked(self):
        if not self.get_filtered_combined_list():
            QMessageBox.warning(self, "Peak Analysis Export",
                                "No combined data to export.")
            return

        s = self.wavelength_range_input.text().strip()
        if not s:
            QMessageBox.warning(self, "Peak Analysis Export",
                                "Please enter one or more wavelength ranges.")
            return

        try:
            ranges = []
            for part in s.split(","):
                part = part.strip()
                if not part:
                    continue
                start_str, end_str = part.replace(" ", "").split("-")
                start, end = float(start_str), float(end_str)
                ranges.append((start, end))
        except Exception:
            QMessageBox.warning(self, "Peak Analysis Export",
                                "Invalid format. Use e.g. 950-1050, 3050-3150")
            return

        filepath = export_peak_analysis_csv(
            self.get_filtered_combined_list(), ranges, parent=self)

        msg = QMessageBox(self)
        msg.setWindowTitle("Export Result")

        if filepath:
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText(f"CSV successfully saved:\n{filepath}")
        else:
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Export cancelled.")

        msg.exec()

# ---------------------- Launch ----------------------


def resource_path(rel_path: str) -> str:
    """
    Return the absolute path to a bundled resource (works with PyInstaller).
    When running as source it uses the folder of the main script (sys.argv[0]),
    not the folder of this module (__file__).
    """
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(sys.argv[0]).resolve().parent
    return str(base / rel_path)


def launch_app():
    if sys.platform == "win32":
        try:
            import ctypes
            myappid = "com.ceavan.OpusSpectrumVisualizator"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                myappid)
        except Exception:
            pass

    app = QApplication(sys.argv)

    icon = QIcon(resource_path("icon.ico"))

    if not icon.isNull():
        app.setWindowIcon(icon)

    window = MainWindow()
    window.setWindowTitle("Opus Spectrum Visualizator")
    if not icon.isNull():
        window.setWindowIcon(icon)

    window.show()
    QTimer.singleShot(0, lambda: (window.raise_(), window.activateWindow()))
    sys.exit(app.exec())


if __name__ == '__main__':
    launch_app()
