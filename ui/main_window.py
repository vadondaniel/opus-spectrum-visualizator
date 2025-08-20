#!/usr/bin/env python3
import sys
import os
from datetime import datetime
import tempfile
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QProgressBar, QLineEdit, QCheckBox, QStackedWidget, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QIntValidator, QGuiApplication
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go

from data_processing.spectrum import (
    interpolate_spectrum_data, process_spectrum_data
)
from data_processing.temperature import (
    interpolate_to_round_seconds, process_temperature_data
)
from data_processing.combined_data import (
    combine_temperature_and_spectrum_data, smooth_combined_by_temperature
)
from utils.plot import (
    plot_absorption_vs_temperature, plot_spectra, plot_temperature, plot_3d_surface
)


class ProcessingThread(QThread):
    """Generic worker thread that calls a processing function with a progress_callback.
    The processing functions in your code accept `progress_callback` which sometimes
    emits strings like '10%' or integers. This thread normalizes and emits ints.
    """
    progress_update = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, folder_path, *args, **kwargs):
        super().__init__()
        self.func = func
        self.folder_path = folder_path
        self.args = args
        self.kwargs = kwargs

    def run(self):
        def callback(msg):
            # Accept '10%', 10, '10.0', etc. Emit integer percent when possible.
            pct = None
            try:
                if isinstance(msg, str):
                    s = msg.strip()
                    if s.endswith("%"):
                        s = s[:-1].strip()
                    pct = int(float(s))
                elif isinstance(msg, (int, float)):
                    pct = int(msg)
            except Exception:
                pct = None

            if pct is not None:
                # Clamp
                if pct < 0:
                    pct = 0
                if pct > 100:
                    pct = 100
                self.progress_update.emit(int(pct))

        try:
            # Call the function with the progress callback
            result = self.func(self.folder_path, progress_callback=callback, *self.args, **self.kwargs)
            if result is None:
                result = []
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class WizardMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Temperature & Spectrum Data Analyzer")

        # Set a larger size for the window
        self.resize(700, 400)  # Width, Height in pixels
        
        # Center the window on the screen
        screen_geometry = QGuiApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # self.showMaximized()

        # Raw lists (list-of-dicts) returned by your processing functions
        self.spectrum_list = None
        self.temperature_list_raw = None
        self.temperature_list_filtered = None
        self.combined_list = None

        self.spectrum_thread = None
        self.temperature_thread = None

        # Stacked widget for steps
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # build pages
        self._build_spectrum_page()
        self._build_temperature_page()
        self._build_combined_page()

        # Navigation buttons area (bottom-right)
        # We'll show simple Next/Previous inside each page's layout to keep it full-screen

    # ----------------------- Spectrum Page -----------------------
    def _build_spectrum_page(self):
        page = QWidget()
        v = QVBoxLayout(page)

        title = QLabel("<h2>Step 1 of 3 — Spectrum</h2>")
        v.addWidget(title)

        # Folder selector + process
        h = QHBoxLayout()
        self.spec_folder_label = QLineEdit()
        self.spec_folder_label.setReadOnly(True)
        self.spec_browse_btn = QPushButton("Browse Spectrum Folder")
        self.spec_browse_btn.clicked.connect(self._browse_spectrum)
        self.spec_process_btn = QPushButton("Process Spectrum Data")
        self.spec_process_btn.clicked.connect(self._process_spectrum)
        h.addWidget(self.spec_folder_label)
        h.addWidget(self.spec_browse_btn)
        h.addWidget(self.spec_process_btn)
        v.addLayout(h)

        # Progress bar (percentage visible)
        self.spec_progress = QProgressBar()
        self.spec_progress.setFormat("%p%")
        self.spec_progress.setTextVisible(True)
        v.addWidget(self.spec_progress)

        # General data label (hidden initially, styled like progress bar)
        self.spec_general_label = QLabel("")
        self.spec_general_label.setVisible(False)
        self.spec_general_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spec_general_label.setStyleSheet("""
            QLabel {
                background-color: palette(highlight);
                color: palette(highlighted-text);
                padding: 3px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        v.addWidget(self.spec_general_label)


        # Table (File, Datetime, Abs Range)
        self.spec_table = QTableWidget()
        self.spec_table.setColumnCount(3)
        self.spec_table.setHorizontalHeaderLabels(["File", "Datetime", "Abs Min–Max"])
        v.addWidget(self.spec_table)
        
        # Auto-size the table columns
        self.spec_table.resizeColumnsToContents()

        # Set the horizontal header to stretch
        self.spec_table.horizontalHeader().setStretchLastSection(True)

        # Plot controls (match your original settings)
        control_row = QHBoxLayout()
        self.spec_start = QLineEdit("1")
        self.spec_start.setValidator(QIntValidator())
        self.spec_end = QLineEdit("-1")
        self.spec_end.setValidator(QIntValidator())
        self.spec_norm = QCheckBox("Normalize")
        self.spec_interp = QCheckBox("Interpolate")

        control_row.addWidget(QLabel("Start:"))
        control_row.addWidget(self.spec_start)
        control_row.addWidget(QLabel("End:"))
        control_row.addWidget(self.spec_end)
        control_row.addWidget(self.spec_norm)
        control_row.addWidget(self.spec_interp)

        btn_plot_spec = QPushButton("Plot Spectrum")
        btn_plot_spec.clicked.connect(self._plot_spectrum)
        control_row.addWidget(btn_plot_spec)

        v.addLayout(control_row)

        # Navigation
        nav = QHBoxLayout()
        nav.addStretch()
        self.btn_next_1 = QPushButton("Next →")
        self.btn_next_1.setEnabled(False)  # enabled only after spectrum processed
        self.btn_next_1.clicked.connect(lambda: self._goto_step(1))
        nav.addWidget(self.btn_next_1)
        v.addLayout(nav)

        self.stack.addWidget(page)

    def _browse_spectrum(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Spectrum Data Folder")
        if folder:
            self.spec_folder_label.setText(folder)

    def _process_spectrum(self):
        folder = self.spec_folder_label.text()
        if not folder:
            QMessageBox.warning(self, "No folder", "Please select a spectrum folder first.")
            return
        
        # Clear previous data
        self.spec_table.setRowCount(0)
        self.btn_next_1.setEnabled(False)
        self.spec_general_label.setVisible(False)
        self.spec_progress.setVisible(True)
        self.spec_progress.setValue(0)
        
        self.spec_process_btn.setEnabled(False)

        # start thread
        self.spectrum_thread = ProcessingThread(process_spectrum_data, folder)
        self.spectrum_thread.progress_update.connect(self.spec_progress.setValue)
        self.spectrum_thread.result_ready.connect(self._on_spectrum_done)
        self.spectrum_thread.error.connect(self._on_error)
        self.spectrum_thread.start()

    def _on_spectrum_done(self, result_list):
        self.spec_process_btn.setEnabled(True)
        self.spectrum_list = sorted(result_list or [], key=lambda e: e.get("timestamp", 0))
        if not self.spectrum_list:
            QMessageBox.warning(self, "Spectrum", "No spectrum data processed.")
            self.btn_next_1.setEnabled(False)
            return

        row_count = len(self.spectrum_list)
        self.spec_table.setRowCount(row_count)

        abs_min_vals = []
        abs_max_vals = []
        wn_min_vals = []
        wn_max_vals = []
        timestamps = []

        for i, entry in enumerate(self.spectrum_list):
            # File name
            file_name = os.path.basename(entry.get("file_path", ""))

            # Timestamp / datetime
            dt = entry.get("datetime") or entry.get("timestamp") or ""
            if isinstance(entry.get("timestamp"), (int, float)):
                timestamps.append(entry["timestamp"])

            # Absorbance
            arr = np.asarray(entry.get("absorbance", []))
            if arr.size:
                a_min = arr.min()
                a_max = arr.max()
                abs_min_vals.append(a_min)
                abs_max_vals.append(a_max)
                abs_text = f"{a_min:.4f}–{a_max:.4f}"
            else:
                abs_text = ""

            # Wavenumbers
            wn_array = np.asarray(entry.get("wavenumbers", []))
            if wn_array.size:
                wn_min_vals.append(wn_array.min())
                wn_max_vals.append(wn_array.max())

            # Populate table row
            self.spec_table.setItem(i, 0, QTableWidgetItem(file_name))
            self.spec_table.setItem(i, 1, QTableWidgetItem(str(dt)))
            self.spec_table.setItem(i, 2, QTableWidgetItem(abs_text))

        self.spec_table.resizeColumnsToContents()
        self.btn_next_1.setEnabled(True)

        # Unlock temperature processing if folder chosen
        if getattr(self, 'temp_folder_text', None):
            try:
                self.temp_process_btn.setEnabled(True)
            except Exception:
                pass

        # Summary label
        if abs_min_vals and abs_max_vals and wn_min_vals and wn_max_vals and timestamps:
            spec_min = min(abs_min_vals)
            spec_max = max(abs_max_vals)
            wn_min = min(wn_min_vals)
            wn_max = max(wn_max_vals)

            general_data = (
                f"Number of Files: {row_count} | "
                f"Abs Range: {spec_min:.4f}–{spec_max:.4f} | "
                f"Wavenumber Range: {wn_min:.2f}–{wn_max:.2f} cm⁻¹"
            )

            self.spec_progress.setVisible(False)
            self.spec_general_label.setText(general_data)
            self.spec_general_label.setVisible(True)

    # ----------------------- Temperature Page -----------------------
    def _build_temperature_page(self):
        page = QWidget()
        v = QVBoxLayout(page)

        title = QLabel("<h2>Step 2 of 3 — Temperature</h2>")
        v.addWidget(title)

        # Folder + process
        h = QHBoxLayout()
        self.temp_folder_label = QLineEdit()
        self.temp_folder_label.setReadOnly(True)
        btn_browse_temp = QPushButton("Browse Temperature Folder")
        btn_browse_temp.clicked.connect(self._browse_temperature)
        self.temp_process_btn = QPushButton("Process Temperature Data")
        self.temp_process_btn.setEnabled(False)  # locked until spectrum processed
        self.temp_process_btn.clicked.connect(self._process_temperature)
        h.addWidget(self.temp_folder_label)
        h.addWidget(btn_browse_temp)
        h.addWidget(self.temp_process_btn)
        v.addLayout(h)

        # Progress bar
        self.temp_progress = QProgressBar()
        self.temp_progress.setFormat("%p%")
        self.temp_progress.setTextVisible(True)
        v.addWidget(self.temp_progress)
        
        # General data label (hidden initially, styled like progress bar)
        self.temp_general_label = QLabel("")
        self.temp_general_label.setVisible(False)
        self.temp_general_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.temp_general_label.setStyleSheet("""
            QLabel {
                background-color: palette(highlight);
                color: palette(highlighted-text);
                padding: 3px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        v.addWidget(self.temp_general_label)

        # Table
        self.temp_table = QTableWidget()
        self.temp_table.setColumnCount(2)
        self.temp_table.setHorizontalHeaderLabels(["Datetime", "Temperature (K)"])
        v.addWidget(self.temp_table)
        
        # Auto-size the table columns
        self.temp_table.resizeColumnsToContents()

        # Set the horizontal header to stretch
        self.temp_table.horizontalHeader().setStretchLastSection(True)

        # Plot controls (Start, End, Smooth)
        ctrl = QHBoxLayout()
        self.temp_start = QLineEdit("1")
        self.temp_start.setValidator(QIntValidator())
        self.temp_end = QLineEdit("-1")
        self.temp_end.setValidator(QIntValidator())
        self.temp_smooth = QLineEdit("0")
        self.temp_smooth.setValidator(QIntValidator())
        self.temp_interp = QCheckBox("Interpolate")

        ctrl.addWidget(QLabel("Start:"))
        ctrl.addWidget(self.temp_start)
        ctrl.addWidget(QLabel("End:"))
        ctrl.addWidget(self.temp_end)
        ctrl.addWidget(QLabel("Smooth:"))
        ctrl.addWidget(self.temp_smooth)
        ctrl.addWidget(self.temp_interp)

        btn_plot_temp = QPushButton("Plot Temperature")
        btn_plot_temp.clicked.connect(self._plot_temperature)
        ctrl.addWidget(btn_plot_temp)
        v.addLayout(ctrl)

        # Navigation
        nav = QHBoxLayout()
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(lambda: self._goto_step(0))
        nav.addWidget(prev_btn)
        nav.addStretch()
        self.btn_next_2 = QPushButton("Next →")
        self.btn_next_2.setEnabled(False)
        self.btn_next_2.clicked.connect(lambda: self._goto_step(2))
        nav.addWidget(self.btn_next_2)
        v.addLayout(nav)

        self.stack.addWidget(page)

    def _browse_temperature(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Temperature Data Folder")
        if folder:
            self.temp_folder_label.setText(folder)
            self.temp_folder_text = folder
            # enable processing only if spectrum already processed
            if self.spectrum_list:
                self.temp_process_btn.setEnabled(True)

    def _process_temperature(self):
        folder = self.temp_folder_label.text()
        if not folder:
            QMessageBox.warning(self, "No folder", "Select a temperature folder first.")
            return
        if not self.spectrum_list:
            QMessageBox.warning(self, "Locked", "Temperature processing is locked until spectrum is processed.")
            return

        # Clear previous data
        self.temp_table.setRowCount(0)
        self.btn_next_2.setEnabled(False)
        self.temp_progress.setValue(0)
        self.temp_progress.setVisible(True)
        self.temp_general_label.setVisible(False)
        self.temp_progress.setVisible(True)
        self.temp_progress.setValue(0)
        
        self.temp_process_btn.setEnabled(False)

        # Start temperature processing thread
        self.temperature_thread = ProcessingThread(process_temperature_data, folder)
        self.temperature_thread.progress_update.connect(self.temp_progress.setValue)
        self.temperature_thread.result_ready.connect(self._on_temperature_done)
        self.temperature_thread.error.connect(self._on_error)
        self.temperature_thread.start()

    def _on_temperature_done(self, result_list):
        self.temp_process_btn.setEnabled(True)
        self.temperature_list_raw = result_list or []
        if not self.temperature_list_raw:
            QMessageBox.warning(self, "Temperature", "No temperature data processed.")
            self.btn_next_2.setEnabled(False)
            return
        # Filter to spectrum timeframe (single pass)
        if self.spectrum_list:
            first_spec_ts = self.spectrum_list[0]["timestamp"]
            last_spec_ts = self.spectrum_list[-1]["timestamp"]

            idx_before = idx_after = None
            for i, entry in enumerate(self.temperature_list_raw):
                ts = entry["timestamp"]
                if ts <= first_spec_ts:
                    idx_before = i
                if idx_after is None and ts >= last_spec_ts:
                    idx_after = i
                    break  # stop early once last is found

            if idx_before is not None and idx_after is not None and idx_after >= idx_before:
                self.temperature_list_filtered = (
                    [self.temperature_list_raw[idx_before]]
                    + self.temperature_list_raw[idx_before + 1:idx_after]
                    + ([self.temperature_list_raw[idx_after]] if idx_after != idx_before else [])
                )
            else:
                self.temperature_list_filtered = self.temperature_list_raw
        else:
            self.temperature_list_filtered = self.temperature_list_raw

        # Populate table + compute min/max in one pass
        row_count = len(self.temperature_list_filtered)
        self.temp_table.setRowCount(row_count)

        min_temp = float("inf")
        max_temp = float("-inf")

        for i, entry in enumerate(self.temperature_list_filtered):
            ts = entry.get("timestamp")
            dt_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else "Not Available"
            temp = entry.get("temperature", "")

            if isinstance(temp, (int, float)):
                min_temp = min(min_temp, temp)
                max_temp = max(max_temp, temp)
                temp_str = f"{temp:.4f}"
            else:
                temp_str = str(temp)

            self.temp_table.setItem(i, 0, QTableWidgetItem(dt_str))
            self.temp_table.setItem(i, 1, QTableWidgetItem(temp_str))

        self.temp_table.resizeColumnsToContents()

        # Display general data
        if min_temp != float("inf"):
            general_data = f"Min Temp: {min_temp:.4f} K | Max Temp: {max_temp:.4f} K"
            self.temp_general_label.setText(general_data)
            self.temp_general_label.setVisible(True)

        self.temp_progress.setVisible(False)
        self.btn_next_2.setEnabled(True)

    # ----------------------- Combined Page -----------------------
    def _build_combined_page(self):
        page = QWidget()
        v = QVBoxLayout(page)

        title = QLabel("<h2>Step 3 of 3 — Combined Data</h2>")
        v.addWidget(title)

        # Summary label
        self.combined_summary_label = QLabel("")
        self.combined_summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.combined_summary_label.setStyleSheet("""
            QLabel {
                background-color: palette(highlight);
                color: palette(highlighted-text);
                padding: 3px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        v.addWidget(self.combined_summary_label)

        self.combined_table = QTableWidget()
        self.combined_table.setColumnCount(3)
        self.combined_table.setHorizontalHeaderLabels(["Datetime", "Temperature (K)", "Abs Min–Max"]) 
        v.addWidget(self.combined_table)

        # Auto-size the table columns
        self.combined_table.resizeColumnsToContents()

        # Set the horizontal header to stretch
        self.combined_table.horizontalHeader().setStretchLastSection(True)

        # Create a horizontal layout for the 3D plot controls
        h3d_layout = QHBoxLayout()

        # Smoothing input for 3D plot
        self.smoothing_input = QLineEdit()
        self.smoothing_input.setPlaceholderText("Enter temperature smoothing factor (can improve performance)")
        h3d_layout.addWidget(self.smoothing_input)

        # Colormap dropdown for 3D plot
        self.cmap_dropdown = QComboBox()
        colormaps = ['plasma', 'viridis', 'inferno', 'cividis', 'magma']
        self.cmap_dropdown.addItems(colormaps)
        h3d_layout.addWidget(QLabel("Select Colormap:"))
        h3d_layout.addWidget(self.cmap_dropdown)

        btn_plot_3d = QPushButton("Plot 3D Surface")
        btn_plot_3d.clicked.connect(self._plot_3d)
        h3d_layout.addWidget(btn_plot_3d)

        v.addLayout(h3d_layout)

        # Create a horizontal layout for the absorption plot controls
        habs_layout = QHBoxLayout()

        # Range selection for wavelengths
        self.wavelength_range_input = QLineEdit()
        self.wavelength_range_input.setPlaceholderText("Enter wavelength range (e.g., 1000-2000)")
        habs_layout.addWidget(self.wavelength_range_input)

        btn_plot_absorption = QPushButton("Plot Absorption Over Temperature")
        btn_plot_absorption.clicked.connect(self._plot_absorption)
        habs_layout.addWidget(btn_plot_absorption)

        v.addLayout(habs_layout)

        # Navigation buttons
        nav = QHBoxLayout()
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(lambda: self._goto_step(1))
        nav.addWidget(prev_btn)
        nav.addStretch()
        v.addLayout(nav)

        self.stack.addWidget(page)

    def _goto_step(self, idx: int):
        # Basic navigation; when jumping to step 3 we ensure combine is run
        self.stack.setCurrentIndex(idx)
        if idx == 2:
            # combine automatically
            self._combine()

    def _combine(self):
        if not self.spectrum_list or not self.temperature_list_filtered:
            QMessageBox.warning(self, "Combine", "Need both spectrum and temperature data to combine.")
            return

        try:
            self.combined_list = combine_temperature_and_spectrum_data(
                self.temperature_list_filtered,
                self.spectrum_list,
                time_buffer=10
            ) or []
        except Exception as e:
            QMessageBox.critical(self, "Combine error", str(e))
            self.combined_list = []

        if not self.combined_list:
            QMessageBox.information(self, "Combine", "No combined entries.")
            self.combined_table.setRowCount(0)
            self.combined_summary_label.setText("No combined data available.")
            return

        row_count = len(self.combined_list)
        self.combined_table.setRowCount(row_count)

        # Running min/max trackers
        abs_min = float("inf")
        abs_max = float("-inf")
        temp_min = float("inf")
        temp_max = float("-inf")
        wn_min = float("inf")
        wn_max = float("-inf")
        timestamps = []

        # Single pass: fill table + compute all stats
        for i, c in enumerate(self.combined_list):
            dt = c.get("datetime") or c.get("timestamp") or ""
            temp = c.get("temperature", "")

            # Absorbance
            arr = np.asarray(c.get("absorbance", []))
            if arr.size:
                a_min = arr.min()
                a_max = arr.max()
                abs_min = min(abs_min, a_min)
                abs_max = max(abs_max, a_max)
                abs_text = f"{a_min:.4f}–{a_max:.4f}"
            else:
                abs_text = ""

            # Temperature
            if isinstance(temp, (int, float)):
                temp_min = min(temp_min, temp)
                temp_max = max(temp_max, temp)

            # Wavenumbers
            wn_array = np.asarray(c.get("wavenumbers", []))
            if wn_array.size:
                wn_min = min(wn_min, wn_array.min())
                wn_max = max(wn_max, wn_array.max())

            # Timestamps
            if isinstance(c.get("timestamp"), (int, float)):
                timestamps.append(c["timestamp"])

            # Table row
            self.combined_table.setItem(i, 0, QTableWidgetItem(str(dt)))
            self.combined_table.setItem(
                i, 1, QTableWidgetItem(f"{temp:.4f}" if isinstance(temp, (int, float)) else str(temp))
            )
            self.combined_table.setItem(i, 2, QTableWidgetItem(abs_text))

        self.combined_table.resizeColumnsToContents()
        self.combined_table.horizontalHeader().setStretchLastSection(True)

        # Summary string
        summary_str = (
            f"Abs Range: {abs_min:.4f}–{abs_max:.4f} | "
            f"Temp Range: {temp_min:.4f}–{temp_max:.4f} K | "
            f"Wavenumber Range: {wn_min:.2f}–{wn_max:.2f} cm⁻¹"
        )

        self.combined_summary_label.setText(summary_str)

    # ----------------------- Plot helpers -----------------------
    def _plot_spectrum(self):
        if not self.spectrum_list:
            QMessageBox.warning(self, "Plot", "No spectrum data.")
            return
        try:
            start = int(self.spec_start.text())
        except Exception:
            start = 1
        try:
            end = int(self.spec_end.text())
        except Exception:
            end = -1

        data = interpolate_spectrum_data(self.spectrum_list) if self.spec_interp.isChecked() else self.spectrum_list

        # call your plotting function; pass a no-op progress callback since logs are removed
        plot_spectra(
            data,
            start_index=start,
            end_index=end,
            normalize=self.spec_norm.isChecked(),
            progress_callback=lambda *a, **k: None,
            index_offset=True
        )

    def _plot_temperature(self):
        if not self.temperature_list_filtered:
            QMessageBox.warning(self, "Plot", "No temperature data.")
            return
        try:
            start = int(self.temp_start.text())
        except Exception:
            start = 1
        try:
            end = int(self.temp_end.text())
        except Exception:
            end = -1
        try:
            win = int(self.temp_smooth.text())
        except Exception:
            win = 0
        
        # Check the state of the interpolation checkbox
        inter = self.temp_interp.checkState() == Qt.CheckState.Checked

        # Interpolate temperature data if the checkbox is checked
        data = interpolate_to_round_seconds(self.temperature_list_filtered) if inter else self.temperature_list_filtered

        plot_temperature(
            data,
            start_index=start,
            end_index=end,
            smooth=(win != 0),
            window_size=win,
            progress_callback=lambda *a, **k: None,
            index_offset=True
        )

    def show_3d_surface_window(self, combined_data, cmap='plasma', max_points=2_000_000):
        if not combined_data or not isinstance(combined_data, list) or not isinstance(combined_data[0], dict):
            QMessageBox.warning(self, "3D Plot", "Data format invalid.")
            return None

        # Convert to arrays
        wavenumbers = np.asarray(combined_data[0]["wavenumbers"])
        temperatures = np.asarray([entry["temperature"] for entry in combined_data])
        absorbances = np.asarray([entry["absorbance"] for entry in combined_data])

        # Downsample if too big
        M, N = absorbances.shape
        if M * N > max_points:
            step_m = max(1, int(np.ceil(M / np.sqrt(max_points / N))))
            step_n = max(1, int(np.ceil(N / np.sqrt(max_points / M))))
            temperatures = temperatures[::step_m]
            wavenumbers = wavenumbers[::step_n]
            absorbances = absorbances[::step_m, ::step_n]

        # Plotly figure
        fig = go.Figure(data=[go.Surface(z=absorbances, x=wavenumbers, y=temperatures, colorscale=cmap)])
        fig.update_layout(
            title="3D Absorbance Surface",
            scene=dict(
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Temperature (K)",
                zaxis_title="Absorbance"
            ),
            autosize=True
        )

        # Save HTML to temp file with Plotly inline
        tmp_html = os.path.join(tempfile.gettempdir(), "plotly_surface.html")
        fig.write_html(tmp_html, include_plotlyjs='inline')

        # PyQt window
        win = QMainWindow()
        win.setWindowTitle("3D Absorbance Surface")
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        view = QWebEngineView()
        view.setUrl(QUrl.fromLocalFile(tmp_html))
        
        # Force repaint when finished loading
        def on_load_finished(ok):
            if ok:
                # tiny resize triggers redraw
                view.resize(view.width() + 1, view.height() + 1)
                view.resize(view.width() - 1, view.height() - 1)

        view.loadFinished.connect(on_load_finished)
        layout.addWidget(view)
        win.setCentralWidget(central_widget)
        win.resize(1200, 800)
        win.show()
        return win

    def _plot_3d(self):
        if not self.combined_list:
            QMessageBox.warning(self, "Plot 3D", "No combined data to plot.")
            return
        
        data_to_plot = self.combined_list
        
        if self.smoothing_input.text() and int(self.smoothing_input.text()) != 0:
            try:
                temp_window = float(self.smoothing_input.text()) / 10
                data_to_plot = smooth_combined_by_temperature(data_to_plot, temp_window=temp_window)
            except ValueError:
                QMessageBox.warning(self, "Smoothing", "Invalid smoothing value.")
                return

        cmap = self.cmap_dropdown.currentText()

        self.surface_window = self.show_3d_surface_window(data_to_plot, cmap)

    def _plot_absorption(self):
        if not self.combined_list:
            QMessageBox.warning(self, "Plot Absorption", "No combined data to plot.")
            return

        wavelength_range = self.wavelength_range_input.text()
        if not wavelength_range:
            QMessageBox.warning(self, "Plot Absorption", "Please enter a wavelength range.")
            return

        try:
            start_wavelength, end_wavelength = map(float, wavelength_range.split('-'))
        except ValueError:
            QMessageBox.warning(self, "Plot Absorption", "Invalid wavelength range format.")
            return

        plot_absorption_vs_temperature(
            self.combined_list,
            start_wavelength,
            end_wavelength,
            smooth=True,
            window_size=float(self.smoothing_input.text()) if self.smoothing_input.text() else 5.0
        )

    # ----------------------- Error handler -----------------------
    def _on_error(self, msg):
        QMessageBox.critical(self, "Processing Error", msg)


# ---------------------- Launch ----------------------

def launch_app():
    app = QApplication(sys.argv)
    win = WizardMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    launch_app()
