#!/usr/bin/env python3
import sys
import os
import tempfile
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QLineEdit, QDialog,
    QMessageBox, QComboBox, QGroupBox, QTextBrowser, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go

from data_processing.spectrum import process_spectrum_data
from data_processing.temperature import process_temperature_data

from data_processing.combined_data import (
    combine_temperature_and_spectrum_data, smooth_combined_by_temperature
)
from utils.plot import plot_absorption_vs_temperature

# === Worker Thread Wrapper ===
class ProcessingThread(QThread):
    progress_update = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        def callback(msg):
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
                pct = max(0, min(100, pct))
                self.progress_update.emit(pct)

        try:
            result = self.func(*self.args, progress_callback=callback, **self.kwargs)
            if result is None:
                result = []
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class InfoWindow(QDialog):
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(500, 350)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowCloseButtonHint)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # QTextBrowser for rich text display
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(text)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setReadOnly(True)
        self.text_browser.setFrameStyle(QTextBrowser.Shape.NoFrame)  # remove borders
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

class DataProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrum & Temperature Data Processor")
        self.setGeometry(150, 150, 720, 460)

        self.spectrum_list = []
        self.temperature_list_filtered = []
        self.combined_list = []

        self.spectrum_path = None
        self.temp_file = None

        # === Menu Bar ===
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")
        
        # How To Use action
        howto_action = help_menu.addAction("How To Use")
        howto_action.setToolTip("Show instructions on how to use the application")
        howto_action.triggered.connect(self.show_howto)
        
        # Credits action
        credits_action = help_menu.addAction("Credits")
        credits_action.setToolTip("Show credits and acknowledgments")
        credits_action.triggered.connect(self.show_credits)
        
        # === Central Widget ===
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # === File Selection Section ===
        file_group = QGroupBox("Data Selection")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(10)
        file_layout.setContentsMargins(12, 8, 12, 8)

        # Spectrum selection row
        spec_layout = QHBoxLayout()
        spec_layout.setSpacing(10)
        self.spectrum_label = QLabel("No folder selected")
        self.spectrum_label.setStyleSheet("color: gray;")
        self.spectrum_btn = QPushButton("Browse")
        self.spectrum_btn.setFixedWidth(100)
        self.spectrum_btn.setToolTip("Select the folder containing spectrum files")
        self.spectrum_btn.clicked.connect(self.select_spectrum_folder)
        spec_layout.addWidget(QLabel("Spectrum:"))
        spec_layout.addWidget(self.spectrum_label)
        spec_layout.addStretch()
        spec_layout.addWidget(self.spectrum_btn)

        # Temperature selection row
        temp_layout = QHBoxLayout()
        temp_layout.setSpacing(10)
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

        # Combine rows
        file_layout.addLayout(spec_layout)
        file_layout.addLayout(temp_layout)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # === Processing Section ===
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout()
        process_layout.setSpacing(12)
        process_layout.setContentsMargins(12, 8, 12, 8)

        # Start Processing button
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setFixedWidth(140)
        self.process_btn.setToolTip("Start processing spectrum and temperature data")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        process_layout.addWidget(self.process_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Helper function to create a progress row
        def create_progress_row(label_text):
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(100)
            progress = QProgressBar()
            progress.setFormat("%p%")
            row.addWidget(label)
            row.addWidget(progress)
            return row, progress

        # Spectrum progress
        spec_row, self.spec_progress = create_progress_row("Spectrum:")
        process_layout.addLayout(spec_row)

        # Temperature progress
        temp_row, self.temp_progress = create_progress_row("Temperature:")
        process_layout.addLayout(temp_row)

        # Combine progress
        comb_row, self.combine_progress = create_progress_row("Combine:")
        process_layout.addLayout(comb_row)

        process_group.setLayout(process_layout)
        main_layout.addWidget(process_group)

        # === 3D Plotting ===
        vis_group = QGroupBox("3D Plotting")
        vis_layout = QHBoxLayout()
        vis_layout.setSpacing(10)
        vis_layout.setContentsMargins(10, 6, 10, 6)

        # Plot type
        self.plot_type_dropdown = QComboBox()
        self.plot_type_dropdown.addItems(["Surface", "Scatter"])
        self.plot_type_dropdown.setFixedWidth(80)
        self.plot_type_dropdown.setToolTip("Select the 3D plot type")

        # Colormap
        self.cmap_dropdown = QComboBox()
        self.cmap_dropdown.addItems(["plasma", "viridis", "inferno", "cividis", "magma"])
        self.cmap_dropdown.setFixedWidth(80)
        self.cmap_dropdown.setToolTip("Select the color map for the plot")

        # Smoothing
        self.smoothing_input = QLineEdit()
        self.smoothing_input.setPlaceholderText("Smoothing factor")
        self.smoothing_input.setFixedWidth(110)
        self.smoothing_input.setToolTip("Enter smoothing factor (optional)")

        # Plot button
        self.plot_btn = QPushButton("Plot 3D")
        self.plot_btn.setFixedWidth(90)
        self.plot_btn.clicked.connect(self._plot_3d)
        self.plot_btn.setEnabled(False)

        # Add widgets with labels
        vis_layout.addWidget(QLabel("Type:"))
        vis_layout.addWidget(self.plot_type_dropdown)
        vis_layout.addSpacing(10)
        vis_layout.addWidget(QLabel("Colormap:"))
        vis_layout.addWidget(self.cmap_dropdown)
        vis_layout.addSpacing(10)
        vis_layout.addWidget(QLabel("Smoothing:"))
        vis_layout.addWidget(self.smoothing_input)
        vis_layout.addStretch()
        vis_layout.addWidget(self.plot_btn)

        vis_group.setLayout(vis_layout)
        main_layout.addWidget(vis_group)

        # === Peak Analysis ===
        range_group = QGroupBox("Peak Analysis")
        range_layout = QHBoxLayout()
        range_layout.setSpacing(10)
        range_layout.setContentsMargins(10, 6, 10, 6)
        
        # Wavelength range input
        self.wavelength_range_input = QLineEdit()
        self.wavelength_range_input.setPlaceholderText("950-1050, 2450-2550, 3050-3150")
        self.wavelength_range_input.setToolTip("Enter range(s separated by commas). Example: 950-1050, 2450-2550")
        
        # Display Type
        self.display_type_dropdown = QComboBox()
        self.display_type_dropdown.addItems(["both", "raw", "smoothed"])
        self.display_type_dropdown.setFixedWidth(80)
        self.display_type_dropdown.setToolTip("Select how to plot")
        
        # Smoothing Method
        self.smoothing_method_dropdown = QComboBox()
        self.smoothing_method_dropdown.addItems(["gaussian", "loess", "spline", "boxcar"])
        self.smoothing_method_dropdown.setFixedWidth(80)
        self.smoothing_method_dropdown.setToolTip("Select the method for smoothing")
        
        # Smoothing parameter widgets
        self.smoothing_param_spin = QDoubleSpinBox()
        self.smoothing_param_spin.setDecimals(3)
        self.smoothing_param_spin.setRange(0.001, 1e6)
        self.smoothing_param_spin.setSingleStep(0.1)
        self.smoothing_param_spin.setValue(5.0)  # default
        self.smoothing_param_spin.setFixedWidth(100)
        
        self.smoothing_param_label = QLabel("Param:")
        
        # Peak analysis button
        self.range_btn = QPushButton("Peak Analysis")
        self.range_btn.setFixedWidth(110)
        self.range_btn.clicked.connect(self._plot_absorption)
        self.range_btn.setEnabled(False)
        
        # Add widgets
        range_layout.addWidget(QLabel("Range(s):"))
        range_layout.addWidget(self.wavelength_range_input)
        range_layout.addWidget(QLabel("Display:"))
        range_layout.addWidget(self.display_type_dropdown)
        range_layout.addWidget(QLabel("Method:"))
        range_layout.addWidget(self.smoothing_method_dropdown)
        range_layout.addWidget(self.smoothing_param_label)
        range_layout.addWidget(self.smoothing_param_spin)
        range_layout.addWidget(self.range_btn)
        
        range_group.setLayout(range_layout)
        main_layout.addWidget(range_group)
        
        self.smoothing_method_dropdown.currentTextChanged.connect(self._update_smoothing_param_ui)
        self._update_smoothing_param_ui(self.smoothing_method_dropdown.currentText())
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def show_info(self, title, text):
        self.info_win = InfoWindow(title, text)
        self.info_win.show()
        
    # === Menu action methods ===
    def show_howto(self):
        self.show_info(
            "How To Use",
            """
            <h2>How To Use</h2>
            <ol>
                <li>Select a <b>folder</b> containing <b>OPUS files</b> and a <b>Temperature file</b>.</li>
                <li>Click <b>Start Processing</b> — Spectrum and Temperature will run in parallel, then Combine.</li>
                <li>Use <b>Visualization</b> to plot 3D (Surface or Scatter, choose colormap, optional smoothing).</li>
                <li>Use <b>Range Comparison</b> to enter ranges (e.g. <code>950-1050, 3050-3150</code>) and compare absorption.</li>
            </ol>
            """
        )
    
    def show_credits(self):
        self.show_info(
            "Credits",
            """
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
            """
        )

    # === Slots ===
    def select_spectrum_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Spectrum Folder")
        if folder:
            self.spectrum_path = folder
            self.spectrum_label.setText(f"{folder}")
        self._update_start_enabled()

    def select_temp_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Temperature File", "", "CSV/TXT Files (*.csv *.txt);;All Files (*)")
        if file:
            self.temp_file = file
            self.temp_label.setText(f"{file}")
        self._update_start_enabled()

    def _update_start_enabled(self):
        self.process_btn.setEnabled(bool(self.spectrum_path and self.temp_file))

    def start_processing(self):
        # Reset progress bars
        self.spec_progress.setValue(0)
        self.temp_progress.setValue(0)
        self.combine_progress.setValue(0)

        # Disable buttons until finished
        self.process_btn.setEnabled(False)
        self.plot_btn.setEnabled(False)
        self.range_btn.setEnabled(False)

        # Spectrum
        self.spectrum_thread = ProcessingThread(process_spectrum_data, self.spectrum_path)
        self.spectrum_thread.progress_update.connect(self.spec_progress.setValue)
        self.spectrum_thread.result_ready.connect(self._on_spectrum_done)
        self.spectrum_thread.error.connect(self._on_error)
        self.spectrum_thread.start()

        # Temperature
        self.temperature_thread = ProcessingThread(process_temperature_data, self.temp_file)
        self.temperature_thread.progress_update.connect(self.temp_progress.setValue)
        self.temperature_thread.result_ready.connect(self._on_temperature_done)
        self.temperature_thread.error.connect(self._on_error)
        self.temperature_thread.start()

    def _on_error(self, message: str):
        QMessageBox.critical(self, "Processing error", message)

    def _on_spectrum_done(self, result):
        self.spectrum_list = result
        self.spec_progress.setValue(100)
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
            self.range_btn.setEnabled(True)

        # Re-enable processing start if inputs are still valid
        self._update_start_enabled()

    def show_3d_plot_window(self, combined_data, plot_type="Surface", cmap='plasma', max_points=2_000_000):
        if not combined_data or not isinstance(combined_data, list) or not isinstance(combined_data[0], dict):
            QMessageBox.warning(self, "3D Plot", "Data format invalid.")
            return None

        # Convert to arrays
        wavenumbers = np.asarray(combined_data[0]["wavenumbers"])
        temperatures = np.asarray([entry["temperature"]
                                  for entry in combined_data])
        absorbances = np.asarray([entry["absorbance"]
                                 for entry in combined_data])

        # Downsample if too big
        M, N = absorbances.shape
        if M * N > max_points:
            step_m = max(1, int(np.ceil(M / np.sqrt(max_points / N))))
            step_n = max(1, int(np.ceil(N / np.sqrt(max_points / M))))
            temperatures = temperatures[::step_m]
            wavenumbers = wavenumbers[::step_n]
            absorbances = absorbances[::step_m, ::step_n]

        # Create Plotly figure depending on type
        if plot_type == "Surface":
            fig = go.Figure(data=[
                go.Surface(z=absorbances, x=wavenumbers,
                           y=temperatures, colorscale=cmap)
            ])
        elif plot_type == "Scatter":
            # Flatten arrays for scatter plot
            T, W = np.meshgrid(temperatures, wavenumbers, indexing="ij")
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=W.flatten(),
                    y=T.flatten(),
                    z=absorbances.flatten(),
                    mode="markers",
                    marker=dict(size=2, color=absorbances.flatten(),
                                colorscale=cmap)
                )
            ])

        fig.update_layout(title=f"3D Absorbance {plot_type} Plot")
        fig.update_layout(
            scene=dict(
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Temperature (K)",
                zaxis_title="Absorbance"
            ),
            autosize=True
        )

        # Save HTML to temp file
        tmp_html = os.path.join(tempfile.gettempdir(), "plotly_3d.html")
        fig.write_html(tmp_html, include_plotlyjs='inline')

        # PyQt window
        win = QMainWindow()
        win.setWindowTitle(f"3D Absorbance {plot_type} Plot")
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        view = QWebEngineView()
        view.setUrl(QUrl.fromLocalFile(tmp_html))

        def on_load_finished(ok):
            if ok:
                view.resize(view.width() + 1, view.height() + 1)
                view.resize(view.width() - 1, view.height() - 1)

        view.loadFinished.connect(on_load_finished)
        layout.addWidget(view)
        win.setCentralWidget(central_widget)
        win.resize(1200, 800)
        win.show()
        return win

    def _update_smoothing_param_ui(self, method: str):
        if method == "gaussian":
            self.smoothing_param_label.setText("Bandwidth (K):")
            self.smoothing_param_spin.setRange(0.1, 1e4)
            self.smoothing_param_spin.setValue(5.0)
            self.smoothing_param_spin.setVisible(True)
            self.smoothing_param_label.setVisible(True)
        elif method == "loess":
            self.smoothing_param_label.setText("Frac (0–1):")
            self.smoothing_param_spin.setRange(0.01, 1.0)
            self.smoothing_param_spin.setValue(0.25)
            self.smoothing_param_spin.setVisible(True)
            self.smoothing_param_label.setVisible(True)
        elif method == "spline":
            self.smoothing_param_label.setText("Strength (0–1):")
            self.smoothing_param_spin.setRange(0.0, 1.0)
            self.smoothing_param_spin.setValue(0.2)
            self.smoothing_param_spin.setVisible(True)
            self.smoothing_param_label.setVisible(True)
        elif method == "boxcar":
            self.smoothing_param_label.setText("Window (K):")
            self.smoothing_param_spin.setRange(0.1, 1e4)
            self.smoothing_param_spin.setValue(5.0)
            self.smoothing_param_spin.setVisible(True)
            self.smoothing_param_label.setVisible(True)
        else:
            # "none"
            self.smoothing_param_spin.setVisible(False)
            self.smoothing_param_label.setVisible(False)

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
        plot_type = self.plot_type_dropdown.currentText()

        self.surface_window = self.show_3d_plot_window(data_to_plot, plot_type=plot_type, cmap=cmap)

    def _plot_absorption(self):
        if not self.combined_list:
            QMessageBox.warning(self, "Peak Analysis", "No combined data to plot.")
            return

        wavelength_ranges = self.wavelength_range_input.text()
        if not wavelength_ranges:
            QMessageBox.warning(self, "Peak Analysis", "Please enter one or more wavelength ranges.")
            return

        try:
            ranges = []
            for part in wavelength_ranges.split(","):
                start, end = map(float, part.strip().split('-'))
                ranges.append((start, end))
        except ValueError:
            QMessageBox.warning(self, "Peak Analysis", "Invalid format. Use e.g. 950-1050, 3050-3150")
            return
        
        display_type = self.display_type_dropdown.currentText()
        smoothing_method = self.smoothing_method_dropdown.currentText()
        
        smoothing_param = None
        if smoothing_method != "none":
            smoothing_param = float(self.smoothing_param_spin.value())

        plot_absorption_vs_temperature(
            self.combined_list,
            ranges,
            display_type=display_type,
            smoothing=smoothing_method,
            smoothing_param=smoothing_param,
            eval_on_grid=True
        )

# ---------------------- Launch ----------------------

def launch_app():
    app = QApplication(sys.argv)
    win = DataProcessingApp()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    launch_app()
