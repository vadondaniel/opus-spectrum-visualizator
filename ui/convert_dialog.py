from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices

from utils.processing_thread import ProcessingThread
from data_processing.spectrum import convert_opus_to_csv, convert_opus_to_txt


class ConvertDialog(QDialog):
    def __init__(self, input_folder: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Convert OPUS Files")
        self.input_folder = Path(input_folder)
        self.output_folder = Path(input_folder)
        self.thread = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(12, 10, 12, 10)

        # Input folder (read-only display)
        layout.addWidget(QLabel("Input Folder:"))
        self.input_label = QLabel(str(self.input_folder))
        self.input_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.input_label)

        # Output folder selector
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output Folder:"))
        self.output_edit = QLineEdit(str(self.output_folder))
        out_row.addWidget(self.output_edit)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_output_folder)
        out_row.addWidget(self.browse_btn)
        layout.addLayout(out_row)

        # Format selection
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV", "TXT"])
        fmt_row.addWidget(self.format_combo)
        fmt_row.addStretch()
        layout.addLayout(fmt_row)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setFormat("%p%")
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.start_btn = QPushButton("Start Conversion")
        self.start_btn.clicked.connect(self._start_conversion)
        btn_row.addWidget(self.start_btn)
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        btn_row.addWidget(self.open_folder_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(self.output_folder))
        if folder:
            self.output_folder = Path(folder)
            self.output_edit.setText(str(self.output_folder))

    def _start_conversion(self):
        # Resolve output folder from UI
        out_text = self.output_edit.text().strip()
        self.output_folder = Path(out_text) if out_text else self.input_folder
        if not self.output_folder.exists():
            try:
                self.output_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Output Folder",
                                     f"Cannot create output folder:\n{e}")
                return

        self.status_label.setText("")
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.format_combo.setEnabled(False)
        self.output_edit.setEnabled(False)

        fmt = self.format_combo.currentText().lower()
        func = convert_opus_to_csv if fmt == "csv" else convert_opus_to_txt

        self.thread = ProcessingThread(
            func, self.input_folder, self.output_folder)
        self.thread.progress_update.connect(self.progress.setValue)
        self.thread.result_ready.connect(self._on_done)
        self.thread.error.connect(self._on_error)
        self.thread.start()

    def _on_error(self, message: str):
        self.status_label.setText("")
        QMessageBox.critical(self, "Conversion error", message)
        self._reset_controls_after_finish()

    def _on_done(self, results):
        # results is a list of Path or error strings
        total = len(results)
        successes = sum(1 for r in results if isinstance(
            r, (str, Path)) and isinstance(r, Path))
        errors = total - successes
        self.progress.setValue(100)

        if total == 0:
            self.status_label.setText("No files found in the input folder.")
        else:
            self.status_label.setText(
                f"Done. Converted {successes} file(s). {errors} error(s).")

        self.open_folder_btn.setEnabled(True)
        self._reset_controls_after_finish()

    def _reset_controls_after_finish(self):
        self.start_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.format_combo.setEnabled(True)
        self.output_edit.setEnabled(True)

    def _open_output_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_folder)))
