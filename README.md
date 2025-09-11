# OPUS Visualizator

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-CeCILL--B-green)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![Last Commit](https://img.shields.io/github/last-commit/vadondaniel/opus-spectrum-visualizator)

A tool for visualizing and analyzing OPUS spectral data with temperature correlation and peak analysis.

---

## 📦 Installation

You can run the project either from source or build it into an executable.

### 1. Clone the repository

```bash
git clone https://github.com/vadondaniel/opus-spectrum-visualizator.git
cd opus-spectrum-visualizator
```

### 2. Install, and optionally create and activate a virtual environment

#### In **CMD** (venv optional)

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

#### In **PowerShell** (venv optional)

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Activate the virtual environment and launch:

```bash
.\venv\Scripts\activate
python main.py
```

---

## ⚙️ Building an Executable

To create a standalone `.exe` with **PyInstaller**:

```bash
pyinstaller --onefile --windowed --additional-hooks-dir=hooks main.py
```

---

## 📚 Usage Guide

1. Select the folder containing OPUS files.
2. Select the `.txt` file containing temperature data.
3. Start the processing.
4. For 3D visualization: set desired options, then click **Plot 3D**.
5. For peak analysis: adjust parameters, then either

   - click **Export as CSV** to save results, or
   - click **Peak Analysis** to visualize directly.

---

## 🖼 Screenshots

### Main Interface

  <img src="images/main_ui.png" alt="Main UI" width="600">

### 3D Visualization

  <img src="images/surface_plot.png" alt="3D Surface Plot" width="600">  
  <img src="images/scatter_plot.png" alt="3D Scatter Plot" width="600">

### Peak Analysis

  <img src="images/peak_analysis.png" alt="Peak Analysis" width="600">

---

## 🛠 Libraries Used

- **PyQt6** – graphical interface
- **matplotlib** – 2D & 3D plotting
- **pandas & numpy** – data parsing and correlation
- [**SpectroChemPy**](https://github.com/spectrochempy/spectrochempy) – OPUS file processing

---

## 📄 License & Attribution

This project uses `read_opus.py` by **LCS – Laboratoire Catalyse et Spectrochimie, Caen, France**, licensed under **CeCILL-B**.

- You may use, modify, and distribute this software.
- Attribution must be retained.
- Provided "as-is" without warranty.

Original source: [SpectroChemPy GitHub](https://github.com/spectrochempy/spectrochempy/)
