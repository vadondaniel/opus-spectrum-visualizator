# OPUS Visualizator

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/github/license/vadondaniel/opus-spectrum-visualizator)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![Last Commit](https://img.shields.io/github/last-commit/vadondaniel/opus-spectrum-visualizator)

A tool for visualizing and analyzing OPUS spectral data with temperature correlation and peak analysis.

---

## 📦 Installation

You can run the project either from source or build it into an executable. You'll need [Python](https://www.python.org/downloads/windows).

### 1. Clone or Download the Repository

You can either clone the repository using [Git](https://git-scm.com/downloads):

```bash
git clone https://github.com/vadondaniel/opus-spectrum-visualizator.git
cd opus-spectrum-visualizator
```

Or, if you prefer not to use Git, download the repository as a ZIP file from the GitHub page, then extract it to a folder of your choice.

### 2. Run the Application

You can launch the application directly using Python:

```bash
python main.pyw
```

> 💡 The `main.pyw` file automatically checks for required Python packages and installs any missing dependencies on first run.

### 3. Notes

* Running `main.pyw` directly works on Windows with a double-click as well, because `.pyw` files start in GUI mode without opening a console window.
* If you want to run from a terminal, you can still use `python main.pyw` for debugging or seeing logs.

---

## ⚙️ Building an Executable

To create a standalone `.exe` with **PyInstaller**:

```bash
pyinstaller --onefile main.spec
```

It will be at `dist/main.exe`

## ⚙️ Building an Installer

Using [InnoSetup](https://jrsoftware.org/isdl.php)

```bash
pyinstaller main.spec
```

Then open installer.iss with Inno Setup, and Press Ctrl+F9 or find the compile button.

It will be at `Output/OpusSpectrumVisualizatorInstaller.exe`

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

### Spectral Data Visualization

  <img src="images/spectra_plot.png" alt="3D Surface Plot" width="600">  

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
