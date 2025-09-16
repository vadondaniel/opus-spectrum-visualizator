# OPUS Visualizator

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/github/license/vadondaniel/opus-spectrum-visualizator)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![Last Commit](https://img.shields.io/github/last-commit/vadondaniel/opus-spectrum-visualizator)

A tool for visualizing and analyzing OPUS spectral data with temperature correlation and peak analysis.

---

## 📦 Installation (End-User)

The easiest way to install is by using the prebuilt Windows installer.

### 1. Download the Installer

1. Go to the [Releases page on GitHub](https://github.com/vadondaniel/opus-spectrum-visualizator/releases).
2. Download the file named similar to: `OpusSpectrumVisualizatorInstaller.exe`

### 2. Install

1. Double-click the downloaded `.exe` file.
2. Follow the installation prompts.

### 3. Launch

Start the program from the **Desktop Icon**, **Start Menu** or search for *Opus Spectrum Visualizator*.

---

## ⚙️ Installation (Developer / Source Build)

If you want to run or modify the program from source, follow these steps:

### 1. Requirements

- [Python](https://www.python.org/downloads/windows) 3.10+

### 2. Clone or Download the Repository

Either clone using [Git](https://git-scm.com/downloads):

```bash
git clone https://github.com/vadondaniel/opus-spectrum-visualizator.git
cd opus-spectrum-visualizator
```

Or download the [repository](https://github.com/vadondaniel/opus-spectrum-visualizator/archive/refs/heads/main.zip) or a [release](https://github.com/vadondaniel/opus-spectrum-visualizator/releases) as a ZIP from GitHub and extract it.

### 3. Run the Application

Launch with Python:

```bash
python main.pyw
```

> (or just open the main.pyw file)

> The program will automatically install missing dependencies on first run.

### 4. Build an Executable

(Optional) Create a standalone `.exe` using PyInstaller:

```bash
pyinstaller --onefile main.spec
```

The executable will appear in the `/dist/` folder as `main.exe`

### 5. Build an Installer

(Optional) Using [Inno Setup](https://jrsoftware.org/isdl.php):

1. Run `pyinstaller main.spec`.
2. Open `installer.iss` in Inno Setup.
3. Compile (Ctrl+F9).

The installer will appear in the `/Output/` folder as `OpusSpectrumVisualizatorInstaller.exe`

> Using a version installed with the installer starts considerably faster than a standalone executable file from step 4 above.

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
