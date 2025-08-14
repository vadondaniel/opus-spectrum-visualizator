# OPUS Visualizator

## to start

in CMD

> ..\venv\Scripts\activate

> pip install -r requirements.txt

> python main.py

## to install exe

> pyinstaller --onefile --windowed --additional-hooks-dir=hooks main.py

---

## ✅ **1. Choose the Right Framework**

Since you want:

* Folder selection
* File parsing and processing
* 3D visualization
* Local desktop-style UI

### 🔹 Best Fit: **PyQt5 / PyQt6** or **PySide6**

* Pros:

  * Native-looking, responsive desktop UI
  * File/folder dialogs
  * Embeds 3D plots (via `matplotlib` or `pyqtgraph`)
  * Highly customizable

### 🔹 Simpler Alternative: **Tkinter**

* Easier to learn, built into Python, but limited GUI flexibility and 3D plotting

### 🔹 Web-based UI: **Dash or Flask**

* Dash is great for data science dashboards with built-in Plotly 3D
* But folder selection & file I/O is clunky via browser

> **Recommendation**: Go with **PyQt5 or PySide6** for a full-featured, desktop-native UI. You can add Dash or Flask later for web-based access if needed.

---

## ✅ **2. Project Structure Suggestion**

Here’s a modular layout:

```html
my_temp_spectrum_app/
│
├── main.py                  # Launches the GUI
├── ui/                      # GUI code (Qt/Tkinter/Dash)
│   └── main_window.py
│
├── data_processing/
│   ├── __init__.py
│   ├── temperature.py       # Functions for temperature file processing
│   └── spectrum.py          # Functions for spectrum file processing
│
├── utils/
│   ├── file_io.py           # Folder/file reading helpers
│   └── plot_3d.py           # 3D plotting (e.g., with matplotlib/plotly)
│
├── resources/               # Icons, test data, etc.
└── README.md
```

---

## ✅ **3. Libraries to Use**

* **PyQt6** – UI
* **matplotlib** or **plotly** – for 3D plots
* **pandas / numpy** – for data parsing and correlation
* **os / pathlib** – for file handling

---

## ✅ **4. Starting Steps**

1. **Set up project structure**
2. **Create dummy temp/spectrum data** to test
3. **Build GUI:**

   * Folder picker
   * File list preview
   * Process button
   * 3D plot widget
4. **Implement data loaders** in `data_processing/`
5. **Build correlation logic** (e.g., correlation matrix)
6. **Plot correlation in 3D**
