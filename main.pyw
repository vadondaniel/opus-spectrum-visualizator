import sys
import subprocess
import importlib

# mapping: import_name -> pip_name
required_packages = {
    "PyQt6": "PyQt6",
    "PyQt6.QtWebEngineWidgets": "PyQt6-WebEngine",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "numpy": "numpy",
    "pandas": "pandas",
    "spectrochempy": "spectrochempy",
    "scipy": "scipy",
    "PyInstaller": "pyinstaller",
}

for import_name, pip_name in required_packages.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"{pip_name} not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

# Now safe to import the app
from ui.main_window import launch_app

if __name__ == "__main__":
    launch_app()
