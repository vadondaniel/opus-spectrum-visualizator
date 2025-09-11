from PyQt6.QtWidgets import QFileDialog, QWidget
import pandas as pd
import numpy as np


def export_combined_data_csv(
    combined_list,
    parent: QWidget = None,
    format: str = "long"
):
    """
    Export combined_data (list of dicts with 'temperature', 'wavenumbers', 'absorbance') as CSV.

    Parameters
    ----------
    combined_list : list of dict
        Each dict must contain 'temperature', 'wavenumbers', 'absorbance'.
    parent : QWidget, optional
        Parent widget for QFileDialog.
    format : str, default "long"
        "long"   -> columns: Temperature, Wavenumber, Absorbance
        "matrix" -> rows: Temperature, columns: Wavenumbers, values: Absorbance
    """
    if not combined_list:
        raise ValueError("No combined data to export.")

    if not isinstance(combined_list, list) or not isinstance(combined_list[0], dict):
        raise ValueError("Data format invalid.")

    # File dialog to choose save location
    filename, _ = QFileDialog.getSaveFileName(
        parent,
        "Save Combined Data as CSV",
        "combined_data.csv",
        "CSV Files (*.csv)"
    )

    if not filename:  # user cancelled
        return None

    # Extract arrays
    wavenumbers = np.asarray(combined_list[0]["wavenumbers"])
    temperatures = np.asarray([entry["temperature"] for entry in combined_list])
    absorbances = np.asarray([entry["absorbance"] for entry in combined_list])

    if format == "long":
        # Flatten for CSV
        T, W = np.meshgrid(temperatures, wavenumbers, indexing="ij")
        df = pd.DataFrame({
            "Temperature": T.flatten(),
            "Wavenumber": W.flatten(),
            "Absorbance": absorbances.flatten()
        })

    elif format == "matrix":
        # Rows = temperatures, columns = wavenumbers
        df = pd.DataFrame(absorbances, index=temperatures, columns=wavenumbers)
        df.index.name = "Temperature"

    else:
        raise ValueError("Invalid format. Choose 'long' or 'matrix'.")

    df.to_csv(filename, float_format="%.6f")

    return filename  # return chosen file path


def export_peak_analysis_csv(
    combined_list,
    wavelength_ranges,
    parent: QWidget = None
):
    """
    Export peak analysis to a CSV file chosen via file dialog.

    Parameters:
        combined_list (list of dict): Each dict has "temperature", "wavenumbers", "absorbance".
        wavelength_ranges (list[tuple[float,float]]): [(start_wavenumber, end_wavenumber), ...]
        parent (QWidget): Optional parent widget for the file dialog.
    """
    if not combined_list:
        raise ValueError("No combined data to export.")

    # Ask user for file path
    filename, _ = QFileDialog.getSaveFileName(
        parent,
        "Save Peak Analysis CSV",
        "peak_analysis.csv",
        "CSV Files (*.csv)"
    )

    if not filename:  # user cancelled
        return None

    # Sort by temperature
    temperatures = np.array([e["temperature"] for e in combined_list], dtype=float)
    sort_idx = np.argsort(temperatures)
    temperatures = temperatures[sort_idx]

    data = {"Temperature (K)": temperatures} #np.round(temperatures, 4)

    for (start_wavelength, end_wavelength) in wavelength_ranges:
        summed_absorbance = np.array([
            np.sum(e["absorbance"][(e["wavenumbers"] >= start_wavelength) &
                                   (e["wavenumbers"] <= end_wavelength)])
            for e in combined_list
        ], dtype=float)[sort_idx]

        col_name = f"{start_wavelength}-{end_wavelength} cm^-1"
        data[col_name] = summed_absorbance

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    return filename  # return chosen file path
