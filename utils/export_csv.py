from typing import Optional
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import QFileDialog, QWidget

from utils.peak_analysis import compute_peak_areas


def export_spectra_csv(
    spectra_list,
    parent: Optional[QWidget] = None,
):
    """
    Export spectra_list (list of dicts with 'wavenumbers', 'absorbance') as a simple CSV.

    Output format:
        Wavenumber, Spectrum1, Spectrum2, ...
        4000,       0.123,    0.234,    ...
        3999,       0.124,    0.233,    ...
        ...

    Parameters
    ----------
    spectra_list : list of dict
        Each dict must contain 'wavenumbers' and 'absorbance'.
    parent : QWidget, optional
        Parent widget for QFileDialog.
    """
    if not spectra_list:
        raise ValueError("No spectra to export.")

    # Ask for save location
    filename, _ = QFileDialog.getSaveFileName(
        parent,
        "Save Spectra as CSV",
        "spectra.csv",
        "CSV Files (*.csv)"
    )

    if not filename:  # user cancelled
        return None

    # Assume all spectra have same wavenumbers
    wavenumbers = np.asarray(spectra_list[0]["wavenumbers"])
    absorbances = [np.asarray(entry["absorbance"]) for entry in spectra_list]

    # Build dataframe: col0 = Wavenumber, others = Spectrum1, Spectrum2...
    data = {"Wavenumber": wavenumbers}
    for i, arr in enumerate(absorbances, start=1):
        data[f"Spectrum{i}"] = arr

    df = pd.DataFrame(data)

    df.to_csv(filename, index=False, float_format="%.6f")
    return filename


def export_combined_data_csv(
    combined_list,
    parent: Optional[QWidget] = None,
    format: str = "long",
    x_label: str = "Temperature (K)"
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
    x_label : str, default "Temperature (K)"
        Label for the primary axis column/index.
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
    temperatures = np.asarray([entry["temperature"]
                              for entry in combined_list])
    absorbances = np.asarray([entry["absorbance"] for entry in combined_list])

    if format == "long":
        # Flatten for CSV
        T, W = np.meshgrid(temperatures, wavenumbers, indexing="ij")
        df = pd.DataFrame({
            x_label: T.flatten(),
            "Wavenumber": W.flatten(),
            "Absorbance": absorbances.flatten()
        })

    elif format == "matrix":
        # Rows = temperatures, columns = wavenumbers
        df = pd.DataFrame(absorbances, index=temperatures, columns=wavenumbers)
        df.index.name = x_label

    else:
        raise ValueError("Invalid format. Choose 'long' or 'matrix'.")

    df.to_csv(filename, float_format="%.6f")

    return filename  # return chosen file path


def export_peak_analysis_csv(
    combined_list,
    wavelength_ranges,
    parent: Optional[QWidget] = None,
    baseline_mode: str = "trapezoid",
    x_key: str = "temperature",
    x_label: Optional[str] = None
):
    """
    Export peak analysis to a CSV file chosen via file dialog.

    Parameters:
        combined_list (list of dict): Each dict has "temperature", "wavenumbers", "absorbance".
        wavelength_ranges (list[tuple[float,float]]): [(start_wavenumber, end_wavenumber), ...]
        parent (QWidget): Optional parent widget for the file dialog.
        baseline_mode (str): Baseline mode forwarded to compute_peak_areas so CSV
            matches the plotted data (e.g., "trapezoid" or "none").
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

    # Compute unified peak areas (trapezoid baseline correction + width-weighted integration)
    x_sorted, results = compute_peak_areas(
        combined_list, wavelength_ranges, baseline_mode=baseline_mode, x_key=x_key)
    if x_sorted is None:
        raise ValueError(f"No valid values found for x-axis key: {x_key!r}")

    x_label_final = x_label or ("Temperature (K)" if x_key == "temperature" else x_key)

    # Sort timestamps and temperatures in the same order as x_sorted
    x_values_all = np.array([e.get(x_key, np.nan) for e in combined_list], dtype=float)
    valid_mask = np.isfinite(x_values_all)
    valid_indices = np.where(valid_mask)[0]
    x_sort_local = np.argsort(x_values_all[valid_mask])
    sorted_indices = valid_indices[x_sort_local]

    timestamps = np.array([e.get("timestamp", np.nan) for e in combined_list], dtype=float)
    temperatures = np.array([e.get("temperature", np.nan) for e in combined_list], dtype=float)
    timestamps_sorted = timestamps[sorted_indices]
    if np.all(np.isfinite(timestamps_sorted)):
        timestamps_sorted = np.rint(timestamps_sorted).astype(np.int64)
    temperatures_sorted = temperatures[sorted_indices]

    data = {
        "Timestamp": timestamps_sorted,
        x_label_final: x_sorted
    }
    if x_key != "temperature":
        data["Temperature (K)"] = temperatures_sorted

    for (start_wavelength, end_wavelength) in wavelength_ranges:
        data[f"{start_wavelength}-{end_wavelength} cm^-1"] = results[(start_wavelength, end_wavelength)]

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    return filename  # return chosen file path
