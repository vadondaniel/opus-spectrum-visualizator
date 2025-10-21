from PyQt6.QtWidgets import QFileDialog, QWidget
import pandas as pd
import numpy as np


def export_spectra_csv(
    spectra_list,
    parent: QWidget = None,
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
    temperatures = np.asarray([entry["temperature"]
                              for entry in combined_list])
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
        df.index.name = ""

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

    # Helper: trapezoid baseline correction, then width-weighted area
    def _baseline_trapezoid(wn: np.ndarray, ab: np.ndarray) -> np.ndarray:
        wn = np.asarray(wn, dtype=float)
        ab = np.asarray(ab, dtype=float)
        shifted = ab + 1.0
        if len(wn) >= 2 and wn[0] != wn[-1]:
            x0, x1 = wn[0], wn[-1]
            y0, y1 = shifted[0], shifted[-1]
            slope = (y1 - y0) / (x1 - x0)
            baseline = y0 + slope * (wn - x0)
            return shifted - baseline
        return shifted

    def _integrate_peak_area(wn: np.ndarray, ab: np.ndarray) -> float:
        if len(wn) < 2:
            return float(np.sum(ab))
        widths = np.abs(np.diff(wn))
        return float(np.sum(widths * ab[1:]))

    # Extract arrays and temperature ordering
    timestamps = np.array([e.get("timestamp", np.nan) for e in combined_list])
    temperatures = np.array([e["temperature"] for e in combined_list], dtype=float)
    sort_idx = np.argsort(temperatures)
    temperatures_sorted = temperatures[sort_idx]
    timestamps_sorted = timestamps[sort_idx].astype(np.int64)

    data = {
        "Timestamp": timestamps_sorted,
        "Temperature (K)": temperatures_sorted
    }

    for (start_wavelength, end_wavelength) in wavelength_ranges:
        areas = []
        for e in combined_list:
            wn = np.asarray(e["wavenumbers"], dtype=float)
            ab = np.asarray(e["absorbance"], dtype=float)
            mask = (wn >= start_wavelength) & (wn <= end_wavelength)
            wn_slice = wn[mask]
            ab_slice = ab[mask]
            if wn_slice.size == 0:
                areas.append(0.0)
                continue
            ab_corr = _baseline_trapezoid(wn_slice, ab_slice)
            area = _integrate_peak_area(wn_slice, ab_corr)
            areas.append(area)

        data[f"{start_wavelength}-{end_wavelength} cm^-1"] = np.array(areas, dtype=float)[sort_idx]

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    return filename  # return chosen file path
