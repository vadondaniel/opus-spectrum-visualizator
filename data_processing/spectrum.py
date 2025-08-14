import math
import os
from datetime import datetime
from pathlib import Path
import spectrochempy as scp
import numpy as np
import re

from concurrent.futures import ThreadPoolExecutor, as_completed


def natural_sort_key(s):
    """
    Key function for natural sorting.
    Splits text into digit and non-digit chunks for proper numeric order.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def process_spectrum_data(folder_path, progress_callback=None, max_workers=4):
    folder_path = Path(folder_path)
    input_files = sorted(folder_path.rglob("*.*"), key=natural_sort_key)
    total_files = len(input_files)

    if not input_files:
        msg = "No OPUS files found."
        if progress_callback: progress_callback(msg)
        else: print(msg)
        return []

    result_data = []
    completed = 0

    def read_file(file_path):
        nonlocal completed
        try:
            timestamp_unix = os.path.getmtime(file_path)
            timestamp_dt = datetime.fromtimestamp(timestamp_unix)
            X = scp.read_opus(file_path)
            wavenumbers = X.x.to('1/cm').data
            absorbance = X.data[0]
            if wavenumbers is None or absorbance is None:
                return None
            return {
                "file_path": str(file_path),
                "timestamp": timestamp_unix,
                "datetime": timestamp_dt,
                "wavenumbers": wavenumbers,
                "absorbance": absorbance,
            }
        finally:
            completed += 1
            if progress_callback:
                # Smooth progress: interpolate between 0–100
                percent = completed / total_files * 100
                progress_callback(f"{percent:.1f}%")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(read_file, f) for f in input_files]
        for future in as_completed(futures):
            data = future.result()
            if data:
                result_data.append(data)

    # Ensure 100% at the end
    if progress_callback:
        progress_callback("100%")

    return result_data


def interpolate_spectrum_data(spectrum_data, progress_callback=None):
    if not spectrum_data:
        msg = "No spectrum data to interpolate."
        if progress_callback: progress_callback(msg)
        else: print(msg)
        return []

    # Convert all wavenumbers and absorbances to 2D NumPy arrays
    wn_arrays = [np.array(entry["wavenumbers"])[::-1] for entry in spectrum_data]
    ab_arrays = [np.array(entry["absorbance"])[::-1] for entry in spectrum_data]

    # Determine common integer wavenumber range
    min_wn = max(w.min() for w in wn_arrays)
    max_wn = min(w.max() for w in wn_arrays)
    common_wavenumbers = np.arange(int(max_wn), int(min_wn) - 1, -1)

    msg = f"Interpolating to common wavenumber range: {common_wavenumbers[0]} to {common_wavenumbers[-1]} ({len(common_wavenumbers)} points)"
    if progress_callback: progress_callback(msg)
    else: print(msg)

    # Vectorized interpolation
    interpolated_absorbances = np.array([
        np.interp(common_wavenumbers, wn, ab) for wn, ab in zip(wn_arrays, ab_arrays)
    ])

    # Build the interpolated data list
    interpolated_data = []
    for i, entry in enumerate(spectrum_data):
        new_entry = entry.copy()
        new_entry["wavenumbers"] = common_wavenumbers
        new_entry["absorbance"] = interpolated_absorbances[i]
        interpolated_data.append(new_entry)
        if progress_callback:
            progress_callback(f"{(i+1)/len(spectrum_data)*100:.1f}%")

    if progress_callback:
        progress_callback("100% - Interpolation complete.")
    else:
        print("Interpolation complete.")

    return interpolated_data