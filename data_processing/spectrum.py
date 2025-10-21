import os
from datetime import datetime
from pathlib import Path
import numpy as np
import re
import csv

from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.read_opus import read_opus


def process_spectrum_data(folder_path, progress_callback=None, max_workers=4):
    """
    Processes spectrum data from OPUS files in a specified folder.

    This function scans a given folder for OPUS files, reads their contents, and extracts relevant
    information such as wavenumbers and absorbance values. The processing is done in parallel using
    a thread pool to improve performance.

    Parameters:
    folder_path (str or Path): The path to the folder containing OPUS files to be processed.
    progress_callback (function, optional): A callback function that takes a string message as an argument.
        This function can be used to report progress during the file processing. If not provided,
        progress messages will be printed to the console.
    max_workers (int, optional): The maximum number of worker threads to use for processing files.
        Default is 4.

    Returns:
    list: A list of dictionaries containing the processed spectrum data from each OPUS file, where each
        dictionary has:
        - "file_path": The path to the processed file.
        - "timestamp": The last modified time of the file (in Unix timestamp format).
        - "datetime": The last modified time of the file (as a datetime object).
        - "wavenumbers": The wavenumber values (in cm^-1).
        - "absorbance": The absorbance values corresponding to the wavenumbers.

    Raises:
    ValueError: If no OPUS files are found in the specified folder, a message will be printed or passed
    to the progress_callback.

    Example:
    >>> folder_path = "/path/to/opus/files"
    >>> spectrum_data = process_spectrum_data(folder_path)
    """
    folder_path = Path(folder_path)
    input_files = sorted(folder_path.rglob("*.*"), key=natural_sort_key)
    total_files = len(input_files)

    if not input_files:
        msg = "No OPUS files found."
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
        return []

    result_data = []
    completed = 0

    def read_file(file_path):
        """
        Reads an OPUS file and extracts spectrum data.

        This function takes the path to an OPUS file, reads its contents, and extracts relevant information
        such as wavenumbers and absorbance values. It also captures the file's last modified timestamp.

        Parameters:
        file_path (str or Path): The path to the OPUS file to be read.

        Returns:
        dict or None: A dictionary containing the processed spectrum data if successful, with the following keys:
            - "file_path": The path to the processed file.
            - "timestamp": The last modified time of the file (in Unix timestamp format).
            - "datetime": The last modified time of the file (as a datetime object).
            - "wavenumbers": The wavenumber values (in cm^-1).
            - "absorbance": The absorbance values corresponding to the wavenumbers.
          Returns None if the wavenumbers or absorbance values are not found.

        Raises:
        Exception: If there is an error reading the file or extracting data, the exception will be raised.

        Example:
        >>> file_path = "/path/to/opus/file"
        >>> spectrum_data = read_file(file_path)
        """
        nonlocal completed
        try:
            timestamp_unix = os.path.getmtime(file_path)
            timestamp_dt = datetime.fromtimestamp(timestamp_unix)
            X = read_opus(file_path)
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
    """
    Interpolates spectrum data to a common wavenumber range.

    This function takes a list of spectrum data entries, each containing wavenumbers and absorbance values,
    and interpolates them to a common range of wavenumbers. The interpolation is performed using NumPy's
    vectorized operations for efficiency.

    Parameters:
    spectrum_data (list): A list of dictionaries, where each dictionary contains:
        - "wavenumbers": A list of wavenumber values (in cm^-1).
        - "absorbance": A list of absorbance values corresponding to the wavenumbers.
    progress_callback (function, optional): A callback function that takes a string message as an argument.
        This function can be used to report progress during the interpolation process. If not provided,
        progress messages will be printed to the console.

    Returns:
    list: A list of dictionaries containing the interpolated spectrum data, where each dictionary has:
        - "wavenumbers": The common wavenumber range.
        - "absorbance": The interpolated absorbance values corresponding to the common wavenumbers.

    Raises:
    ValueError: If the input spectrum_data is empty, a message will be printed or passed to the progress_callback.

    Example:
    >>> spectrum_data = [
    ...     {"wavenumbers": [4000, 3900, 3800], "absorbance": [0.1, 0.2, 0.3]},
    ...     {"wavenumbers": [4050, 3950, 3850], "absorbance": [0.2, 0.3, 0.4]}
    ... ]
    >>> interpolated_data = interpolate_spectrum_data(spectrum_data)
    """
    if not spectrum_data:
        msg = "No spectrum data to interpolate."
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
        return []

    # Convert all wavenumbers and absorbances to 2D NumPy arrays
    wn_arrays = [np.array(entry["wavenumbers"])[::-1]
                 for entry in spectrum_data]
    ab_arrays = [np.array(entry["absorbance"])[::-1]
                 for entry in spectrum_data]

    # Determine common integer wavenumber range
    min_wn = max(w.min() for w in wn_arrays)
    max_wn = min(w.max() for w in wn_arrays)
    common_wavenumbers = np.arange(int(max_wn), int(min_wn) - 1, -1)

    msg = f"Interpolating to common wavenumber range: {common_wavenumbers[0]} to {common_wavenumbers[-1]} ({len(common_wavenumbers)} points)"
    if progress_callback:
        progress_callback(msg)
    else:
        print(msg)

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


def natural_sort_key(s):
    """
    Key function for natural sorting.
    Splits text into digit and non-digit chunks for proper numeric order.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def convert_opus_to_csv(input_folder, output_folder=None, progress_callback=None, max_workers=4):
    """
    Converts all OPUS files in a folder to CSV files containing wavenumber and absorbance data.

    Parameters:
    - input_folder (str or Path): Folder containing Opus files.
    - output_folder (str or Path, optional): Destination folder for .csv files.
      Defaults to 'converted_csv' inside the input folder.
    - progress_callback (function, optional): Function that receives progress messages.
    - max_workers (int, optional): Number of parallel threads for faster conversion.

    Returns:
    - list of Path: Paths to the created CSV files.
    """
    input_folder = Path(input_folder)
    output_folder = Path(
        output_folder) if output_folder else input_folder / "converted_csv"
    output_folder.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(input_folder.rglob("*"), key=natural_sort_key)
    exclude_exts = {".csv", ".txt"}
    opus_files = [
        p for p in all_paths
        if p.is_file() and p.suffix.lower() not in exclude_exts
    ]
    total_files = len(opus_files)

    if not opus_files:
        msg = f"No files found in {input_folder}"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
        return []

    def convert_file(opus_path):
        try:
            X = read_opus(opus_path)
            wavenumbers = X.x.to('1/cm').data
            absorbance = X.data[0]
            base_name = opus_path.name
            csv_path = output_folder / (base_name + ".csv")

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Wavenumber (1/cm)", "Absorbance"])
                writer.writerows(zip(wavenumbers, absorbance))

            return csv_path
        except Exception as e:
            return f"Error processing {opus_path.name}: {e}"

    converted = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_file, f): f for f in opus_files}
        for future in as_completed(futures):
            result = future.result()
            converted.append(result)
            completed += 1
            percent = completed / total_files * 100
            msg = f"{percent:.1f}%"
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

    if progress_callback:
        progress_callback("Conversion complete.")
    else:
        print("Conversion complete.")

    return converted


def convert_opus_to_txt(input_folder, output_folder=None, progress_callback=None, max_workers=4):
    """
    Converts all OPUS files in a folder to TXT files (tab-separated) containing
    wavenumber and absorbance data.

    Parameters:
    - input_folder (str or Path): Folder containing Opus files.
    - output_folder (str or Path, optional): Destination folder for .txt files.
      Defaults to 'converted_txt' inside the input folder.
    - progress_callback (function, optional): Function that receives progress messages.
    - max_workers (int, optional): Number of parallel threads for faster conversion.

    Returns:
    - list of Path: Paths to the created TXT files.
    """
    input_folder = Path(input_folder)
    output_folder = Path(
        output_folder) if output_folder else input_folder / "converted_txt"
    output_folder.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(input_folder.rglob("*"), key=natural_sort_key)
    exclude_exts = {".csv", ".txt"}
    opus_files = [
        p for p in all_paths
        if p.is_file() and p.suffix.lower() not in exclude_exts
    ]
    total_files = len(opus_files)

    if not opus_files:
        msg = f"No files found in {input_folder}"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
        return []

    def convert_file(opus_path):
        try:
            X = read_opus(opus_path)
            wavenumbers = X.x.to('1/cm').data
            absorbance = X.data[0]
            base_name = opus_path.name
            txt_path = output_folder / (base_name + ".txt")

            with open(txt_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["Wavenumber (1/cm)", "Absorbance"])
                writer.writerows(zip(wavenumbers, absorbance))

            return txt_path
        except Exception as e:
            return f"Error processing {opus_path.name}: {e}"

    converted = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_file, f): f for f in opus_files}
        for future in as_completed(futures):
            result = future.result()
            converted.append(result)
            completed += 1
            percent = completed / total_files * 100
            msg = f"{percent:.1f}%"
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

    if progress_callback:
        progress_callback("Conversion complete.")
    else:
        print("Conversion complete.")

    return converted
