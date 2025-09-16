import math
import numpy as np
import pandas as pd


def process_temperature_data(file_path, progress_callback=None, num_chunks=100):
    """
    Processes temperature data from a specified file.

    This function opens a temperature data file, reads the relevant columns (timestamp and temperature),
    and processes the data in chunks. The processing can report progress through a callback function.

    Parameters:
    file_path (str or Path): The path to the temperature data file.
    progress_callback (function, optional): A callback function that takes a string message as an argument.
        This function can be used to report progress during the data processing. If not provided,
        progress messages will be printed to the console.
    num_chunks (int, optional): The number of chunks to divide the data into for processing. Default is 100.

    Returns:
    list: A list of dictionaries containing the processed temperature data, where each dictionary has:
        - "timestamp": The Unix timestamp value from the data.
        - "temperature": The temperature value corresponding to the timestamp. (in Kelvin)

    Raises:
    ValueError: If the specified folder does not contain exactly one temperature file.

    Example:
    >>> file_path = "/path/to/temperature/data"
    >>> temperature_data = process_temperature_data(file_path)
    """
    # Read only the columns we need (2=timestamp, 3=temperature)
    df = pd.read_csv(file_path, sep='\t', header=0, usecols=[
                     2, 3], dtype=float, engine='c')
    df = df.dropna()
    total_rows = len(df)

    if total_rows == 0:
        if progress_callback:
            progress_callback("100%")
        return []

    # Determine chunk size
    chunk_size = max(1, total_rows // num_chunks)
    result_data = []

    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        # Vectorized conversion to list of dicts
        result_data.extend(
            [{"timestamp": t, "temperature": temp}
             for t, temp in zip(chunk.iloc[:, 0], chunk.iloc[:, 1])]
        )

        # Update progress
        if progress_callback:
            percent = math.ceil(
                min(i + chunk_size, total_rows) / total_rows * 100)
            progress_callback(f"{percent}%")

    return result_data


def interpolate_to_round_seconds(temperature_data):
    """
    Interpolates temperature data to round second timestamps.

    This function takes a list of temperature data entries, each containing a Unix timestamp and a temperature value
    in Kelvin, and interpolates the temperatures to round second intervals between the minimum and maximum timestamps.

    Parameters:
    temperature_data (list): A list of dictionaries, where each dictionary contains:
        - "timestamp": The Unix timestamp value.
        - "temperature": The temperature value in Kelvin corresponding to the timestamp.

    Returns:
    list: A list of dictionaries containing the interpolated temperature data, where each dictionary has:
        - "timestamp": The rounded Unix timestamp.
        - "temperature": The interpolated temperature value in Kelvin corresponding to the rounded timestamp.

    Example:
    >>> temperature_data = [
    ...     {"timestamp": 1609459200, "temperature": 293.15},  # Jan 1, 2021, 00:00:00 UTC
    ...     {"timestamp": 1609459260, "temperature": 294.15},  # Jan 1, 2021, 00:01:00 UTC
    ...     {"timestamp": 1609459320, "temperature": 295.15}   # Jan 1, 2021, 00:02:00 UTC
    ... ]
    >>> interpolated_data = interpolate_to_round_seconds(temperature_data)
    """
    if not temperature_data:
        return []

    # Extract timestamps and temperatures
    timestamps = np.array([entry["timestamp"] for entry in temperature_data])
    temperatures = np.array([entry["temperature"]
                            for entry in temperature_data], dtype=float)

    # Create a new array of round seconds from the minimum to the maximum timestamp
    round_seconds = np.arange(timestamps.min(), timestamps.max() + 1, 1)

    # Interpolate temperatures at the new timestamps
    interpolated_temperatures = np.interp(
        round_seconds, timestamps, temperatures)

    # Create a new list of dictionaries with interpolated data
    interpolated_data = [{"timestamp": int(ts), "temperature": float(
        temp)} for ts, temp in zip(round_seconds, interpolated_temperatures)]

    return interpolated_data
