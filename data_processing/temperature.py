import math
from pathlib import Path
import numpy as np
import pandas as pd

def process_temperature_data(folder_path, progress_callback=None, num_chunks=100):
    """
    Ultra-fast temperature data loader with periodic progress updates.

    Args:
        folder_path (Path): Directory containing the temperature file.
        progress_callback (callable, optional): For GUI progress updates.
        num_chunks (int): Number of chunks to split the data into for progress updates.

    Returns:
        list of dicts: Each dict contains 'file_path', 'timestamp', and 'temperature'.
    """
    file_paths = list(Path(folder_path).rglob("*.*"))
    if len(file_paths) != 1:
        raise ValueError("The folder must contain exactly one temperature file.")
    file_path = file_paths[0]

    # Read only the columns we need (2=timestamp, 5=temperature)
    df = pd.read_csv(file_path, sep='\t', header=0, usecols=[2, 5], dtype=float, engine='c')
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
            [{"file_path": file_path, "timestamp": t, "temperature": temp}
             for t, temp in zip(chunk.iloc[:, 0], chunk.iloc[:, 1])]
        )

        # Update progress
        if progress_callback:
            percent = math.ceil(min(i + chunk_size, total_rows) / total_rows * 100)
            progress_callback(f"{percent}%")

    return result_data

def interpolate_to_round_seconds(temperature_data):
    """
    Interpolate temperature data to round seconds.
    
    Parameters:
        temperature_data (List[dict]): 
            A list of dictionaries containing temperature data. Each dictionary should have the keys 
            "timestamp" (Unix timestamp) and "temperature" (temperature value in Kelvin).
    
    Returns:
        List[dict]: A list of dictionaries with interpolated temperature data at each second.
    """
    if not temperature_data:
        return []

    # Extract timestamps and temperatures
    timestamps = np.array([entry["timestamp"] for entry in temperature_data])
    temperatures = np.array([entry["temperature"] for entry in temperature_data], dtype=float)

    # Create a new array of round seconds from the minimum to the maximum timestamp
    round_seconds = np.arange(timestamps.min(), timestamps.max() + 1, 1)

    # Interpolate temperatures at the new timestamps
    interpolated_temperatures = np.interp(round_seconds, timestamps, temperatures)

    # Create a new list of dictionaries with interpolated data
    interpolated_data = [{"timestamp": int(ts), "temperature": float(temp)} for ts, temp in zip(round_seconds, interpolated_temperatures)]

    return interpolated_data
