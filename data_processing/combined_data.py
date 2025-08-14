import numpy as np

def combine_temperature_and_spectrum_data(temperature_data, spectrum_data, time_buffer=60):
    """
    Match temperature data to each spectrum based on timestamp.

    Args:
        temperature_data (List[dict]): List of temperature entries with 'timestamp' and 'temperature'.
        spectrum_data (List[dict]): List of spectrum entries with 'timestamp' and interpolated absorbance data.
        time_buffer (int): Seconds to add before/after spectrum range for cutting temp data.

    Returns:
        combined_data (List[dict]): Each entry contains:
            - 'timestamp': Spectrum timestamp
            - 'datetime': Spectrum datetime
            - 'temperature': Interpolated temperature at that timestamp
            - 'wavenumbers': Interpolated wavenumbers
            - 'absorbance': Interpolated absorbance
    """
    if not temperature_data or not spectrum_data:
        print("❌ Missing temperature or spectrum data.")
        return []

    # Extract and sort timestamps
    temp_times = np.array([entry["timestamp"] for entry in temperature_data])
    temp_vals = np.array([entry["temperature"] for entry in temperature_data])

    # Define spectrum time window with buffer
    spec_times = [entry["timestamp"] for entry in spectrum_data]
    start_time = min(spec_times) - time_buffer
    end_time = max(spec_times) + time_buffer

    # Trim temperature data to relevant window
    mask = (temp_times >= start_time) & (temp_times <= end_time)
    trimmed_times = temp_times[mask]
    trimmed_vals = temp_vals[mask]

    if len(trimmed_times) == 0:
        print("⚠️ No temperature data in the spectrum time range.")
        return []

    # Interpolate temperature at spectrum timestamps
    combined_data = []
    for spec in spectrum_data:
        t_spec = spec["timestamp"]
        try:
            interp_temp = float(np.interp(t_spec, trimmed_times, trimmed_vals))
        except Exception as e:
            print(f"Interpolation error at {t_spec}: {e}")
            interp_temp = None

        combined_data.append({
            "timestamp": t_spec,
            "datetime": spec["datetime"],
            "temperature": interp_temp,
            "wavenumbers": spec["wavenumbers"],
            "absorbance": spec["absorbance"],
        })

    return combined_data

def smooth_combined_by_temperature(combined_list, temp_window=2.0):
    """
    Smooth combined_list along the temperature (Y) axis.
    
    Args:
        combined_list (list of dict): Each dict must have "temperature", "wavenumbers", "absorbance".
        temp_window (float): Temperature bin width (K). Average all spectra within each bin.
        
    Returns:
        List[dict]: Smoothed combined list with averaged spectra per temperature bin.
    """
    if not combined_list:
        return []

    # Extract temperatures and spectra
    temps = np.array([item["temperature"] for item in combined_list])
    wavenumbers = combined_list[0]["wavenumbers"]  # assume all spectra share wavenumbers
    absorbances = np.array([item["absorbance"] for item in combined_list])

    # Define temperature bins
    t_min, t_max = temps.min(), temps.max()
    bins = np.arange(t_min, t_max + temp_window, temp_window)

    smoothed_list = []

    for i in range(len(bins) - 1):
        mask = (temps >= bins[i]) & (temps < bins[i + 1])
        if not mask.any():
            continue  # skip empty bins

        avg_temp = temps[mask].mean()
        avg_abs = absorbances[mask].mean(axis=0)  # average absorbance at each wavenumber

        smoothed_list.append({
            "timestamp": np.mean([combined_list[j]["timestamp"] for j in range(len(combined_list)) if mask[j]]),
            "datetime": combined_list[np.where(mask)[0][0]]["datetime"],  # pick first datetime in bin
            "temperature": avg_temp,
            "wavenumbers": wavenumbers,
            "absorbance": avg_abs
        })

    return smoothed_list