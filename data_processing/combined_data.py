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
