import matplotlib.pyplot as plt
import numpy as np

def plot_spectra(
        spectra_data, start_index=0, end_index=10,
        normalize=False, progress_callback=None, index_offset=False
        ):
    """
    Plot spectra with start/end index range.

    Parameters:
        spectra_data: List of dicts from process_spectrum_data_with_interpolation().
        start_index: First spectrum index to plot.
        end_index: Last spectrum index to plot (inclusive). Use -1 or None to plot all from start_index.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        invert_xaxis: If True, invert X-axis (IR convention).
        normalize: If True, normalize each spectrum to its max(abs).
        figsize: Figure size as (width, height).
        alpha: Line transparency.
        legend: Show legend with spectrum number.
        progress_callback: Optional function to report progress.
        index_offset: If True, use 1-based indexing for start_index and end_index.
    """
    if not spectra_data:
        if progress_callback:
            progress_callback("⚠️ No spectra to plot.")
        else:
            print("⚠️ No spectra to plot.")
        return

    total = len(spectra_data)

    if index_offset:
        # Convert 1-based to 0-based indices
        start_index -= 1
        if end_index is not None and end_index != -1:
            end_index -= 1

    if end_index is None or end_index == -1:
        end_index = total - 1

    # Clip indices to valid range
    start_index = max(0, start_index)
    end_index = min(total - 1, end_index)

    if start_index > end_index:
        if index_offset:
            if progress_callback:
                progress_callback(
                    f"⚠️ Invalid index range: start={start_index + 1}, end={end_index + 1}")
            else:
                print(
                    f"⚠️ Invalid index range: start={start_index + 1}, end={end_index + 1}")
        else:
            if progress_callback:
                progress_callback(
                    f"⚠️ Invalid index range: start={start_index}, end={end_index}")
            else:
                print(
                    f"⚠️ Invalid index range: start={start_index}, end={end_index}")
        return

    spectra_to_plot = spectra_data[start_index:end_index + 1]

    plt.figure(figsize=(12, 6))

    for i, entry in enumerate(spectra_to_plot, start=start_index):
        x = entry["wavenumbers"]
        y = entry["absorbance"]

        if normalize:
            y = np.array(y)
            y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y

        plt.plot(x, y, label=f"Spectrum #{i+1}", alpha=0.7)

    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Absorbance')
    plt.title('Spectra')

    plt.gca().invert_xaxis()

    plt.legend(loc="best", fontsize="small")

    plt.tight_layout()
    plt.show()


def plot_temperature(
        temperature_data,
        start_index=0, end_index=None,
        smooth=False, window_size=3,
        progress_callback=None, index_offset=False
        ):
    """
    Plot temperature data with optional smoothing and index range.

    Parameters:
        temperature_data (List[dict]): 
            A list of dictionaries containing temperature data. Each dictionary should have the keys 
            "timestamp" (Unix timestamp) and "temperature" (temperature value in Kelvin).

        title (str): 
            The title of the plot. Default is 'Temperature Over Time'.

        xlabel (str): 
            The label for the x-axis. Default is 'Time'.

        ylabel (str): 
            The label for the y-axis. Default is 'Temperature (K)'.

        marker (str): 
            The marker style for the data points (e.g., '.', 'o', 's'). Default is '.'.

        linestyle (str): 
            The line style for the plot (e.g., '-', '--', ':'). Default is '-'.

        figsize (tuple): 
            The size of the figure in inches (width, height). Default is (12, 6).

        start_index (int): 
            The index to start plotting from. Default is 0.

        end_index (int or None): 
            The index to stop plotting. If None, includes all data until the end. Default is None.

        smooth (bool): 
            If True, applies smoothing to the temperature data using a moving average. Default is False.

        window_size (int): 
            The size of the moving average window. Must be less than or equal to the length of the data. Default is 3.

        progress_callback (callable or None): 
            A callback function to report progress or warnings. If provided, it will be called with a message string. Default is None.

        index_offset (bool): 
            If True, treats start_index and end_index as 1-based indices. Default is False.

    Returns:
        None: The function does not return any value. It generates a plot of the temperature data.

    Raises:
        ValueError: If the specified window_size for smoothing is greater than the length of the temperature data.
    """
    if not temperature_data:  # No data to plot
        if progress_callback:
            progress_callback("⚠️ No temperatures to plot.")
        else:
            print("⚠️ No temperatures to plot.")
        return

    total = len(temperature_data)

    if index_offset:  # Convert 1-based to 0-based indices
        start_index -= 1
        if end_index is not None and end_index != -1:
            end_index -= 1

    if end_index is None or end_index == -1:
        end_index = total - 1

    # Clip indices to valid range
    start_index = max(0, start_index)
    end_index = min(total - 1, end_index)

    if start_index > end_index:  # Invalid index range
        if index_offset:
            if progress_callback:
                progress_callback(
                    f"⚠️ Invalid index range: start={start_index + 1}, end={end_index + 1}")
            else:
                print(
                    f"⚠️ Invalid index range: start={start_index + 1}, end={end_index + 1}")
        else:
            if progress_callback:
                progress_callback(
                    f"⚠️ Invalid index range: start={start_index}, end={end_index}")
            else:
                print(
                    f"⚠️ Invalid index range: start={start_index}, end={end_index}")
        return

    # Extract time and temperature into NumPy arrays
    times = np.array([entry["timestamp"] for entry in temperature_data])
    temperatures = np.array([entry["temperature"]
                            for entry in temperature_data], dtype=float)

    # Convert Unix timestamps to NumPy datetime64
    time_numeric = np.array([np.datetime64(int(t), 's') for t in times])

    # Slice the data based on start_index and end_index
    time_numeric = time_numeric[start_index:end_index]
    temperatures = temperatures[start_index:end_index]

    # Apply smoothing if requested
    if smooth:
        if window_size > len(temperatures):
            raise ValueError(
                "Window size must be less than or equal to the length of the data.")
        temperatures = moving_average(temperatures, window_size)
        # Adjust time_numeric to match the smoothed data length
        time_numeric = time_numeric[window_size - 1:]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_numeric, temperatures, marker='.', linestyle='-')
    plt.title('Temperature Over Time')
    plt.xlabel('Time')
    plt.ylabel('Temperature (K)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_3d_surface(combined_data, cmap='plasma'):
    """
    Plot a 3D surface: Wavenumber (X) vs Temperature (Y) vs Absorbance (Z)

    Parameters:
    - combined_data: List of dictionaries containing 'wavenumbers', 'temperature', and 'absorbance'.
    - cmap: Colormap for the surface (default: plasma).
    """
    # Stack all data
    wavenumbers = np.array(combined_data[0]["wavenumbers"])
    temperatures = np.array([entry["temperature"] for entry in combined_data])
    absorbances = np.array([entry["absorbance"] for entry in combined_data])

    # Create meshgrid
    WN, TEMP = np.meshgrid(wavenumbers, temperatures)
    AB = absorbances

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(WN, TEMP, AB, cmap=cmap, edgecolor='none')

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Temperature (K)")
    ax.set_zlabel("Absorbance")
    ax.set_title("3D Absorbance Surface")

    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()


def moving_average(data, window_size):
    """Calculate the moving average of the data."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_absorption_vs_temperature(
    combined_list,
    start_wavelength,
    end_wavelength,
    smooth=False,
    window_size=5,
    figsize=(10, 6),
    marker='o',
    linestyle='-',
    smooth_color='red',
    raw_color='blue',
    title="Absorption vs Temperature"
):
    """
    Plot summed absorbance vs temperature with optional moving average smoothing.

    Parameters:
        combined_list (list of dict): Each dict should have "temperature", "wavenumbers", "absorbance".
        start_wavelength (float): Start of wavelength range.
        end_wavelength (float): End of wavelength range.
        smooth (bool): If True, overlay smoothed data.
        window_size (int): Moving average window size for smoothing (in number of points).
        figsize (tuple): Figure size.
        marker (str): Marker style for the raw plot.
        linestyle (str): Line style for the raw plot.
        smooth_color (str): Color for smoothed curve.
        raw_color (str): Color for raw curve.
        title (str): Plot title.
    """
    if not combined_list:
        raise ValueError("No combined data to plot.")

    # Extract temperatures and summed absorbance
    temperatures = np.array([e["temperature"] for e in combined_list])
    summed_absorbance = np.array([
        np.sum(e["absorbance"][(e["wavenumbers"] >= start_wavelength) &
                               (e["wavenumbers"] <= end_wavelength)])
        for e in combined_list
    ])

    # Sort by temperature
    sort_idx = np.argsort(temperatures)
    temperatures = temperatures[sort_idx]
    summed_absorbance = summed_absorbance[sort_idx]

    # Plot raw data
    plt.figure(figsize=figsize)
    plt.plot(temperatures, summed_absorbance, marker=marker, linestyle=linestyle, color=raw_color, label="Raw")

    if smooth:
        smoothed_abs = np.zeros_like(summed_absorbance, dtype=float)

        for i, T in enumerate(temperatures):
            # Find indices within smooth_width/2 around current temperature
            mask = np.abs(temperatures - T) <= window_size / 2
            smoothed_abs[i] = np.mean(summed_absorbance[mask])

        # Overlay smoothed curve
        plt.plot(temperatures, smoothed_abs, marker='', linestyle='-', color=smooth_color, linewidth=2, label="Smoothed")

    plt.xlabel("Temperature (K)")
    plt.ylabel(f"Summed Absorbance ({start_wavelength}-{end_wavelength})")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()