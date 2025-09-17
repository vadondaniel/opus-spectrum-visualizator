import os
import tempfile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

from utils.smoothing import gaussian_kernel_smooth, loess_1d, moving_average, smooth_combined_by_temperature, spline_safe_normalized


def plot_spectra(ax, spectra_data, start_index=0, end_index=None, normalize=False):
    """
    Plot spectra on the given matplotlib Axes.
    Returns nothing; just draws on the provided ax.
    """
    if end_index is None:
        end_index = len(spectra_data) - 1

    spectra_to_plot = spectra_data[start_index:end_index+1]

    for i, entry in enumerate(spectra_to_plot, start=start_index):
        x = entry["wavenumbers"]
        y = np.array(entry["absorbance"])
        if normalize:
            y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y

        # Optional baseline correction
        # y = baseline_correction(x, y)

        ax.plot(x, y, label=f"Spectrum #{i+1}", alpha=0.7)

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Spectra")


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


def plot_3d(
    combined_data,
    plot_type="Surface",
    cmap="plasma",
    max_points=2_000_000,
    smoothing_factor: int | None = None,
):
    if not combined_data or not isinstance(combined_data, list) or not isinstance(combined_data[0], dict):
        raise ValueError("Data format invalid.")

    # Apply smoothing if requested
    if smoothing_factor is not None and smoothing_factor != 0:
        try:
            temp_window = float(smoothing_factor) / 10
            combined_data = smooth_combined_by_temperature(
                combined_data, temp_window=temp_window)
        except Exception as e:
            raise ValueError(f"Smoothing failed: {e}")

    # Convert to arrays
    wavenumbers = np.asarray(combined_data[0]["wavenumbers"])
    temperatures = np.asarray([entry["temperature"]
                              for entry in combined_data])
    absorbances = np.asarray([entry["absorbance"] for entry in combined_data])

    # Downsample if too big
    M, N = absorbances.shape
    if M * N > max_points:
        step_m = max(1, int(np.ceil(M / np.sqrt(max_points / N))))
        step_n = max(1, int(np.ceil(N / np.sqrt(max_points / M))))
        temperatures = temperatures[::step_m]
        wavenumbers = wavenumbers[::step_n]
        absorbances = absorbances[::step_m, ::step_n]

    # Create Plotly figure depending on type
    if plot_type == "Surface":
        fig = go.Figure(data=[
            go.Surface(z=absorbances, x=wavenumbers,
                       y=temperatures, colorscale=cmap)
        ])
    elif plot_type == "Scatter":
        T, W = np.meshgrid(temperatures, wavenumbers, indexing="ij")
        fig = go.Figure(data=[
            go.Scatter3d(
                x=W.flatten(),
                y=T.flatten(),
                z=absorbances.flatten(),
                mode="markers",
                marker=dict(size=2, color=absorbances.flatten(),
                            colorscale=cmap)
            )
        ])
    elif plot_type == "Contour":
        step_m = max(1, int(np.ceil(M / np.sqrt(max_points / N))))
        step_n = max(1, int(np.ceil(N / np.sqrt(max_points / M))))
        T_ds = temperatures[::step_m]
        W_ds = wavenumbers[::step_n]
        A_ds = absorbances[::step_m, ::step_n]

        fig = go.Figure(data=[
            go.Contour(
                z=A_ds,
                x=W_ds,
                y=T_ds,
                colorscale=cmap,
                contours=dict(showlines=True, showlabels=True,
                              labelfont=dict(size=10))
            )
        ])
        fig.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Temperature (K)"
        )
    elif plot_type == "Heatmap":
        fig = go.Figure(data=[
            go.Heatmap(
                z=absorbances,
                x=wavenumbers,
                y=temperatures,
                colorscale=cmap
            )
        ])
        fig.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Temperature (K)"
        )

    # Layout
    fig.update_layout(title=f"Absorbance {plot_type} Plot")
    fig.update_layout(
        scene=dict(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Temperature (K)",
            zaxis_title="Absorbance"
        ),
        autosize=True
    )

    # Save HTML to temp file
    tmp_html = os.path.join(tempfile.gettempdir(), "plotly_3d.html")
    fig.write_html(tmp_html, include_plotlyjs='inline')

    return tmp_html


def plot_peak_analysis(
    combined_list,
    wavelength_ranges,
    display_type="both",
    smoothing="gaussian",
    smoothing_param=None,
    window_size=5,
    eval_on_grid=True,
    grid_points=300,
    ax=None
):
    """
    Compute and optionally plot summed absorbance vs temperature.
    Returns (temperatures, dict of {range: summed_absorbance}).
    If ax is provided, plots on the given matplotlib Axes; otherwise creates new figure.
    """
    if not combined_list:
        return None, None

    temperatures = np.array([e["temperature"]
                            for e in combined_list], dtype=float)
    sort_idx = np.argsort(temperatures)
    temperatures = temperatures[sort_idx]

    T_grid = np.linspace(temperatures.min(), temperatures.max(),
                         grid_points) if eval_on_grid else None
    results = {}
    base_colors = plt.cm.tab10.colors
    base_markers = ["o", "s", "D", "*", "X", "^", "h", "p", "v", "d"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    for i, (start_w, end_w) in enumerate(wavelength_ranges):
        color = base_colors[i % len(base_colors)]
        marker = base_markers[i % len(base_markers)]

        summed_absorbance = np.array([
            np.sum(e["absorbance"][(e["wavenumbers"] >= start_w)
                   & (e["wavenumbers"] <= end_w)])
            for e in combined_list
        ], dtype=float)[sort_idx]

        summed_absorbance = baseline_correction(
            temperatures, summed_absorbance)
        results[(start_w, end_w)] = summed_absorbance

        # --- smoothing ---
        x_eval = T_grid if eval_on_grid else temperatures
        smoothed = None
        param = smoothing_param if smoothing_param is not None else (
            window_size if smoothing in ("boxcar", "gaussian") else 0.3)

        if display_type in ("smoothed", "both") and smoothing != "none":
            if smoothing == "boxcar":
                width = float(param)
                smoothed = np.array([
                    np.mean(summed_absorbance[np.abs(
                        temperatures - T0) <= width / 2])
                    for T0 in x_eval
                ])
            elif smoothing == "gaussian":
                _, smoothed = gaussian_kernel_smooth(
                    temperatures, summed_absorbance, float(param), x_eval=x_eval)
            elif smoothing == "loess":
                _, smoothed = loess_1d(
                    temperatures, summed_absorbance, frac=float(param), x_eval=x_eval)
            elif smoothing == "spline":
                smoothed = spline_safe_normalized(
                    temperatures, summed_absorbance, x_eval, float(param))

        # --- plotting ---
        if display_type in ("raw", "both"):
            ax.plot(temperatures, summed_absorbance, marker=marker,
                    linestyle='-', color=color, label=f"{start_w}-{end_w}")
        if smoothed is not None:
            ax.plot(x_eval, smoothed, linestyle='--', linewidth=2, color=color)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Integrated Absorbance")
    if len(wavelength_ranges) == 1:
        title = "Peak Analysis"
    elif len(wavelength_ranges) == 2:
        title = "Dual Peak Analysis"
    elif len(wavelength_ranges) == 3:
        title = "Triple Peak Analysis"
    else:
        title = f"Multi-Peak Analysis ({len(wavelength_ranges)} ranges)"
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return temperatures, results


def baseline_correction(temperatures, absorbance):
    """Linear baseline between first and last points."""
    x0, x1 = temperatures[0], temperatures[-1]
    y0, y1 = absorbance[0], absorbance[-1]
    baseline = y0 + (y1 - y0) * (temperatures - x0) / (x1 - x0)
    return absorbance - (baseline - y0)
