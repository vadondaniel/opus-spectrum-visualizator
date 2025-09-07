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
        normalize: If True, normalize each spectrum to its max(abs).
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

def plot_absorption_vs_temperature(
    combined_list,
    wavelength_ranges,
    window_size=5,                # kept for backward-compat; interpreted in K if used
    display_type="both",
    smoothing="gaussian",         # "boxcar", "gaussian", "loess", "spline", or "none"
    smoothing_param=None,         # boxcar/gaussian: bandwidth in K; loess: frac (0..1); spline: s
    eval_on_grid=False,           # if True, evaluate smoothed curve on a dense grid for prettier lines
    grid_points=300
):
    """
    Plot summed absorbance vs temperature with selectable smoothing.

    Parameters:
        combined_list (list of dict): Each dict has "temperature", "wavenumbers", "absorbance".
        wavelength_ranges (list[tuple[float,float]]): [(start_wavenumber, end_wavenumber), ...]
        window_size (float): Backward-compat. If smoothing='boxcar' and smoothing_param is None,
                             this is used as the temperature window *in K*.
        display_type (str): "raw", "smoothed", or "both".
        smoothing (str): "boxcar" (uniform kernel in K), "gaussian" (bandwidth in K),
                         "loess" (local linear; smoothing_param=frac 0..1),
                         "spline" (requires SciPy; smoothing_param=s),
                         or "none".
        smoothing_param: See above.
        eval_on_grid (bool): If True, smoothed curves are drawn on an evenly spaced temperature grid.
        grid_points (int): Number of grid points when eval_on_grid=True.
    """
    if not combined_list:
        raise ValueError("No combined data to plot.")

    # Sort by temperature
    temperatures = np.array([e["temperature"] for e in combined_list], dtype=float)
    sort_idx = np.argsort(temperatures)
    temperatures = temperatures[sort_idx]

    plt.figure(figsize=(10, 6))

    # Discrete tableau colors (10 highly distinct colors)
    base_colors = plt.cm.tab10.colors
    base_markers = ["o", "s", "D", "*", "X", "^", "h", "p", "v", "d"]

    # Prepare dense grid if requested
    T_grid = None
    if eval_on_grid:
        T_grid = np.linspace(temperatures.min(), temperatures.max(), int(grid_points))

    for i, (start_wavelength, end_wavelength) in enumerate(wavelength_ranges):
        color  = base_colors[i % len(base_colors)]      # cycle if >10 ranges
        marker = base_markers[i % len(base_markers)]

        # Sum/integrate absorbance in the wavelength window for each spectrum
        # NOTE: if your wavenumbers have non-uniform spacing, consider np.trapz instead of np.sum.
        summed_absorbance = np.array([
            np.sum(e["absorbance"][(e["wavenumbers"] >= start_wavelength) &
                                   (e["wavenumbers"] <= end_wavelength)])
            for e in combined_list
        ], dtype=float)[sort_idx]

        if display_type in ("raw", "both"):
            plt.plot(
                temperatures,
                summed_absorbance,
                marker=marker,
                linestyle='-',
                color=color,
                label=f"{start_wavelength}-{end_wavelength} raw"
            )

        if display_type in ("smoothed", "both") and smoothing != "none":
            # Decide parameter defaults and evaluation x-axis
            if smoothing_param is None:
                smoothing_param = window_size if smoothing in ("boxcar", "gaussian") else (0.3 if smoothing=="loess" else 0.0)

            x_eval = T_grid if eval_on_grid else temperatures

            if smoothing == "boxcar":
                width = float(smoothing_param)
                # Uniform (boxcar) kernel in temperature units (K)
                smoothed = np.array([
                    np.mean(summed_absorbance[np.abs(temperatures - T0) <= width/2]) 
                    for T0 in x_eval
                ], dtype=float)

            elif smoothing == "gaussian":
                _, smoothed = _gaussian_kernel_smooth(temperatures, summed_absorbance,
                                                      bandwidth=float(smoothing_param),
                                                      x_eval=x_eval)

            elif smoothing == "loess":
                _, smoothed = _loess_1d(temperatures, summed_absorbance,
                                        frac=float(smoothing_param),
                                        degree=1, x_eval=x_eval)

            elif smoothing == "spline":
                try:
                    from scipy.interpolate import UnivariateSpline
                    s = float(smoothing_param) if smoothing_param is not None else None
                    k = 3
                    spl = UnivariateSpline(temperatures, summed_absorbance, s=s, k=k)
                    x_eval = T_grid if eval_on_grid else temperatures
                    smoothed = spl(x_eval)
                except Exception:
                    # Fallback to Gaussian if SciPy unavailable
                    _, smoothed = _gaussian_kernel_smooth(temperatures, summed_absorbance,
                                                          bandwidth=float(window_size if smoothing_param is None else smoothing_param),
                                                          x_eval=x_eval)

            else:
                smoothed = None  # nothing

            if smoothed is not None:
                plt.plot(
                    x_eval,
                    smoothed,
                    linestyle='--',
                    linewidth=2,
                    color=color,
                    label=f"{start_wavelength}-{end_wavelength} {smoothing}"
                )

    plt.xlabel("Temperature (K)")
    plt.ylabel("Integrated Absorbance (Area under curve)")
    if len(wavelength_ranges) == 1:
        title = "Peak Analysis"
    elif len(wavelength_ranges) == 2:
        title = "Dual Peak Analysis"
    elif len(wavelength_ranges) == 3:
        title = "Triple Peak Analysis"
    else:
        title = f"Multi-Peak Analysis ({len(wavelength_ranges)} ranges)"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def moving_average(data, window_size):
    """Calculate the moving average of the data."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def _gaussian_kernel_smooth(x, y, bandwidth, x_eval=None):
    """Gaussian kernel smoother with bandwidth in same units as x (e.g., K)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_eval is None:
        x_eval = x
    x_col = x.reshape(-1, 1)
    x_eval_row = np.asarray(x_eval).reshape(1, -1)
    w = np.exp(-0.5 * ((x_col - x_eval_row) / float(bandwidth))**2)  # (n, m)
    y_s = (w * y.reshape(-1, 1)).sum(axis=0) / np.clip(w.sum(axis=0), 1e-12, None)
    return np.asarray(x_eval), y_s

def _loess_1d(x, y, frac=0.3, degree=1, x_eval=None):
    """
    Minimal LOESS (tricube weights, local polynomial of given degree).
    frac is the fraction of points used in each local fit (0 < frac <= 1).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    r = max(2, int(np.ceil(frac * n)))
    if x_eval is None:
        x_eval = x
    x_eval = np.asarray(x_eval, dtype=float)

    y_fit = np.empty_like(x_eval, dtype=float)
    for j, x0 in enumerate(x_eval):
        dist = np.abs(x - x0)
        idx = np.argpartition(dist, r-1)[:r]         # r nearest neighbors
        di = dist[idx]
        dmax = di.max()
        if dmax == 0:
            y_fit[j] = y[idx].mean()
            continue
        w = (1 - (di / dmax)**3)**3                  # tricube weights
        # Center x at x0 for numerical stability
        X = np.vander(x[idx] - x0, N=degree+1, increasing=True)  # [1, (x-x0), ...]
        # Weighted least squares using sqrt weights
        Wsqrt = np.sqrt(w)
        Xw = X * Wsqrt[:, None]
        yw = y[idx] * Wsqrt
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        y_fit[j] = beta[0]                           # at x0, (x-x0)=0 → value = intercept
    return x_eval, y_fit