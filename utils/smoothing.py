import numpy as np
from scipy.interpolate import UnivariateSpline


def smooth_combined_by_temperature(combined_data, temp_window=1.0):
    """
    Smooth absorbance data across temperature dimension.

    Args:
        combined_data (list[dict]): each entry has "temperature", "wavenumbers", "absorbance"
        temp_window (float): smoothing window size in temperature units

    Returns:
        list[dict]: smoothed combined_data
    """
    if not combined_data:
        return combined_data

    temperatures = np.array([d["temperature"] for d in combined_data])
    wavenumbers = np.array(combined_data[0]["wavenumbers"])
    absorbances = np.array([d["absorbance"] for d in combined_data])

    smoothed_abs = []
    for i, T in enumerate(temperatures):
        weights = np.exp(-0.5 * ((temperatures - T) / temp_window) ** 2)
        weights /= weights.sum()
        smoothed_abs.append(np.average(absorbances, axis=0, weights=weights))

    smoothed_data = []
    for T, A in zip(temperatures, smoothed_abs):
        smoothed_data.append({
            "temperature": T,
            "wavenumbers": wavenumbers,
            "absorbance": A,
        })

    return smoothed_data


def moving_average(data, window_size):
    """Calculate the moving average of the data."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def spline_safe_normalized(x, y, x_eval, strength=0.5):
    if UnivariateSpline is None:
        raise RuntimeError("SciPy is required for spline smoothing")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_min, y_max = np.min(y), np.max(y)
    y_norm = (y - y_min) / (y_max - y_min) if y_max - y_min > 0 else y - y_min
    s_min = 1e-12
    s_max = np.sum((y_norm - np.mean(y_norm)) ** 2)
    s_val = s_min + strength * (s_max - s_min)
    spline = UnivariateSpline(x, y_norm, s=s_val)
    y_smooth_norm = spline(x_eval)
    return y_smooth_norm * (y_max - y_min) + y_min


def gaussian_kernel_smooth(x, y, bandwidth, x_eval=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_eval is None:
        x_eval = x
    x_col = x.reshape(-1, 1)
    x_eval_row = np.asarray(x_eval).reshape(1, -1)
    w = np.exp(-0.5 * ((x_col - x_eval_row) / float(bandwidth))**2)
    y_s = (w * y.reshape(-1, 1)).sum(axis=0) / \
        np.clip(w.sum(axis=0), 1e-12, None)
    return np.asarray(x_eval), y_s


def loess_1d(x, y, frac=0.3, degree=1, x_eval=None):
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
        idx = np.argpartition(dist, r - 1)[:r]
        di = dist[idx]
        dmax = di.max()
        if dmax == 0:
            y_fit[j] = y[idx].mean()
            continue
        w = (1 - (di / dmax) ** 3) ** 3
        X = np.vander(x[idx] - x0, N=degree + 1, increasing=True)
        Wsqrt = np.sqrt(w)
        Xw = X * Wsqrt[:, None]
        yw = y[idx] * Wsqrt
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        y_fit[j] = beta[0]
    return x_eval, y_fit


def estimate_bandwidth(temperatures):
    """Estimate a 1D bandwidth (K) from temperature array using Silverman's rule of thumb."""
    x = np.asarray(temperatures, dtype=float)
    n = len(x)
    if n < 2:
        return 1.0
    sd = np.std(x, ddof=1)
    # Silverman: 1.06 * sd * n^(-1/5)
    return float(1.06 * sd * n ** (-1/5))
