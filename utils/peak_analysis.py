import numpy as np


def baseline_correction(wavenumbers, absorbance, mode="none"):
    """
    Baseline correction for absorbance data.

    Parameters
    ----------
    wavenumbers : array-like
        X values (wavenumbers, can be unevenly spaced, ascending or descending).
    absorbance : array-like
        Y values (absorbance).
    mode : {"none", "trapezoid"}, default "none"
        - "none": no correction
        - "trapezoid": add +1 to absorbance, subtract trapezoid baseline
                       (line between endpoints relative to x-axis)
    """
    wavns = np.asarray(wavenumbers, dtype=float)
    absb = np.asarray(absorbance, dtype=float).copy()

    if mode == "none":
        return absb

    if mode == "trapezoid":
        shifted = absb + 1.0
        if len(wavns) >= 2 and wavns[0] != wavns[-1]:
            x0, x1 = wavns[0], wavns[-1]
            y0, y1 = shifted[0], shifted[-1]
            slope = (y1 - y0) / (x1 - x0)
            baseline = y0 + slope * (wavns - x0)
            return shifted - baseline
        return shifted

    raise ValueError(f"Unknown baseline correction mode: {mode!r}")


def integrate_area_variable_dx(wavenumbers, corrected_absorbance):
    """
    Compute area under corrected_absorbance using variable intervals on x-axis.
    Uses right-rectangle rule: sum(|dx_i| * y_i_right), i = 1..N-1.
    """
    wn = np.asarray(wavenumbers, dtype=float)
    y = np.asarray(corrected_absorbance, dtype=float)
    if wn.size < 2:
        return float(np.sum(y))
    widths = np.abs(np.diff(wn))
    return float(np.sum(widths * y[1:]))


def compute_peak_areas(combined_list, wavelength_ranges, baseline_mode="none"):
    """
    For each (start, end) range, compute peak areas per entry in combined_list using
    trapezoid baseline correction (if requested) and width-weighted integration.

    Returns temperatures_sorted, results dict mapping (start, end) -> np.ndarray
    of areas sorted by ascending temperature.
    """
    if not combined_list:
        return None, {}

    temperatures = np.array([e["temperature"] for e in combined_list], dtype=float)
    sort_idx = np.argsort(temperatures)
    temperatures_sorted = temperatures[sort_idx]

    results = {}
    for (start_w, end_w) in wavelength_ranges:
        areas = []
        for e in combined_list:
            wn = np.asarray(e["wavenumbers"], dtype=float)
            ab = np.asarray(e["absorbance"], dtype=float)
            mask = (wn >= start_w) & (wn <= end_w)
            wn_slice = wn[mask]
            ab_slice = ab[mask]
            if wn_slice.size == 0:
                areas.append(0.0)
                continue
            ab_corr = baseline_correction(wn_slice, ab_slice, mode=baseline_mode)
            area = integrate_area_variable_dx(wn_slice, ab_corr)
            areas.append(area)
        results[(start_w, end_w)] = np.array(areas, dtype=float)[sort_idx]

    return temperatures_sorted, results

