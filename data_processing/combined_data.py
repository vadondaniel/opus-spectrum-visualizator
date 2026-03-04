from pathlib import Path
import numpy as np


def combine_temperature_and_spectrum_data(temperature_data, spectrum_data, time_buffer=60):
    """
    Match temperature data to each spectrum based on timestamp.
    """
    if not temperature_data or not spectrum_data:
        print("Missing temperature or spectrum data.")
        return []

    temp_times = np.array([entry["timestamp"] for entry in temperature_data], dtype=float)
    temp_vals = np.array([entry["temperature"] for entry in temperature_data], dtype=float)

    spec_times = np.array([entry["timestamp"] for entry in spectrum_data], dtype=float)
    start_time = float(np.min(spec_times) - time_buffer)
    end_time = float(np.max(spec_times) + time_buffer)

    mask = (temp_times >= start_time) & (temp_times <= end_time)
    trimmed_times = temp_times[mask]
    trimmed_vals = temp_vals[mask]

    if len(trimmed_times) == 0:
        print("No temperature data in the spectrum time range.")
        return []

    combined_data = []
    for spec in spectrum_data:
        t_spec = float(spec["timestamp"])
        try:
            interp_temp = float(np.interp(t_spec, trimmed_times, trimmed_vals))
        except Exception as exc:  # noqa: BLE001
            print(f"Interpolation error at {t_spec}: {exc}")
            interp_temp = None

        combined_data.append({
            "timestamp": t_spec,
            "datetime": spec["datetime"],
            "temperature": interp_temp,
            "wavenumbers": spec["wavenumbers"],
            "absorbance": spec["absorbance"],
        })

    return combined_data


def _normalize_name_variants(text: str) -> set[str]:
    raw = str(text or "").strip().lower().replace("\\", "/")
    if not raw:
        return set()

    base = raw.split("/")[-1]
    variants = {raw, base}

    # Add stem only for regular alphabetic file extensions (e.g. .txt),
    # not for numeric OPUS-style suffixes like ".11" which are part of the id.
    suffix = Path(base).suffix
    if suffix and any(ch.isalpha() for ch in suffix[1:]):
        stem = Path(base).stem
        if stem:
            variants.add(stem.lower())

    return {v for v in variants if v}


def _build_entry(spec, value: float, variable_name: str):
    return {
        "timestamp": float(spec["timestamp"]),
        "datetime": spec["datetime"],
        "temperature": float(value),  # Keep legacy key for compatibility in existing plotting/export code.
        "wavenumbers": spec["wavenumbers"],
        "absorbance": spec["absorbance"],
        "custom_variable_name": variable_name,
        "custom_variable_value": float(value),
    }


def _combine_custom_by_filename(entries, spectrum_data, variable_name: str):
    value_by_variant = {}
    for row in entries:
        name = row.get("filename")
        value = row.get("value")
        if name is None or value is None:
            continue
        for variant in _normalize_name_variants(name):
            if variant not in value_by_variant:
                value_by_variant[variant] = float(value)

    combined = []
    for spec in spectrum_data:
        spec_variants = _normalize_name_variants(spec.get("file_path", ""))
        matched_value = None
        for variant in spec_variants:
            if variant in value_by_variant:
                matched_value = value_by_variant[variant]
                break
        if matched_value is None:
            continue
        combined.append(_build_entry(spec, matched_value, variable_name))

    return combined


def _combine_custom_by_timestamp(entries, spectrum_data, variable_name: str, time_buffer=60):
    if not entries:
        return []

    x_times = np.array([float(e["timestamp"]) for e in entries], dtype=float)
    x_vals = np.array([float(e["value"]) for e in entries], dtype=float)

    spec_times = np.array([float(entry["timestamp"]) for entry in spectrum_data], dtype=float)
    start_time = float(np.min(spec_times) - time_buffer)
    end_time = float(np.max(spec_times) + time_buffer)

    mask = (x_times >= start_time) & (x_times <= end_time)
    trimmed_times = x_times[mask]
    trimmed_vals = x_vals[mask]
    if len(trimmed_times) == 0:
        return []

    combined = []
    for spec in spectrum_data:
        t_spec = float(spec["timestamp"])
        value = float(np.interp(t_spec, trimmed_times, trimmed_vals))
        combined.append(_build_entry(spec, value, variable_name))
    return combined


def _combine_custom_by_sequence(entries, spectrum_data, variable_name: str):
    values = [float(e["value"]) for e in entries if "value" in e]
    if not values:
        return []
    if len(values) != len(spectrum_data):
        raise ValueError(
            f"Sequence mode requires exactly one value per spectrum. "
            f"Got {len(values)} values for {len(spectrum_data)} spectra."
        )

    ordered_specs = sorted(
        list(enumerate(spectrum_data)),
        key=lambda item: float(item[1].get("timestamp", 0.0))
    )
    value_by_original_index = {
        original_index: value
        for (original_index, _spec), value in zip(ordered_specs, values)
    }

    combined = []
    for idx, spec in enumerate(spectrum_data):
        combined.append(_build_entry(spec, value_by_original_index[idx], variable_name))
    return combined


def combine_custom_variable_and_spectrum_data(custom_variable_data, spectrum_data, time_buffer=60):
    """
    Match custom variable data to spectra.

    custom_variable_data format:
      {
        "mode": "filename" | "timestamp" | "sequence",
        "variable_name": str,
        "entries": list[dict]
      }
    """
    if not custom_variable_data or not spectrum_data:
        print("Missing custom variable or spectrum data.")
        return []

    mode = str(custom_variable_data.get("mode", "sequence")).strip().lower()
    variable_name = str(custom_variable_data.get("variable_name") or "Custom Variable").strip()
    entries = custom_variable_data.get("entries") or []

    if mode == "filename":
        return _combine_custom_by_filename(entries, spectrum_data, variable_name)
    if mode == "timestamp":
        return _combine_custom_by_timestamp(entries, spectrum_data, variable_name, time_buffer=time_buffer)
    return _combine_custom_by_sequence(entries, spectrum_data, variable_name)
