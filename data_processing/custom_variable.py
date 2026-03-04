import re
import numpy as np
import pandas as pd


VALUE_HINTS = (
    "fluence", "fluens", "dose", "dozis", "dózis",
    "value", "variable", "x", "axis", "temperature", "temp"
)
FILENAME_HINTS = (
    "filename", "file", "filen", "name", "fajl", "fájl", "nev", "név"
)
TIMESTAMP_HINTS = ("timestamp", "time", "unix", "epoch")
HEADER_HINTS = VALUE_HINTS + FILENAME_HINTS + TIMESTAMP_HINTS


def _normalize_col_name(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _try_parse_markdown_pipe_table(file_path):
    for encoding in ("utf-8", "cp1250", "latin1"):
        try:
            with open(file_path, "r", encoding=encoding) as fh:
                lines = [line.strip() for line in fh if line.strip()]
        except Exception:  # noqa: BLE001
            continue

        rows = []
        for line in lines:
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            # Skip markdown separator rows like: |---|:---:|---:|
            if all(re.fullmatch(r":?-+:?", c or "") for c in cells):
                continue
            rows.append(cells)

        if len(rows) >= 2:
            max_len = max(len(r) for r in rows)
            padded = [r + [""] * (max_len - len(r)) for r in rows]
            return pd.DataFrame(padded)

    return None


def _find_header_row(df: pd.DataFrame) -> int | None:
    max_rows = min(len(df), 25)
    for idx in range(max_rows):
        row_vals = [str(v).strip() for v in df.iloc[idx].tolist() if str(v).strip() and str(v).strip().lower() != "nan"]
        if len(row_vals) < 2:
            continue
        row_text = " ".join(v.lower() for v in row_vals)
        if any(h in row_text for h in HEADER_HINTS):
            return idx
    return None


def _read_table(file_path):
    markdown_table_df = _try_parse_markdown_pipe_table(file_path)
    if markdown_table_df is not None:
        return markdown_table_df

    read_errors = []
    for encoding in ("utf-8", "cp1250", "latin1"):
        for kwargs in (
            {"sep": None, "engine": "python", "header": None, "dtype": str, "on_bad_lines": "skip", "encoding": encoding},
            {"sep": "\t", "engine": "python", "header": None, "dtype": str, "on_bad_lines": "skip", "encoding": encoding},
            {"sep": ";", "engine": "python", "header": None, "dtype": str, "on_bad_lines": "skip", "encoding": encoding},
            {"sep": ",", "engine": "python", "header": None, "dtype": str, "on_bad_lines": "skip", "encoding": encoding},
        ):
            try:
                return pd.read_csv(file_path, **kwargs)
            except Exception as exc:  # noqa: BLE001
                read_errors.append(str(exc))
    raise ValueError(f"Unable to read custom variable file. Errors: {' | '.join(read_errors[:3])}")


def _prepare_dataframe(file_path) -> pd.DataFrame:
    df_raw = _read_table(file_path)
    if df_raw.empty:
        return df_raw

    df_raw = df_raw.fillna("").astype(str)
    df_raw = df_raw.apply(lambda c: c.str.strip())
    df_raw = df_raw.loc[(df_raw != "").any(axis=1)]
    df_raw = df_raw.loc[:, (df_raw != "").any(axis=0)]

    if df_raw.empty:
        return df_raw

    header_idx = _find_header_row(df_raw)
    if header_idx is None:
        df = df_raw.reset_index(drop=True).copy()
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
        return df

    header_vals = df_raw.iloc[header_idx].tolist()
    seen = {}
    columns = []
    for i, val in enumerate(header_vals):
        name = str(val).strip() or f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        columns.append(name)

    df = df_raw.iloc[header_idx + 1:].reset_index(drop=True).copy()
    df.columns = columns
    df = df.loc[(df != "").any(axis=1)]
    return df


def _choose_value_column(df: pd.DataFrame):
    numeric = {}
    for col in df.columns:
        numeric[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False), errors="coerce"
        )

    # Prefer columns with value-like header names.
    for col in df.columns:
        col_norm = _normalize_col_name(col)
        if any(h in col_norm for h in VALUE_HINTS) and numeric[col].notna().sum() > 0:
            return col, numeric[col]

    # Fallback to the most populated numeric column.
    best_col = None
    best_count = -1
    for col in df.columns:
        count = int(numeric[col].notna().sum())
        if count > best_count:
            best_count = count
            best_col = col

    if best_col is None or best_count <= 0:
        raise ValueError("No numeric column found for custom variable values.")

    return best_col, numeric[best_col]


def _choose_key_columns(df: pd.DataFrame, value_col: str):
    filename_col = None
    timestamp_col = None

    for col in df.columns:
        if col == value_col:
            continue
        col_norm = _normalize_col_name(col)
        if filename_col is None and any(h in col_norm for h in FILENAME_HINTS):
            filename_col = col
        if timestamp_col is None and any(h in col_norm for h in TIMESTAMP_HINTS):
            timestamp_col = col

    if filename_col is None:
        for col in df.columns:
            if col == value_col:
                continue
            non_empty = int((df[col] != "").sum())
            if non_empty == 0:
                continue
            as_num = pd.to_numeric(
                df[col].str.replace(",", ".", regex=False), errors="coerce"
            )
            num_count = int(as_num.notna().sum())
            if num_count < non_empty:
                filename_col = col
                break

    return filename_col, timestamp_col


def _prettify_variable_name(raw_name: str) -> str:
    name = str(raw_name or "").strip()
    if not name:
        return "Custom Variable"
    lowered = name.lower().replace("_", " ").strip()
    return lowered.title()


def process_custom_variable_data(file_path, progress_callback=None):
    """
    Parse a custom variable file for correlation with spectra.

    Returns:
      dict with keys:
        - mode: "filename" | "timestamp" | "sequence"
        - variable_name: column name used as value source
        - entries: parsed rows
    """
    if progress_callback:
        progress_callback("5%")

    df = _prepare_dataframe(file_path)
    if df.empty:
        if progress_callback:
            progress_callback("100%")
        return {"mode": "sequence", "variable_name": "Custom Variable", "entries": []}

    value_col, value_series = _choose_value_column(df)
    filename_col, timestamp_col = _choose_key_columns(df, value_col)

    raw_name = str(value_col).strip()
    if raw_name.lower().startswith("col_"):
        variable_name = "Custom Variable"
    else:
        variable_name = _prettify_variable_name(raw_name)

    entries = []
    mode = "sequence"

    if filename_col is not None:
        mode = "filename"
        for i in range(len(df)):
            value = value_series.iloc[i]
            if not np.isfinite(value):
                continue
            name = str(df[filename_col].iloc[i]).strip()
            if not name or name.lower() == "nan":
                continue
            entries.append({
                "filename": name,
                "filenames": [name],
                "value": float(value)
            })
    elif timestamp_col is not None:
        mode = "timestamp"
        ts_series = pd.to_numeric(
            df[timestamp_col].str.replace(",", ".", regex=False), errors="coerce"
        )
        for i in range(len(df)):
            ts = ts_series.iloc[i]
            value = value_series.iloc[i]
            if not (np.isfinite(ts) and np.isfinite(value)):
                continue
            entries.append({"timestamp": float(ts), "value": float(value)})
    else:
        mode = "sequence"
        for value in value_series:
            if np.isfinite(value):
                entries.append({"value": float(value)})

    if progress_callback:
        progress_callback("100%")

    return {
        "mode": mode,
        "variable_name": variable_name,
        "entries": entries,
    }
