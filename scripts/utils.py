import yaml
import re
import pandas as pd
import numpy as np
from scipy import stats

def standardize_id(val):
    """Clean and standardize user IDs; return 'UNKNOWN' if missing."""
    if pd.isna(val):
        return "UNKNOWN"
    s = str(val).strip().upper()
    s = re.sub(r"[^A-Z0-9_-]+", "", s)
    return s


def normalize_accuracy(series):
    """Normalize accuracy to 0-1 range, fill NaNs with 0."""
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if s.max() > 1.5:
        s = s / 100.0
    return s.clip(0.0, 1.0)


def compute_basic_kpis(df, id_col, task_col, acc_col, rt_col):
    """Compute average accuracy, reaction time, and error rate per student-task."""
    tmp = df.copy()
    tmp[acc_col] = normalize_accuracy(tmp[acc_col])
    tmp["error_rate"] = 1.0 - tmp[acc_col]
    tmp[rt_col] = pd.to_numeric(tmp[rt_col], errors="coerce")
    aggs = {acc_col: "mean", rt_col: "mean", "error_rate": "mean"}
    out = tmp.groupby([id_col, task_col], dropna=False).agg(aggs).reset_index()
    return out.rename(columns={acc_col: "avg_accuracy", rt_col: "avg_reaction_time"})

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_excel_sheets(file_path, sheets):
    """
    Load multiple sheets from an Excel file into a dict of DataFrames.
    Handles missing sheets gracefully.
    """
    sheet_names = list(sheets.values())
    raw = pd.read_excel(file_path, sheet_name=sheet_names)
    result = {}
    for key, sheet_name in sheets.items():
        df = raw.get(sheet_name)
        if df is None:
            df = pd.DataFrame()
        else:
            df.columns = df.columns.str.strip()
        result[key] = df
    return result

def to_datetime_safe(x, fmt="%d/%m/%Y %H:%M"):
    """Convert to datetime safely; returns NaT on failure."""
    try:
        return pd.to_datetime(x, format=fmt, errors="coerce", utc=True)
    except Exception:
        return pd.NaT

def nearest_time_merge(left, right, on_id, left_time, right_time, tolerance_seconds, direction="nearest"):
    """
    Robust nearest-time merge implemented per-id to avoid pandas' global
    'left keys must be sorted' requirement.

    Strategy:
      - copy inputs and create standardized temporary id/time columns
      - parse timestamps to UTC-naive
      - drop rows missing id/time
      - for each id in left, sort that id's rows and the matching right rows,
        then run pd.merge_asof on the pair and collect results
      - concat results and return with original columns preserved
    """
    left = left.copy()
    right = right.copy()

    # Debug helper: print incoming column names
    try:
        print("nearest_time_merge called; on_id=", on_id)
        print("left columns:", list(left.columns))
        print("right columns:", list(right.columns))
    except Exception:
        pass

    tmp_id = "__merge_id__"
    tmp_lt = "__merge_time_l__"
    tmp_rt = "__merge_time_r__"

    # Allow case-insensitive id column matching; fall back to provided on_id
    def _find_col(df, candidate):
        if candidate in df.columns:
            return candidate
        # try case-insensitive
        for c in df.columns:
            if c.lower() == candidate.lower():
                return c
        return None

    # locate actual id columns in each frame
    left_id_col = _find_col(left, on_id)
    right_id_col = _find_col(right, on_id)
    if left_id_col is None or right_id_col is None:
        # give full context and raise
        raise KeyError(
            "nearest_time_merge: id column '{}' not found in left or right DataFrame. left cols: {} right cols: {}".format(
                on_id, list(left.columns), list(right.columns)
            )
        )

    # Standardize IDs into temp column (preserve original on_id)
    try:
        left[tmp_id] = left[left_id_col].astype(str).str.upper().str.strip()
        right[tmp_id] = right[right_id_col].astype(str).str.upper().str.strip()
    except Exception as e:
        raise RuntimeError(f"nearest_time_merge: failed to create standardized id column from '{left_id_col}'/'{right_id_col}': {e}\nLeft cols: {list(left.columns)}\nRight cols: {list(right.columns)}")

    # Create temporary timestamp columns (parse to UTC then drop tz)
    left[tmp_lt] = pd.to_datetime(left[left_time], errors="coerce", utc=True)
    right[tmp_rt] = pd.to_datetime(right[right_time], errors="coerce", utc=True)
    # Robustly remove/normalize tz info (works for tz-aware and tz-naive series)
    try:
        left[tmp_lt] = left[tmp_lt].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        try:
            left[tmp_lt] = left[tmp_lt].dt.tz_localize(None)
        except Exception:
            # leave as-is (may already be naive or all NaT)
            pass
    try:
        right[tmp_rt] = right[tmp_rt].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        try:
            right[tmp_rt] = right[tmp_rt].dt.tz_localize(None)
        except Exception:
            pass

    # Drop rows missing id or timestamps
    left = left.dropna(subset=[tmp_id, tmp_lt])
    right = right.dropna(subset=[tmp_id, tmp_rt])

    # Prepare container for per-id merge results
    chunks = []
    tol = pd.Timedelta(seconds=tolerance_seconds)

    # Iterate left ids (smaller cardinality expected) and merge per-group
    left_ids = left[tmp_id].unique()
    for uid in left_ids:
        lgrp = left[left[tmp_id] == uid].sort_values(tmp_lt).reset_index(drop=True)
        rgrp = right[right[tmp_id] == uid].sort_values(tmp_rt).reset_index(drop=True)
        if rgrp.empty:
            # no right rows for this id -> keep left rows with NaNs for right cols
            # preserve original columns by concatenating lgrp with NaN columns from right
            # easiest: perform merge_asof with empty right (will produce lgrp with NaNs)
            out_grp = pd.merge_asof(
                lgrp, rgrp,
                left_on=tmp_lt, right_on=tmp_rt,
                by=tmp_id,
                tolerance=tol,
                direction=direction
            )
        else:
            out_grp = pd.merge_asof(
                lgrp, rgrp,
                left_on=tmp_lt, right_on=tmp_rt,
                by=tmp_id,
                tolerance=tol,
                direction=direction
            )
        chunks.append(out_grp)

    if not chunks:
        # nothing to merge -> return empty frame with left's columns
        empty_out = left.iloc[0:0].copy()
        # ensure requested id column exists on empty frame
        empty_out[on_id] = pd.Series(dtype=object)
        for c in (tmp_id, tmp_lt, tmp_rt):
            if c in empty_out.columns:
                empty_out = empty_out.drop(columns=[c])
        return empty_out

    out = pd.concat(chunks, axis=0, ignore_index=True, sort=False)

    # Ensure original id column name exists (populate from tmp_id)
    try:
        out[on_id] = out[tmp_id]
    except Exception:
        # fallback: if tmp_id missing, try to copy from one of the original id cols
        if left_id_col in out.columns:
            out[on_id] = out[left_id_col]
        elif right_id_col in out.columns:
            out[on_id] = out[right_id_col]

    # Drop temporary helper columns
    for c in (tmp_id, tmp_lt, tmp_rt):
        if c in out.columns:
            out = out.drop(columns=[c])

    return out

def paired_confidence_interval(a, b, alpha=0.05):
    """
    Compute 95% CI for paired differences (a - b).
    Returns (lower, upper).
    """
    diff = np.array(a) - np.array(b)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 2:
        return (np.nan, np.nan)

    mean_diff = diff.mean()
    # compute standard error manually to avoid scipy nan_policy differences
    se = np.std(diff, ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(1 - alpha/2, n - 1)
    return (mean_diff - h, mean_diff + h)


def safe_numeric_fill(df, columns=None):
    """Ensure numeric columns are numeric and fill NaNs with 0."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df
