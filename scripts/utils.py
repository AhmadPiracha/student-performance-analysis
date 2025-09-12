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

def nearest_time_merge(left, right, on_id, left_time, right_time, tolerance_seconds=5, direction="nearest"):
    left2 = left.copy()
    right2 = right.copy()

    if left2.empty or right2.empty:
        return pd.DataFrame()

    # Sort safely
    left2 = left2.sort_values([on_id, left_time]).reset_index(drop=True)
    right2 = right2.sort_values([on_id, right_time]).reset_index(drop=True)

    # Drop null timestamps and IDs
    left2 = left2.dropna(subset=[on_id, left_time])
    right2 = right2.dropna(subset=[on_id, right_time])

    out = pd.merge_asof(
        left2, right2,
        by=on_id,
        left_on=left_time, right_on=right_time,
        direction=direction,
        tolerance=pd.Timedelta(seconds=tolerance_seconds)
    )
    return out


def paired_confidence_interval(a, b, alpha=0.05):
    """
    Compute 95% CI for paired differences (a - b).
    Returns (lower, upper).
    """
    diff = np.array(a) - np.array(b)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return (np.nan, np.nan)

    mean_diff = diff.mean()
    se = stats.sem(diff, nan_policy="omit")
    h = se * stats.t.ppf(1 - alpha/2, len(diff)-1)
    return (mean_diff - h, mean_diff + h)


def safe_numeric_fill(df, columns=None):
    """Ensure numeric columns are numeric and fill NaNs with 0."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df
