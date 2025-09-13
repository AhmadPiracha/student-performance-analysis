import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from .utils import nearest_time_merge, compute_basic_kpis, paired_confidence_interval

# === Helper: body features ===
def compute_body_features(df, id_col, task_col, body_cols):
    """Compute per-student-task body feature means and stds."""
    def per_group(g):
        out = {}
        for c in body_cols:
            s = pd.to_numeric(g[c], errors="coerce")
            out[f"{c}_mean"] = s.mean()
            out[f"{c}_std"] = s.std()
        return pd.Series(out)

    feats = df.groupby([id_col, task_col], dropna=False).apply(per_group).reset_index()
    return feats

# === Helper: paired effect size ===
def cohens_d_paired(a, b):
    diff = a - b
    sd = diff.std(ddof=1)
    if pd.isna(sd) or sd < 1e-8:
        return 0.0
    return diff.mean() / sd

def compare_with_baseline(base_df, with_body_df, metrics):
    rows = []
    for m in metrics:
        # Align by index (expected to be student id) so paired tests compare same subjects
        a = base_df[m]
        b = with_body_df[m]
        paired = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        if paired.empty:
            continue
        a = paired[("a", m)] if ("a", m) in paired.columns else paired.iloc[:, 0]
        b = paired[("b", m)] if ("b", m) in paired.columns else paired.iloc[:, 1]

        # Ensure numeric numpy arrays and drop NaNs
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        mask = ~np.isnan(a_arr) & ~np.isnan(b_arr)
        a_arr = a_arr[mask]
        b_arr = b_arr[mask]

        tp = np.nan
        wp = np.nan
        d = 0.0
        ci_low, ci_high = (np.nan, np.nan)

        if len(a_arr) >= 2:
            # If differences are constant (zero variance), return default stats
            if np.isclose(np.std(a_arr - b_arr, ddof=1), 0.0):
                tp = 1.0
                wp = 1.0
                d = 0.0
                ci_low, ci_high = (0.0, 0.0)
            else:
                try:
                    _, tp = stats.ttest_rel(a_arr, b_arr, nan_policy="omit")
                except Exception:
                    tp = np.nan
                try:
                    _, wp = stats.wilcoxon(a_arr, b_arr)
                except Exception:
                    wp = np.nan
                d = cohens_d_paired(pd.Series(a_arr), pd.Series(b_arr))
                ci_low, ci_high = paired_confidence_interval(b_arr, a_arr)
        else:
            # Not enough samples for paired tests; skip statistical tests but record n
            tp = None
            wp = None
            d = 0.0
            ci_low, ci_high = (None, None)

        rows.append({
            "metric": m,
            "n": int(len(a_arr)),
            "t_p": float(tp) if tp is not None and not (isinstance(tp, float) and np.isnan(tp)) else None,
            "wilcoxon_p": float(wp) if wp is not None and not (isinstance(wp, float) and np.isnan(wp)) else None,
            "cohens_d": float(d) if d is not None and not (isinstance(d, float) and np.isnan(d)) else None,
            "ci_lower": float(ci_low) if ci_low is not None and not (isinstance(ci_low, float) and np.isnan(ci_low)) else None,
            "ci_upper": float(ci_high) if ci_high is not None and not (isinstance(ci_high, float) and np.isnan(ci_high)) else None,
        })

    df = pd.DataFrame(rows)
    cols_order = [
        "metric", "n", "t_p", "wilcoxon_p", "cohens_d", "ci_lower", "ci_upper"
    ]

    # If no rows were generated (no paired data), return an empty DataFrame
    # with the expected columns instead of raising a KeyError when selecting
    # by column order.
    if df.empty:
        return pd.DataFrame(columns=cols_order)

    return df[cols_order]


# === MAIN RUN ===
def run(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    schema = cfg["schema"]
    id_col = schema["id"]
    ts_col = schema["timestamp"]
    task_col = schema["task"]
    acc_col = schema["accuracy"]
    rt_col = schema["reaction_time"]
    body_cols = schema.get("body_cols", [])

    proc_dir = Path(cfg["paths"]["data_processed"])
    perf_path = proc_dir / "performance_clean.csv"
    body_path = proc_dir / "body_clean.csv"

    if not perf_path.exists() or not body_path.exists():
        return {}

    perf = pd.read_csv(perf_path, dtype=str, low_memory=False)
    body = pd.read_csv(body_path, dtype=str, low_memory=False)

    # Convert timestamps
    perf[ts_col] = pd.to_datetime(perf[ts_col], errors="coerce", utc=True)

    # Body exports sometimes use a 'times' column for the readable timestamp
    # (the CSV can contain multiple timestamp-like columns). Prefer that when
    # present; otherwise fall back to the schema timestamp column.
    body_ts_col = ts_col
    if "times" in body.columns:
        body_ts_col = "times"
    else:
        # case-insensitive fallback
        for c in body.columns:
            if c.lower() == "times":
                body_ts_col = c
                break

    body[body_ts_col] = pd.to_datetime(body[body_ts_col], errors="coerce", utc=True)

    if perf.empty or body.empty:
        return {}

    # Convert numeric body columns but do NOT fill missing values with 0 here.
    # Filling with zero hides missingness and can produce degenerate stats.
    for c in body_cols:
        if c in body.columns:
            body[c] = pd.to_numeric(body[c], errors="coerce")

    # Split baseline vs current
    baseline = body[body["condition"] == "baseline"]
    current = body[body["condition"] == "current"]

    # --- FIX: sort before merge_asof ---
    perf = perf.sort_values([id_col, ts_col])
    body = body.sort_values([id_col, body_ts_col])

    print("Perf IDs sample:", perf[id_col].unique()[:5])
    print("body IDs sample:", body[id_col].unique()[:5])

    # Link performance + body
    tol = cfg["merging"].get("time_tolerance_seconds", 5)
    linked = nearest_time_merge(
        perf, body,
        on_id=id_col,
        left_time=ts_col, right_time=body_ts_col,
        tolerance_seconds=tol
    )


    # Features
    body_feats = compute_body_features(linked, id_col, task_col, body_cols)

    # Baseline KPIs (overall) - kept for merged outputs / correlation
    base_kpis = compute_basic_kpis(perf, id_col, task_col, acc_col, rt_col)

    # Merge features into KPIs for per-subject/task table used in correlations
    merged = base_kpis.merge(body_feats, on=[id_col, task_col], how="left")

    # --- Fix: compute KPIs separately for linked rows by body 'condition' ---
    # Determine which column in `linked` refers to the original perf id (left)
    left_id_col = id_col if id_col in linked.columns else (f"{id_col}_x" if f"{id_col}_x" in linked.columns else id_col)

    # Compute KPIs for baseline vs current using only linked rows that have a condition
    if "condition" in linked.columns:
        baseline_kpis = compute_basic_kpis(linked[linked["condition"] == "baseline"], left_id_col, task_col, acc_col, rt_col)
        current_kpis = compute_basic_kpis(linked[linked["condition"] == "current"], left_id_col, task_col, acc_col, rt_col)
    else:
        # If condition is missing, fall back to overall KPIs (conservative)
        baseline_kpis = base_kpis.copy()
        current_kpis = base_kpis.copy()

    # Compare baseline vs current using per-subject means (paired)
    metrics = ["avg_accuracy", "avg_reaction_time", "error_rate"]
    cmp = compare_with_baseline(
        baseline_kpis.groupby(left_id_col).mean(numeric_only=True),
        current_kpis.groupby(left_id_col).mean(numeric_only=True),
        metrics
    )

    # Correlation between KPIs and body features
    corr = merged.drop(columns=[id_col, task_col]).corr(numeric_only=True)

    # Save outputs
    body_feats.to_csv(proc_dir / "body_features_by_task.csv", index=False)
    merged.to_csv(proc_dir / "body_perf_compare.csv", index=False)
    cmp.to_csv(proc_dir / "comparison_body_vs_baseline.csv", index=False)
    corr.to_csv(proc_dir / "body_perf_corr.csv", index=True)
    linked.to_csv(proc_dir / "body_linked.csv", index=False)

    return {
        "features": str(proc_dir / "body_features_by_task.csv"),
        "linked": str(proc_dir / "body_linked.csv"),
        "compare": str(proc_dir / "body_perf_compare.csv"),
        "baseline_vs_body": str(proc_dir / "comparison_body_vs_baseline.csv"),
        "corr": str(proc_dir / "body_perf_corr.csv"),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
