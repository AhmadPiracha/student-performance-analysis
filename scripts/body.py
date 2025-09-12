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
    return diff.mean() / (diff.std(ddof=1) + 1e-9)

def compare_with_baseline(base_df, with_body_df, metrics):
    rows = []
    for m in metrics:
        a = base_df[m].dropna()
        b = with_body_df[m].dropna()

        if len(a) == 0 or len(b) == 0 or len(a) != len(b):
            continue

        try:
            tstat, tp = stats.ttest_rel(a, b, nan_policy="omit")
        except Exception:
            tp = np.nan
        try:
            wstat, wp = stats.wilcoxon(a, b)
        except Exception:
            wp = np.nan
        d = cohens_d_paired(a, b)
        ci_low, ci_high = paired_confidence_interval(b, a)

        rows.append({
            "metric": m,
            "n": len(a),
            "t_p": float(tp) if not np.isnan(tp) else None,
            "wilcoxon_p": float(wp) if not np.isnan(wp) else None,
            "cohens_d": float(d) if not np.isnan(d) else None,
            "ci_lower": float(ci_low) if not np.isnan(ci_low) else None,
            "ci_upper": float(ci_high) if not np.isnan(ci_high) else None,
        })

    df = pd.DataFrame(rows)
    cols_order = [
        "metric", "n", "t_p", "wilcoxon_p", "cohens_d", "ci_lower", "ci_upper"
    ]

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
    body[ts_col] = pd.to_datetime(body[ts_col], errors="coerce", utc=True)

    if perf.empty or body.empty:
        return {}

    # Convert numeric body columns
    for c in body_cols:
        if c in body.columns:
            body[c] = pd.to_numeric(body[c], errors="coerce").fillna(0)

    # Split baseline vs current
    baseline = body[body["condition"] == "baseline"]
    current = body[body["condition"] == "current"]

    # Link performance + body
    tol = cfg["merging"].get("time_tolerance_seconds", 5)
    linked = nearest_time_merge(perf, body, on_id=id_col,
                                left_time=ts_col, right_time=ts_col,
                                tolerance_seconds=tol)

    # Features
    body_feats = compute_body_features(linked, id_col, task_col, body_cols)

    # Baseline KPIs
    base_kpis = compute_basic_kpis(perf, id_col, task_col, acc_col, rt_col)

    # Merge features into KPIs
    merged = base_kpis.merge(body_feats, on=[id_col, task_col], how="left")

    # Compare baseline vs current using condition
    metrics = ["avg_accuracy", "avg_reaction_time", "error_rate"]
    cmp = compare_with_baseline(
        base_kpis.groupby(id_col).mean(numeric_only=True),
        merged.groupby(id_col).mean(numeric_only=True),
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
