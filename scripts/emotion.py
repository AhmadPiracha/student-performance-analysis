import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from .utils import nearest_time_merge, compute_basic_kpis, paired_confidence_interval

# === Helper: emotion features ===
def compute_emotion_features(df, id_col, task_col, emo_cols, ts_col):
    """Compute per-student-task emotion deltas + means."""
    def per_group(g):
        g = g.sort_values(ts_col)
        out = {}
        for c in emo_cols:
            s = pd.to_numeric(g[c], errors="coerce")
            out[f"{c}_delta"] = s.diff().mean() if len(s) > 1 else np.nan
            out[f"{c}_mean"] = s.mean()
        return pd.Series(out)

    feats = df.groupby([id_col, task_col], dropna=False).apply(per_group).reset_index()
    return feats

# === Helper: paired effect size ===
def cohens_d_paired(a, b):
    diff = a - b
    return diff.mean() / (diff.std(ddof=1) + 1e-9)

def compare_with_baseline(base_df, with_emo_df, metrics):
    """Compare baseline vs current using condition column."""
    rows = []
    for m in metrics:
        a = base_df[m].dropna()
        b = with_emo_df[m].dropna()

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
        ci_low, ci_high = paired_confidence_interval(b, a)   # note: b - a

        rows.append({
            "metric": m,
            "mean_base": a.mean(),
            "mean_with_emotion": b.mean(),
            "delta": b.mean() - a.mean(),
            "t_p": tp,
            "wilcoxon_p": wp,
            "cohens_d": d,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "n": len(a)
        })
    df = pd.DataFrame(rows)
    cols_order = [
        "metric", "mean_base", "mean_with_emotion", "delta",
        "t_p", "wilcoxon_p", "cohens_d", "ci_lower", "ci_upper", "n"
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
    emo_cols = schema.get("emotion_cols", [])

    proc_dir = Path(cfg["paths"]["data_processed"])
    perf_path = proc_dir / "performance_clean.csv"
    emo_path = proc_dir / "emotion_clean.csv"

    if not perf_path.exists() or not emo_path.exists():
        return {}

    perf = pd.read_csv(perf_path, dtype=str, low_memory=False)
    emo = pd.read_csv(emo_path, dtype=str, low_memory=False)

    # --- Standardize IDs, tasks, timestamps ---
    for df in [perf, emo]:
        df[id_col] = df[id_col].astype(str)
        if task_col in df.columns:
            df[task_col] = df[task_col].astype(str)

    perf[ts_col] = pd.to_datetime(perf[ts_col], errors="coerce", utc=True)
    emo[ts_col] = pd.to_datetime(emo[ts_col], errors="coerce", utc=True)

    # Drop rows with missing timestamps
    perf = perf.dropna(subset=[ts_col])
    emo = emo.dropna(subset=[ts_col])

    # Ensure sorted
    perf = perf.sort_values([id_col, ts_col])
    emo = emo.sort_values([id_col, ts_col])

    if perf.empty or emo.empty:
        return {}

    # Convert numeric emotion columns
    for c in emo_cols:
        if c in emo.columns:
            emo[c] = pd.to_numeric(emo[c], errors="coerce").fillna(0)

    # Split baseline vs current
    baseline = emo[emo["condition"] == "baseline"]
    current = emo[emo["condition"] == "current"]

    # Link performance + emotion
    tol = cfg["merging"].get("time_tolerance_seconds", 5)
    linked = nearest_time_merge(perf, emo, on_id=id_col,
                                left_time=ts_col, right_time=ts_col,
                                tolerance_seconds=tol)

    # Features
    emo_feats = compute_emotion_features(linked, id_col, task_col, emo_cols, ts_col)

    # Baseline KPIs
    base_kpis = compute_basic_kpis(perf, id_col, task_col, acc_col, rt_col)

    # Merge features into KPIs
    merged = base_kpis.merge(emo_feats, on=[id_col, task_col], how="left")

    # Compare baseline vs current using condition
    metrics = ["avg_accuracy", "avg_reaction_time", "error_rate"]
    cmp = compare_with_baseline(
        base_kpis.groupby(id_col).mean(numeric_only=True),
        merged.groupby(id_col).mean(numeric_only=True),
        metrics
    )

    # Correlation between KPIs and emotion
    corr = merged.drop(columns=[id_col, task_col]).corr(numeric_only=True)

    # Save outputs
    emo_feats.to_csv(proc_dir / "emotion_features_by_task.csv", index=False)
    merged.to_csv(proc_dir / "emotion_perf_compare.csv", index=False)
    cmp.to_csv(proc_dir / "comparison_emotion_vs_baseline.csv", index=False)
    corr.to_csv(proc_dir / "emotion_perf_corr.csv", index=True)
    linked.to_csv(proc_dir / "emotion_linked.csv", index=False)

    return {
        "features": str(proc_dir / "emotion_features_by_task.csv"),
        "linked": str(proc_dir / "emotion_linked.csv"),
        "compare": str(proc_dir / "emotion_perf_compare.csv"),
        "baseline_vs_emotion": str(proc_dir / "comparison_emotion_vs_baseline.csv"),
        "corr": str(proc_dir / "emotion_perf_corr.csv"),
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
