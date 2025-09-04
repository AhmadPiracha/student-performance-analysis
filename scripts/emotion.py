import os
import yaml
import numpy as np
import pandas as pd
from .utils import nearest_time_merge, compute_basic_kpis

def run(config_path):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    schema = cfg["schema"]
    id_col = schema["id"]
    ts_col = schema["timestamp"]
    task_col = schema["task"]
    acc_col = schema["accuracy"]
    rt_col = schema["reaction_time"]
    emo_cols = schema["emotion_cols"]

    # Paths
    perf_path = os.path.join(cfg["paths"]["data_processed"], "performance_clean.csv")
    emo_path = os.path.join(cfg["paths"]["data_processed"], "emotion_clean.csv")

    # Read CSVs safely
    perf = pd.read_csv(perf_path)
    emo = pd.read_csv(emo_path)

    # Convert timestamps
    perf[ts_col] = pd.to_datetime(perf[ts_col], format="%d/%m/%Y %H:%M", errors="coerce", utc=True)
    emo[ts_col] = pd.to_datetime(emo[ts_col], format="%d/%m/%Y %H:%M", errors="coerce", utc=True)

    if perf.empty or emo.empty:
        # write empty outputs
        out_dir = cfg["paths"]["data_processed"]
        pd.DataFrame().to_csv(os.path.join(out_dir, "emotion_linked.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, "emotion_perf_compare.csv"), index=False)
        return {
            "emotion_linked": "data/processed/emotion_linked.csv",
            "emotion_perf_compare": "data/processed/emotion_perf_compare.csv"
        }

    # Link emotion to performance by nearest timestamp per student
    tol = cfg["merging"]["time_tolerance_seconds"]
    linked = nearest_time_merge(perf, emo, on_id=id_col, left_time=ts_col, right_time=ts_col, tolerance_seconds=tol)

    # Compute simple emotion-change metrics per task
    def per_student_task_changes(df):
        df = df.sort_values(ts_col)
        out = {}
        for c in emo_cols:
            s = df[c].astype(float)
            out[f"{c}_delta"] = s.diff().mean() if len(s) > 1 else np.nan
            out[f"{c}_mean"] = s.mean()
        return pd.Series(out)

    # Apply groupby safely
    changes = linked.groupby([id_col, task_col], dropna=False, group_keys=False)\
                    .apply(per_student_task_changes)\
                    .reset_index(drop=True)

    # Compare performance with vs without emotion features
    kpis = compute_basic_kpis(perf, id_col, task_col, acc_col, rt_col)
    emo_means = linked.groupby([id_col, task_col], dropna=False)[emo_cols].mean().reset_index()
    compare = kpis.merge(emo_means, on=[id_col, task_col], how="left")

    # Quick correlation matrix
    corr = compare.drop(columns=[id_col, task_col]).corr(numeric_only=True)
    corr_path = os.path.join(cfg["paths"]["data_processed"], "emotion_perf_corr.csv")
    corr.to_csv(corr_path, index=False)

    # Save outputs
    out_dir = cfg["paths"]["data_processed"]
    linked.to_csv(os.path.join(out_dir, "emotion_linked.csv"), index=False)
    changes.to_csv(os.path.join(out_dir, "emotion_changes_by_task.csv"), index=False)
    compare.to_csv(os.path.join(out_dir, "emotion_perf_compare.csv"), index=False)

    return {
        "emotion_linked": "data/processed/emotion_linked.csv",
        "emotion_changes_by_task": "data/processed/emotion_changes_by_task.csv",
        "emotion_perf_compare": "data/processed/emotion_perf_compare.csv",
        "emotion_perf_corr": "data/processed/emotion_perf_corr.csv"
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
