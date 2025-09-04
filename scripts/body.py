
import os, yaml, numpy as np, pandas as pd
from .utils import nearest_time_merge

def run(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    schema = cfg["schema"]
    id_col = schema["id"]
    ts_col = schema["timestamp"]
    task_col = schema["task"]
    acc_col = schema["accuracy"]
    rt_col = schema["reaction_time"]
    body_cols = schema["body_cols"]

    perf_path = os.path.join(cfg["paths"]["data_processed"], "performance_clean.csv")
    body_path = os.path.join(cfg["paths"]["data_processed"], "body_clean.csv")
    perf = pd.read_csv(perf_path, parse_dates=[ts_col]) if os.path.exists(perf_path) else pd.DataFrame()
    body = pd.read_csv(body_path, parse_dates=[ts_col]) if os.path.exists(body_path) else pd.DataFrame()

    if perf.empty or body.empty:
        out_dir = cfg["paths"]["data_processed"]
        pd.DataFrame().to_csv(os.path.join(out_dir, "body_linked.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, "body_perf_compare.csv"), index=False)
        return {"body_linked": "data/processed/body_linked.csv",
                "body_perf_compare": "data/processed/body_perf_compare.csv"}

    tol = cfg["merging"]["time_tolerance_seconds"]
    linked = nearest_time_merge(perf, body, on_id=id_col, left_time=ts_col, right_time=ts_col, tolerance_seconds=tol)

    # Aggregate body metrics per student-task
    agg = linked.groupby([id_col, task_col], dropna=False)[body_cols].agg(["mean", "std"]).reset_index()
    # Flatten columns
    agg.columns = ['_'.join([c for c in col if c]) if isinstance(col, tuple) else col for col in agg.columns]

    # Compare performance with vs without body features (simple correlations)
    from .utils import compute_basic_kpis
    kpis = compute_basic_kpis(perf, id_col, task_col, acc_col, rt_col)
    body_means = linked.groupby([id_col, task_col], dropna=False)[body_cols].mean().reset_index()
    compare = kpis.merge(body_means, on=[id_col, task_col], how="left")
    corr = compare.drop(columns=[id_col, task_col]).corr(numeric_only=True)
    corr_path = os.path.join(cfg["paths"]["data_processed"], "body_perf_corr.csv")
    corr.to_csv(corr_path)

    out_dir = cfg["paths"]["data_processed"]
    linked.to_csv(os.path.join(out_dir, "body_linked.csv"), index=False)
    agg.to_csv(os.path.join(out_dir, "body_summary_by_task.csv"), index=False)
    compare.to_csv(os.path.join(out_dir, "body_perf_compare.csv"), index=False)

    return {"body_linked": "data/processed/body_linked.csv",
            "body_summary_by_task": "data/processed/body_summary_by_task.csv",
            "body_perf_compare": "data/processed/body_perf_compare.csv",
            "body_perf_corr": "data/processed/body_perf_corr.csv"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
