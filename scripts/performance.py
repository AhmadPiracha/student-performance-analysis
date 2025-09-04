
import os, yaml, pandas as pd
from .utils import compute_basic_kpis

def run(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    schema = cfg["schema"]
    id_col = schema["id"]
    ts_col = schema["timestamp"]
    task_col = schema["task"]
    acc_col = schema["accuracy"]
    rt_col = schema["reaction_time"]

    perf_path = os.path.join(cfg["paths"]["data_processed"], "performance_clean.csv")
    if not os.path.exists(perf_path):
        raise FileNotFoundError("performance_clean.csv not found; run preprocess first.")

    perf = pd.read_csv(perf_path, parse_dates=[ts_col])
    kpis = compute_basic_kpis(perf, id_col, task_col, acc_col, rt_col)

    # Domain rollups
    task_domains = cfg["schema"]["task_domains"]
    def map_domain(task):
        for dom, tasks in task_domains.items():
            if task in tasks:
                return dom
        return "other"
    kpis["domain"] = kpis[task_col].apply(map_domain)

    # Domain-level per-student
    dom_summary = kpis.groupby([id_col, "domain"], dropna=False).agg({
        "avg_accuracy": "mean",
        "avg_reaction_time": "mean",
        "error_rate": "mean"
    }).reset_index()

    out_dir = cfg["paths"]["data_processed"]
    kpis.to_csv(os.path.join(out_dir, "performance_kpis.csv"), index=False)
    dom_summary.to_csv(os.path.join(out_dir, "performance_domain_summary.csv"), index=False)

    return {"kpis": "data/processed/performance_kpis.csv",
            "domain_summary": "data/processed/performance_domain_summary.csv"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
