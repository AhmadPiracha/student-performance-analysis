import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from openpyxl import Workbook
from .utils import compute_basic_kpis


# === Helper: Outlier filtering for RT ===
def iqr_filter_rt(df, rt_col, group_cols):
    """
    Removes RT outliers per student-task using IQR.
    - Removes RT < 50 ms (too fast to be real)
    - Removes values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    """
    df = df.copy()
    df = df[df[rt_col] > 50]

    def filter_group(g):
        if g[rt_col].notna().sum() < 3:
            return g
        q1 = g[rt_col].quantile(0.25)
        q3 = g[rt_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return g[(g[rt_col] >= lower) & (g[rt_col] <= upper)]

    return df.groupby(group_cols, group_keys=False).apply(filter_group).reset_index(drop=True)


# === Helper: Save explained KPIs to Excel ===
def save_kpis_explained(kpis_df, out_path):
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "games_summary"

    # Write KPI data
    ws1.append(list(kpis_df.columns))
    for _, row in kpis_df.iterrows():
        ws1.append(row.tolist())

    # Add definitions sheet
    ws2 = wb.create_sheet("definitions")
    ws2.append(["metric", "definition", "outlier_method", "notes"])
    ws2.append([
        "avg_accuracy",
        "Mean accuracy per task (normalized 0–1).",
        "N/A",
        "If original accuracy was % it was normalized."
    ])
    ws2.append([
        "avg_reaction_time",
        "Mean reaction time (ms).",
        "Removed RT < 50 ms, IQR filtering per student-task.",
        "Ensures unrealistic RTs are excluded."
    ])
    ws2.append([
        "error_rate",
        "1 – avg_accuracy.",
        "N/A",
        "Error rate complements accuracy."
    ])
    

    wb.save(out_path)


# === MAIN RUN ===
def run(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    schema = cfg["schema"]
    id_col = schema["id"]
    task_col = schema["task"]
    acc_col = schema["accuracy"]
    rt_col = schema["reaction_time"]

    proc_dir = Path(cfg["paths"]["data_processed"])
    perf_path = proc_dir / "performance_clean.csv"
    if not perf_path.exists():
        raise FileNotFoundError("performance_clean.csv not found; run preprocess first.")

    # Load and filter outliers
    perf = pd.read_csv(perf_path)
    perf = iqr_filter_rt(perf, rt_col, [id_col, task_col])

    # Compute KPIs
    kpis = compute_basic_kpis(perf, id_col, task_col, acc_col, rt_col)

    # Domain mapping
    task_domains = cfg["schema"]["task_domains"]

    def map_domain(task):
        for dom, tasks in task_domains.items():
            if task in tasks:
                return dom
        return "other"

    kpis["domain"] = kpis[task_col].apply(map_domain)

    # Domain-level rollups
    dom_summary = kpis.groupby([id_col, "domain"], dropna=False).agg({
        "avg_accuracy": "mean",
        "avg_reaction_time": "mean",
        "error_rate": "mean"
    }).reset_index()

    # Save outputs
    out_dir = proc_dir
    kpis_file = out_dir / "performance_kpis.csv"
    dom_file = out_dir / "performance_domain_summary.csv"
    explained_file = out_dir / "games_summary_explained.xlsx"

    kpis.to_csv(kpis_file, index=False)
    dom_summary.to_csv(dom_file, index=False)
    save_kpis_explained(kpis, explained_file)

    return {
        "kpis": str(kpis_file),
        "domain_summary": str(dom_file),
        "explained": str(explained_file),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
