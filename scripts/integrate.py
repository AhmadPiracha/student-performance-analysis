import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind

# -----------------------------
# Effect size helper
# -----------------------------
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * x.std()**2 + (ny - 1) * y.std()**2) / dof)
    return (x.mean() - y.mean()) / pooled_std

# -----------------------------
# Helper to remove timezone
# -----------------------------
def make_timestamps_naive(df):
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = df[col].dt.tz_localize(None)
    return df

# -----------------------------
# Main run function
# -----------------------------
def run(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    schema = cfg["schema"]
    id_col = schema["id"]
    task_col = schema["task"]

    proc = Path(cfg["paths"]["data_processed"])

    # === Step 1: Load feature CSVs safely ===
    kpis = pd.read_csv(proc / "performance_kpis.csv") if (proc / "performance_kpis.csv").exists() else pd.DataFrame()
    emo = pd.read_csv(proc / "emotion_features_by_task.csv") if (proc / "emotion_features_by_task.csv").exists() else pd.DataFrame()
    body = pd.read_csv(proc / "body_features_by_task.csv") if (proc / "body_features_by_task.csv").exists() else pd.DataFrame()
    dom = pd.read_csv(proc / "performance_domain_summary.csv") if (proc / "performance_domain_summary.csv").exists() else pd.DataFrame()

    # === Step 1b: Normalize ID column names ===
    for df_ in [kpis, emo, body, dom]:
        if not df_.empty:
            if 'UserID' in df_.columns:
                df_.rename(columns={'UserID': id_col}, inplace=True)
            if id_col not in df_.columns:
                raise ValueError(f"Expected ID column '{id_col}' not found in DataFrame")
            if task_col in df_.columns:
                df_[task_col] = df_[task_col].astype(str)
            df_[id_col] = df_[id_col].astype(str)

    # === Step 2: Merge per-task features ===
    df = kpis.copy()
    if not emo.empty:
        df = df.merge(emo, on=[id_col, task_col], how="left")
    if not body.empty:
        df = df.merge(body, on=[id_col, task_col], how="left")

    # === Step 3: Compute EF Profiles per student ===
    if not dom.empty:
        ef_profile = dom.pivot_table(
            index=id_col,
            columns="domain",
            values=["avg_accuracy", "avg_reaction_time", "error_rate"],
            aggfunc="mean"
        )
        ef_profile.columns = [f"{a}_{b}" for a, b in ef_profile.columns]
        ef_profile = ef_profile.reset_index()
    else:
        ef_profile = pd.DataFrame()

    # Add emotion + body averages per student
    emo_body_avgs = df.groupby(id_col).mean(numeric_only=True).reset_index()
    student_profiles = ef_profile.merge(emo_body_avgs, on=id_col, how="left")

    # === Step 4: Map user_id → Student #N safely ===
    id_map_file = proc / "id_mapping_secure.csv"
    if id_map_file.exists():
        id_map = pd.read_csv(id_map_file)  # columns: user_id,label
        if 'user_id' not in id_map.columns or 'label' not in id_map.columns:
            raise ValueError("id_mapping_secure.csv must have columns 'user_id' and 'label'")
        uuid_to_label = dict(zip(id_map['user_id'].astype(str), id_map['label']))
    else:
        raise FileNotFoundError(f"{id_map_file} not found")

    # Apply mapping safely
    for df_ in [df, ef_profile, emo_body_avgs, student_profiles]:
        if id_col in df_.columns:
            df_[id_col] = df_[id_col].map(uuid_to_label).fillna(df_[id_col])

    # === Step 5: Compute Risk Scores ===
    if not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in [id_col, task_col]]
        numeric = df[numeric_cols].copy()
        numeric = numeric.fillna(numeric.mean())

        if numeric.empty:
            df["risk_score"] = np.nan
            df["needs_review"] = False
        else:
            norm = (numeric - numeric.mean()) / (numeric.std(ddof=0) + 1e-9)
            score = np.zeros(len(df))
            for col in numeric.columns:
                if "avg_accuracy" in col:
                    score += -norm[col]
                elif "error_rate" in col:
                    score += norm[col]
                elif "avg_reaction_time" in col:
                    score += norm[col]
                elif any(x in col for x in ["fear", "anger", "arousal", "movement", "pupil"]):
                    score += norm[col]

            df["risk_score"] = score
            thresh = np.nanpercentile(score, 90) if np.any(~np.isnan(score)) else np.inf
            df["needs_review"] = df["risk_score"] >= thresh

        # Save early indicators
        early_flags = df.groupby(id_col)[["risk_score", "needs_review"]].max().reset_index()
        early_flags = make_timestamps_naive(early_flags)
        early_flags.to_excel(proc / "early_indicators_report.xlsx", index=False, engine='openpyxl')
        # Also write CSV so downstream report readers find early_indicators_report.csv
        try:
            early_flags.to_csv(proc / "early_indicators_report.csv", index=False)
        except Exception as e:
            # Non-fatal; warn and continue
            print(f"Warning: failed to write early_indicators_report.csv: {e}")
    else:
        df["risk_score"] = np.nan
        df["needs_review"] = False

    # === Step 6: Save outputs ===
    student_profiles = make_timestamps_naive(student_profiles)
    df = make_timestamps_naive(df)
    student_profiles.to_excel(proc / "student_profiles.xlsx", index=False, engine='openpyxl')
    integrated_file = proc / "integrated_ef_table.xlsx"
    df.to_excel(integrated_file, index=False, engine='openpyxl')

    # === Step 7: Generate student summaries ===
    summaries = []
    for _, row in student_profiles.iterrows():
        sid = row[id_col]
        lines = [f"Student {sid} Profile:"]
        for dom_name in ["working_memory", "inhibitory_control", "cognitive_flexibility"]:
            col = f"avg_accuracy_{dom_name}"
            if col in row and not pd.isna(row[col]):
                lines.append(f"- {dom_name.replace('_',' ').title()} Accuracy: {row[col]:.2f}")
        if "risk_score" in df.columns:
            risk_val = df[df[id_col] == sid]["risk_score"].mean()
            if not pd.isna(risk_val):
                lines.append(f"- Risk Score: {risk_val:.2f}")
        lines.append("- Emotion & body indicators included.")
        summaries.append("\n".join(lines))

    with open(proc / "student_summaries.txt", "w") as f:
        f.write("\n\n".join(summaries))

    # === Step 8: Safe emotion/body stats ===
    for data, out_file in [(emo, "emotion_perf_corr.xlsx"), (body, "body_perf_corr.xlsx")]:
        if not data.empty and 'condition' in data.columns:
            metrics = [c for c in data.columns if c not in [id_col, task_col, "condition"]]
            results = []
            for metric in metrics:
                group1 = data[data['condition']=='baseline'][metric].dropna()
                group2 = data[data['condition']=='current'][metric].dropna()
                if len(group1) < 2 or len(group2) < 2:
                    p_val = np.nan
                    d_val = np.nan
                else:
                    p_val = ttest_ind(group1, group2).pvalue
                    d_val = cohen_d(group1, group2)
                results.append({'metric': metric, 'p_value': p_val, 'cohen_d': d_val})
            pd.DataFrame(results).to_excel(proc / out_file, index=False, engine='openpyxl')

    return {
    "integrated_table": str(integrated_file),
    "student_profiles": str(proc / "student_profiles.xlsx"),
    "early_indicators": str(proc / "early_indicators_report.xlsx"),
    "early_indicators_csv": str(proc / "early_indicators_report.csv"),
    "summaries": str(proc / "student_summaries.txt"),
        "emotion_stats": str(proc / "emotion_perf_corr.xlsx") if not emo.empty else None,
        "body_stats": str(proc / "body_perf_corr.xlsx") if not body.empty else None
    }

# -----------------------------
# CLI entry
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
