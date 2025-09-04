
import os, yaml, numpy as np, pandas as pd

def run(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    schema = cfg["schema"]
    id_col = schema["id"]
    task_col = schema["task"]

    proc = cfg["paths"]["data_processed"]
    kpis_path = os.path.join(proc, "performance_kpis.csv")
    emo_link = os.path.join(proc, "emotion_linked.csv")
    body_link = os.path.join(proc, "body_linked.csv")

    kpis = pd.read_csv(kpis_path) if os.path.exists(kpis_path) else pd.DataFrame()
    emo = pd.read_csv(emo_link) if os.path.exists(emo_link) else pd.DataFrame()
    body = pd.read_csv(body_link) if os.path.exists(body_link) else pd.DataFrame()

    # Mean emotion + mean body per student-task
    emo_means = emo.groupby([id_col, task_col], dropna=False).mean(numeric_only=True).reset_index() if not emo.empty else pd.DataFrame()
    body_means = body.groupby([id_col, task_col], dropna=False).mean(numeric_only=True).reset_index() if not body.empty else pd.DataFrame()

    df = kpis.copy()
    if not emo_means.empty:
        df = df.merge(emo_means, on=[id_col, task_col], how="left", suffixes=("", "_emo"))
    if not body_means.empty:
        df = df.merge(body_means, on=[id_col, task_col], how="left", suffixes=("", "_body"))

    # Build EF profile per student (domain-level aggregation already computed in performance step)
    dom_path = os.path.join(proc, "performance_domain_summary.csv")
    dom = pd.read_csv(dom_path) if os.path.exists(dom_path) else pd.DataFrame()

    # Early indicators (very simple heuristics to flag students)
    # Example rule: low accuracy + high error + high reaction time variance + high arousal/fear + high movement_intensity
    flags = []
    if not df.empty:
        # Normalize columns for scoring
        numeric = df.select_dtypes(include=[np.number]).copy()
        norm = (numeric - numeric.mean()) / (numeric.std(ddof=0) + 1e-9)
        score = np.zeros(len(df))

        for col in numeric.columns:
            if "avg_accuracy" in col:
                score += (-norm[col])  # lower accuracy worse
            if "error_rate" in col:
                score += (norm[col])   # higher error worse
            if "avg_reaction_time" in col:
                score += (norm[col])   # slower worse
            if "arousal" in col or "anger" in col or "fear" in col:
                score += (norm[col])   # higher negative emotion worse
            if "movement_intensity" in col or "stillness_ratio" in col:
                score += (norm[col])   # treat as potential restlessness signal (domain-specific tuning needed)

        df["risk_score"] = score
        # Flag top 10% as "needs_review"
        thresh = np.nanpercentile(df["risk_score"], 90) if df["risk_score"].notna().any() else np.nan
        df["needs_review"] = df["risk_score"] >= thresh if not np.isnan(thresh) else False

    out_dir = cfg["paths"]["data_processed"]
    integrate_path = os.path.join(out_dir, "integrated_ef_table.csv")
    df.to_csv(integrate_path, index=False)

    # Build per-student EF profile summary (join with domain summary if available)
    if not dom.empty:
        ef_profile = dom.pivot_table(index=id_col, columns="domain",
                                     values=["avg_accuracy", "avg_reaction_time", "error_rate"],
                                     aggfunc="mean")
        ef_profile.columns = [f"{a}_{b}" for a,b in ef_profile.columns]
        ef_profile = ef_profile.reset_index()
    else:
        ef_profile = pd.DataFrame()

    ef_path = os.path.join(out_dir, "ef_profiles.csv")
    ef_profile.to_csv(ef_path, index=False)

    return {"integrated": "data/processed/integrated_ef_table.csv",
            "ef_profiles": "data/processed/ef_profiles.csv"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
