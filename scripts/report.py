import os, yaml, pandas as pd
from datetime import datetime

# -----------------------------
# Helper Functions
# -----------------------------
def section(title):
    return f"\n\n## {title}\n\n"

def embed_figures(fig_dir):
    """Group figures into categories and embed inline with captions."""
    lines = []
    if not os.path.exists(fig_dir):
        return lines

    categories = {
        "Per-Student KPIs": [],
        "Heatmaps": [],
        "Other Figures": []
    }

    for fname in sorted(os.listdir(fig_dir)):
        if not fname.endswith(".png"):
            continue
        fpath = os.path.join(fig_dir, fname)
        caption = fname.replace("_", " ").replace(".png", "")

        if "heatmap" in fname.lower():
            categories["Heatmaps"].append((fpath, caption))
        elif "accuracy" in fname.lower() or "reaction" in fname.lower() or "error" in fname.lower():
            categories["Per-Student KPIs"].append((fpath, caption))
        else:
            categories["Other Figures"].append((fpath, caption))

    for cat, figs in categories.items():
        if figs:
            lines.append(section(cat))
            for fpath, caption in figs:
                lines.append(f"![{caption}]({fpath})")
                lines.append(f"*Figure: {caption}*")
    return lines

# -----------------------------
# Main Report Writer (Markdown only)
# -----------------------------
def write_report(cfg):
    out_dir = cfg["paths"]["outputs_reports"]
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "EF_Report.md")
    
    # Processed paths
    proc = cfg["paths"]["data_processed"]
    perf_path   = os.path.join(proc, "performance_kpis.csv")
    dom_path    = os.path.join(proc, "performance_domain_summary.csv")
    emo_comp    = os.path.join(proc, "comparison_emotion_vs_baseline.csv")
    body_comp   = os.path.join(proc, "comparison_body_vs_baseline.csv")
    emo_corr    = os.path.join(proc, "emotion_perf_corr.csv")
    body_corr   = os.path.join(proc, "body_perf_corr.csv")
    ef_prof     = os.path.join(proc, "ef_profiles.csv")
    integ       = os.path.join(proc, "integrated_ef_table.csv")
    indicators  = os.path.join(proc, "early_indicators_report.csv")
    summaries   = os.path.join(proc, "student_summaries.txt")
    prep_rep    = os.path.join(proc, "preprocessing_report.csv")
    dict_xlsx   = os.path.join(proc, "data_dictionary.xlsx")
    fig_dir     = cfg["paths"]["outputs_figures"]

    def exists_read(p):
        return os.path.exists(p) and os.path.getsize(p) > 0

    lines = []
    lines.append(f"# {cfg['report']['title']}")
    lines.append(f"**Author**: {cfg['report']['author']}  ")
    lines.append(f"**Generated**: {datetime.utcnow().isoformat()}Z")

    # === Core Performance KPIs ===
    if exists_read(perf_path):
        perf = pd.read_csv(perf_path)
        lines.append(section("Core Performance KPIs"))
        lines.append(perf.head(20).to_markdown(index=False))

    if exists_read(dom_path):
        dom = pd.read_csv(dom_path)
        lines.append(section("Executive Function Domains (WM / IC / CF)"))
        lines.append(dom.head(20).to_markdown(index=False))

    # === Emotion Analysis ===
    if exists_read(emo_comp):
        emo = pd.read_csv(emo_comp)
        lines.append(section("Emotion vs Baseline Performance"))
        lines.append(emo.to_markdown(index=False))

    if exists_read(emo_corr):
        corr = pd.read_csv(emo_corr)
        lines.append(section("Emotion–Performance Correlations"))
        lines.append(corr.to_markdown(index=False))

    # === Body Motion Analysis ===
    if exists_read(body_comp):
        body = pd.read_csv(body_comp)
        lines.append(section("Body Motion vs Baseline Performance"))
        lines.append(body.to_markdown(index=False))

    if exists_read(body_corr):
        corr = pd.read_csv(body_corr)
        lines.append(section("Body–Performance Correlations"))
        lines.append(corr.to_markdown(index=False))

    # === Integrated Profiles ===
    if exists_read(ef_prof):
        ef = pd.read_csv(ef_prof)
        lines.append(section("Executive Function Profiles (Per Student)"))
        lines.append(ef.to_markdown(index=False))

    if exists_read(integ):
        integ_df = pd.read_csv(integ)
        if "needs_review" in integ_df.columns:
            flagged = integ_df[integ_df["needs_review"] == True]
            lines.append(section("Early Indicators (Needs Review)"))
            lines.append(flagged.to_markdown(index=False) if len(flagged) > 0 else "  No students flagged in current data.")

    if exists_read(indicators):
        ind = pd.read_csv(indicators)
        lines.append(section("Early Indicators Report (Quantitative)"))
        lines.append(ind.to_markdown(index=False))

    # === Student Summaries ===
    if exists_read(summaries):
        lines.append(section("Student Profiles (Narrative Summaries)"))
        with open(summaries, "r") as f:
            summaries_text = f.read().split("\n\n")
        for summary in summaries_text:
            if summary.strip():
                lines.append(f"```\n{summary}\n```")

    # === Data Preprocessing ===
    if exists_read(prep_rep):
        prep = pd.read_csv(prep_rep)
        lines.append(section("Data Preprocessing"))
        lines.append("Summary of missing values handled, duplicates removed, and ID normalization:")
        lines.append(prep.to_markdown(index=False))

    # Data dictionary preview
    if exists_read(dict_xlsx):
        try:
            dict_df = pd.read_excel(dict_xlsx)
            lines.append("\n**Data Dictionary (Preview)**")
            lines.append(dict_df.head(10).to_markdown(index=False))
            lines.append("\n*Full data dictionary available in* `data/processed/data_dictionary.xlsx`.")
        except Exception as e:
            lines.append(f"Could not read data dictionary: {e}")

    # Appendices
    lines.append(section("Appendix A: Statistical Notes"))
    lines.append("All t-tests, Wilcoxon tests, confidence intervals, and effect sizes (Cohen’s d) are computed per metric per student group.")

    lines.append(section("Appendix B: Figures & Heatmaps"))
    lines.extend(embed_figures(fig_dir))

    # Save Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("✅ Markdown report generated at:", md_path)
    return md_path

# -----------------------------
# Run via config YAML
# -----------------------------
def run(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return write_report(cfg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
