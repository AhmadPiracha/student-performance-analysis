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

    # collect dropped columns info
    dropped_columns = {}

    def load_df(path):
        """Try reading a CSV first, then an XLSX with the same path stem.

        Returns a DataFrame or None if no readable file exists.
        After reading, drop columns that are all-NaN and replace remaining NaNs with empty string
        to make markdown output concise.
        """
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                df = pd.read_csv(path)
            except Exception:
                try:
                    df = pd.read_excel(path)
                except Exception:
                    return None
        else:
            # try xlsx variant
            xlsx = os.path.splitext(path)[0] + ".xlsx"
            if os.path.exists(xlsx) and os.path.getsize(xlsx) > 0:
                try:
                    df = pd.read_excel(xlsx)
                except Exception:
                    return None
            else:
                return None

        # Drop columns that are entirely NA to reduce noise, then fill remaining NaNs
        try:
            all_cols = list(df.columns)
            df = df.dropna(axis=1, how='all')
            dropped = [c for c in all_cols if c not in df.columns]
            if dropped:
                dropped_columns[path] = dropped
            df = df.fillna("")
        except Exception:
            pass
        return df

    lines = []
    lines.append(f"# {cfg['report']['title']}")
    lines.append(f"**Author**: {cfg['report']['author']}  ")
    lines.append(f"**Generated**: {datetime.utcnow().isoformat()}Z")

    # === Core Performance KPIs ===
    perf = load_df(perf_path)
    if perf is not None:
        lines.append(section("Core Performance KPIs"))
        lines.append(perf.head(20).to_markdown(index=False))

    dom = load_df(dom_path)
    if dom is not None:
        lines.append(section("Executive Function Domains (WM / IC / CF)"))
        lines.append(dom.head(20).to_markdown(index=False))

    # === Emotion Analysis ===
    emo = load_df(emo_comp)
    if emo is not None:
        lines.append(section("Emotion vs Baseline Performance"))
        lines.append(emo.to_markdown(index=False))

    corr = load_df(emo_corr)
    if corr is not None:
        lines.append(section("Emotion–Performance Correlations"))
        lines.append(corr.to_markdown(index=False))

    # === Body Motion Analysis ===
    body = load_df(body_comp)
    if body is not None:
        lines.append(section("Body Motion vs Baseline Performance"))
        lines.append(body.to_markdown(index=False))

    corr = load_df(body_corr)
    if corr is not None:
        lines.append(section("Body–Performance Correlations"))
        lines.append(corr.to_markdown(index=False))

    # === Integrated Profiles ===
    ef = load_df(ef_prof)
    if ef is not None:
        lines.append(section("Executive Function Profiles (Per Student)"))
        lines.append(ef.to_markdown(index=False))

    integ_df = load_df(integ)
    if integ_df is not None:
        if "needs_review" in integ_df.columns:
            flagged = integ_df[integ_df["needs_review"] == True]
            lines.append(section("Early Indicators (Needs Review)"))
            lines.append(flagged.to_markdown(index=False) if len(flagged) > 0 else "  No students flagged in current data.")

    ind = load_df(indicators)
    if ind is not None:
        lines.append(section("Early Indicators Report (Quantitative)"))
        lines.append(ind.to_markdown(index=False))

    # === Student Summaries ===
    if os.path.exists(summaries) and os.path.getsize(summaries) > 0:
        lines.append(section("Student Profiles (Narrative Summaries)"))
        with open(summaries, "r", encoding="utf-8") as f:
            summaries_text = f.read().split("\n\n")
        for summary in summaries_text:
            if summary.strip():
                lines.append(f"```\n{summary}\n```")

    # === Data Preprocessing ===
    prep = load_df(prep_rep)
    if prep is not None:
        lines.append(section("Data Preprocessing"))
        lines.append("Summary of missing values handled, duplicates removed, and ID normalization:")
        lines.append(prep.to_markdown(index=False))

    # Report columns dropped due to being all-NaN
    if dropped_columns:
        lines.append("\n**Columns dropped (all-NA) during report load:**")
        for p, cols in dropped_columns.items():
            lines.append(f"- {os.path.basename(p)}: {', '.join(cols)}")

    # Data dictionary preview
    # Data dictionary preview: try xlsx first
    dict_df = None
    if os.path.exists(dict_xlsx) and os.path.getsize(dict_xlsx) > 0:
        try:
            dict_df = pd.read_excel(dict_xlsx)
        except Exception:
            dict_df = None
    if dict_df is not None:
        dict_df = dict_df.dropna(axis=1, how='all').fillna("")
        lines.append("\n**Data Dictionary (Preview)**")
        lines.append(dict_df.head(10).to_markdown(index=False))
        lines.append("\n*Full data dictionary available in* `data/processed/data_dictionary.xlsx`.")

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
