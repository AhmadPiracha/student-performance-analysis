
import os, yaml, pandas as pd
from datetime import datetime

def section(title):
    return f"\n\n## {title}\n\n"

def write_report(cfg):
    out_dir = cfg["paths"]["outputs_reports"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "EF_Report.md")

    schema = cfg["schema"]
    id_col = schema["id"]

    # Load data
    proc = cfg["paths"]["data_processed"]
    perf_path = os.path.join(proc, "performance_kpis.csv")
    dom_path  = os.path.join(proc, "performance_domain_summary.csv")
    emo_comp  = os.path.join(proc, "emotion_perf_compare.csv")
    body_comp = os.path.join(proc, "body_perf_compare.csv")
    integ     = os.path.join(proc, "integrated_ef_table.csv")
    ef_prof   = os.path.join(proc, "ef_profiles.csv")

    def exists_read(p):
        return os.path.exists(p) and os.path.getsize(p) > 0

    lines = []
    lines.append(f"# {cfg['report']['title']}")
    lines.append(f"**Author**: {cfg['report']['author']}  ")
    lines.append(f"**Generated**: {datetime.utcnow().isoformat()}Z")

    # Summary KPIs
    if exists_read(perf_path):
        perf = pd.read_csv(perf_path)
        lines.append(section("Core Performance KPIs"))
        lines.append(perf.head(20).to_markdown(index=False))

    if exists_read(dom_path):
        dom = pd.read_csv(dom_path)
        lines.append(section("Executive Function Domains (Summary)"))
        lines.append(dom.head(20).to_markdown(index=False))

    if exists_read(emo_comp):
        emo = pd.read_csv(emo_comp)
        lines.append(section("Emotion vs Performance (Comparison)"))
        lines.append(emo.head(20).to_markdown(index=False))

    if exists_read(body_comp):
        body = pd.read_csv(body_comp)
        lines.append(section("Body Motion vs Performance (Comparison)"))
        lines.append(body.head(20).to_markdown(index=False))

    if exists_read(ef_prof):
        ef = pd.read_csv(ef_prof)
        lines.append(section("EF Profiles (Per Student)"))
        lines.append(ef.head(20).to_markdown(index=False))

    if exists_read(integ):
        integ_df = pd.read_csv(integ)
        if "needs_review" in integ_df.columns:
            flagged = integ_df[integ_df["needs_review"]==True]
            lines.append(section("Early Indicators (Needs Review)"))
            if len(flagged) > 0:
                lines.append(flagged.head(50).to_markdown(index=False))
            else:
                lines.append("No students flagged in current data.")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    return path

def run(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    p = write_report(cfg)
    return {"report": p}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
