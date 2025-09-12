import os
from pathlib import Path
import pandas as pd

def embed_figures(fig_dir: Path, markdown_lines: list):
    """
    Embed all figures in the report. Assumes .png figures in fig_dir.
    """
    fig_dir = Path(fig_dir)
    if not fig_dir.exists():
        return markdown_lines

    for fig_file in sorted(fig_dir.glob("*.png")):
        caption = fig_file.stem.replace("_", " ").title()
        markdown_lines.append(f"![{caption}]({fig_file})\n")
        markdown_lines.append(f"*Figure: {caption}*\n")
    return markdown_lines


def write_report(proc_dir):
    proc_dir = Path(proc_dir)
    lines = []

    # === Step 1: Title ===
    lines.append("# Executive Function (EF) Integrated Analysis Report\n")

    # === Step 2: Preprocessing Summary ===
    prep_file = proc_dir / "preprocessing_report.csv"
    if prep_file.exists():
        prep = pd.read_csv(prep_file)
        lines.append("## Preprocessing Summary\n")
        lines.append(prep.to_markdown(index=False))
        lines.append("\n")

    # === Step 3: Student Profiles Summary ===
    student_profiles_file = proc_dir / "student_profiles.xlsx"
    if student_profiles_file.exists():
        profiles = pd.read_excel(student_profiles_file)
        lines.append("## Student Profiles\n")
        lines.append(profiles.head(10).to_markdown(index=False))  # Show first 10
        lines.append("\n")

    # === Step 4: Early Indicators ===
    early_file = proc_dir / "early_indicators_report.csv"
    if early_file.exists():
        early = pd.read_csv(early_file)
        lines.append("## Early Indicators\n")
        lines.append(early.to_markdown(index=False))
        lines.append("\n")

    # === Step 5: Emotion & Body Stats ===
    for stat_file, title in [
        ("emotion_perf_corr.csv", "Emotion Statistical Tests"),
        ("body_perf_corr.csv", "Body Statistical Tests")
    ]:
        path = proc_dir / stat_file
        if path.exists():
            df = pd.read_csv(path)
            lines.append(f"## {title}\n")
            lines.append(df.to_markdown(index=False))
            lines.append("\n")

    # === Step 6: Student Summaries ===
    summary_file = proc_dir / "student_summaries.txt"
    if summary_file.exists():
        with open(summary_file) as f:
            summaries = f.read()
        lines.append("## Individual Student Summaries\n")
        lines.append(summaries)
        lines.append("\n")

    # === Step 7: Embed Figures ===
    fig_dir = proc_dir / "outputs_figures"
    lines = embed_figures(fig_dir, lines)

    # === Step 8: Save Markdown Report ===
    report_file = proc_dir / "EF_Final_Report.md"
    with open(report_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Markdown report saved at: {report_file}")
    return report_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dir", required=True, help="Path to processed data folder")
    args = parser.parse_args()
    write_report(args.proc_dir)
