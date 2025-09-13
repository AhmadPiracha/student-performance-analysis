import pandas as pd
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import textwrap


files = ["data/processed/performance_clean.csv", "data/processed/emotion_clean.csv"]
checks = ["timestamp","Timestamp","start_time","Start_Time","UserID","user_id"]

for f in files:
    try:
        df = pd.read_csv(f, nrows=50, low_memory=False)
    except Exception as e:
        print(f"ERROR reading {f}: {e}")
        continue
    print("\nFILE:", f)
    print("Columns:", df.columns.tolist())
    for c in checks:
        print(f"Has column '{c}':", c in df.columns)
    print("Dtypes:", df.dtypes.to_dict())
    for cand in ["timestamp", "start_time", "Timestamp", "startTime"]:
        if cand in df.columns:
            parsed = pd.to_datetime(df[cand], errors='coerce', utc=True, dayfirst=True)
            nnull = parsed.isna().sum()
            print(f"Parse check '{cand}': {len(parsed)} rows, {nnull} NaT")
    print("Sample rows:")
    print(df.head(3).to_string(index=False))

ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

TIMESTAMP_CANDIDATES = ["timestamp", "times", "start_time", "end_time", "time", "relative_time"]

def summarize_df(fp: Path):
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception as e:
        return {"file": str(fp), "error": f"read_error: {e}"}
    cols = list(df.columns)
    nrows = len(df)
    missing = df.isna().sum().to_dict()
    pct_missing = {k: v / max(1, nrows) for k, v in missing.items()}
    dtypes = df.dtypes.astype(str).to_dict()
    unique_counts = {c: int(df[c].nunique(dropna=True)) for c in cols}
    # detect constant columns or single unique value (excluding NaN)
    constants = [c for c, u in unique_counts.items() if u <= 1]
    # zero-variance numeric columns
    zero_var = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].dropna().size == 0:
                zero_var.append(c)
            else:
                if float(df[c].dropna().std(ddof=1)) == 0.0:
                    zero_var.append(c)
    # timestamp parsing diagnostics
    ts_parsed = {}
    for cand in TIMESTAMP_CANDIDATES:
        if cand in df.columns:
            parsed = pd.to_datetime(df[cand], errors="coerce", utc=True)
            ts_parsed[cand] = {
                "found": True,
                "n_missing_after_parse": int(parsed.isna().sum()),
                "pct_missing_after_parse": float(parsed.isna().sum() / max(1, nrows)),
                "sample_parsed": str(parsed.dropna().head(3).astype(str).tolist())
            }
        else:
            ts_parsed[cand] = {"found": False}
    # per-student counts if possible
    student_counts = {}
    for candidate in ["user_id", "UserID", "student_label"]:
        if candidate in df.columns:
            student_counts = df[candidate].value_counts(dropna=False).to_dict()
            break
    summary = {
        "file": str(fp.relative_to(ROOT)),
        "nrows": nrows,
        "ncols": len(cols),
        "columns": cols,
        "dtypes": dtypes,
        "missing_counts": {k: int(v) for k, v in missing.items()},
        "pct_missing": {k: float(v) for k, v in pct_missing.items()},
        "unique_counts": unique_counts,
        "constant_or_single_value_columns": constants,
        "zero_variance_numeric_columns": zero_var,
        "timestamp_parse": ts_parsed,
        "sample_student_counts": {k: int(v) for k, v in list(student_counts.items())[:20]}
    }
    return summary

def scan_processed():
    summaries = []
    for fp in sorted(PROCESSED.glob("*.csv")):
        summaries.append(summarize_df(fp))
    out_json = OUT / "processed_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote processed summary to {out_json}")
    # produce a compact CSV summary
    rows = []
    for s in summaries:
        if "error" in s:
            rows.append({"file": s["file"], "error": s["error"]})
            continue
        rows.append({
            "file": s["file"],
            "nrows": s["nrows"],
            "ncols": s["ncols"],
            "n_constant_cols": len(s["constant_or_single_value_columns"]),
            "n_zero_variance_cols": len(s["zero_variance_numeric_columns"]),
            "has_timestamp_candidate": any(v.get("found", False) for v in s["timestamp_parse"].values()),
            "timestamp_candidates": ",".join([k for k, v in s["timestamp_parse"].items() if v.get("found", False)]),
        })
    pd.DataFrame(rows).to_csv(OUT / "processed_overview.csv", index=False)
    print(f"Wrote processed_overview.csv")

def check_scripts_conflicts():
    findings = []
    # check for multiple write_report definitions or conflicting filenames
    for fp in sorted(SCRIPTS.glob("*.py")):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        if "def write_report" in text or "write_report(" in text:
            findings.append({"script": str(fp.relative_to(ROOT)), "has_write_report": True})
        if "EF_Final_Report" in text or "EF_Report" in text:
            findings.append({"script": str(fp.relative_to(ROOT)), "mentions_report_names": True})
    # check for pandoc usage in export_report
    export_fp = SCRIPTS / "export_report.py"
    if export_fp.exists():
        t = export_fp.read_text(encoding="utf-8", errors="ignore")
        findings.append({"script": str(export_fp.relative_to(ROOT)), "mentions_pandoc": "pandoc" in t or "pypandoc" in t})
    out = OUT / "script_conflicts.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"Wrote script conflicts to {out}")
    return findings

def identify_high_risk_files():
    # quick heuristics from processed_overview.csv
    overview = pd.read_csv(OUT / "processed_overview.csv")
    high_missing = overview[overview["n_constant_cols"] > 0]
    high_zero_var = overview[overview["n_zero_variance_cols"] > 0]
    print("Files with constant columns:", high_missing["file"].tolist())
    print("Files with zero-variance numeric columns:", high_zero_var["file"].tolist())
    return {
        "constant_cols_files": high_missing["file"].tolist(),
        "zero_var_files": high_zero_var["file"].tolist()
    }

def main():
    print("Starting diagnostics...")
    scan_processed()
    script_findings = check_scripts_conflicts()
    risk = identify_high_risk_files()
    # quick human-friendly summary
    summary_lines = []
    summary_lines.append(f"Processed CSVs scanned: {len(list(PROCESSED.glob('*.csv')))}")
    summary_lines.append(f"Scripts reviewed in {SCRIPTS}: {len(list(SCRIPTS.glob('*.py')))}")
    if script_findings:
        summary_lines.append("Potential script conflicts or report-name mentions found. See outputs/diagnostics/script_conflicts.json")
    summary_lines.append(f"High risk files: {risk}")
    (OUT / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("Diagnostics complete. See outputs/diagnostics/ for results.")

if __name__ == "__main__":
    main()
