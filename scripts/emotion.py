import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from .utils import nearest_time_merge, compute_basic_kpis

# `compute_emotion_features` and `compare_with_baseline` live in other scripts in this
# repo (body.py / integrate.py define similar helpers). Provide a small, clear
# implementation here to avoid external import errors and to reduce NaNs in reports.
def compute_emotion_features(df, id_col, task_col, emo_cols, ts_col):
    """Compute emotion features per (student, task).

    Features produced (per group):
      - n_obs: number of emotion detections
      - detect_rate: n_obs / total_rows_for_student_task (if available)
      - mean_confidence, std_confidence
      - mean_emotion_code, std_emotion_code
      - mean_box_area (w*h), mean_x, mean_y
      - top_emotion (most frequent label) and emotion_entropy

    The function is defensive: returns empty frame with expected id/task columns when input is empty.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[id_col, task_col])

    # Ensure numeric conversions for expected numeric columns
    for c in ["confidence", "emotion_code", "x", "y", "w", "h"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    grouped = df.groupby([id_col, task_col], dropna=False)
    for (uid, task), g in grouped:
        row = {id_col: uid, task_col: task}
        n = len(g)
        row["n_obs"] = int(n)

        # detection rate: if group contains a column 'trials' or similar use it, else leave NaN
        # (keeps function generic)
        # Confidence
        if "confidence" in g.columns:
            row["mean_confidence"] = float(g["confidence"].mean(skipna=True)) if g["confidence"].notna().any() else np.nan
            row["std_confidence"] = float(g["confidence"].std(skipna=True)) if g["confidence"].notna().any() else np.nan

        # emotion code stats
        if "emotion_code" in g.columns:
            row["mean_emotion_code"] = float(g["emotion_code"].mean(skipna=True)) if g["emotion_code"].notna().any() else np.nan
            row["std_emotion_code"] = float(g["emotion_code"].std(skipna=True)) if g["emotion_code"].notna().any() else np.nan

        # box area and positions
        if "w" in g.columns and "h" in g.columns:
            area = (pd.to_numeric(g["w"], errors="coerce") * pd.to_numeric(g["h"], errors="coerce"))
            row["mean_box_area"] = float(area.mean(skipna=True)) if area.notna().any() else np.nan
        if "x" in g.columns:
            row["mean_x"] = float(g["x"].mean(skipna=True)) if g["x"].notna().any() else np.nan
        if "y" in g.columns:
            row["mean_y"] = float(g["y"].mean(skipna=True)) if g["y"].notna().any() else np.nan

        # emotion label distribution: top and entropy
        if "emotion_label" in g.columns:
            labels = g["emotion_label"].dropna().astype(str)
            if not labels.empty:
                vc = labels.value_counts(normalize=True)
                top = vc.idxmax()
                # entropy (base 2)
                p = vc.values
                entropy = -float((p * np.log2(p)).sum()) if len(p) > 0 else 0.0
                row["top_emotion"] = top
                row["emotion_entropy"] = entropy
                # include top proportion
                row["top_emotion_prop"] = float(vc.max())
            else:
                row["top_emotion"] = None
                row["emotion_entropy"] = np.nan
                row["top_emotion_prop"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    # ensure numeric types where appropriate
    for c in [c for c in out.columns if c not in [id_col, task_col, "top_emotion"]]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def compare_with_baseline(base_df, with_df, metrics):
    """Lightweight comparison that aligns indices and computes differences and means.

    Returns a DataFrame with metric, base_mean, with_mean, diff.
    """
    rows = []
    for m in metrics:
        a = base_df.get(m) if hasattr(base_df, 'get') else base_df[m]
        b = with_df.get(m) if hasattr(with_df, 'get') else with_df[m]
        a = pd.to_numeric(a, errors="coerce").dropna()
        b = pd.to_numeric(b, errors="coerce").dropna()
        # align by index if possible
        if len(a) == 0 and len(b) == 0:
            continue
        base_mean = float(a.mean()) if len(a) > 0 else np.nan
        with_mean = float(b.mean()) if len(b) > 0 else np.nan
        diff = with_mean - base_mean if (not np.isnan(with_mean) and not np.isnan(base_mean)) else np.nan
        rows.append({"metric": m, "base_mean": base_mean, "with_mean": with_mean, "diff": diff})
    return pd.DataFrame(rows)


# --- helper to load csv robustly ---
def _load_any_csv(proc_dir: Path, filenames):
    for name in filenames:
        fpath = proc_dir / name
        if fpath.exists():
            try:
                return pd.read_csv(fpath)
            except Exception as e:
                print(f"⚠️ Failed loading {fpath}: {e}")
                continue
    return pd.DataFrame()


# === MAIN RUN ===
def run(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    schema = cfg["schema"]
    id_col = schema["id"]
    ts_col = schema["timestamp"]
    task_col = schema["task"]
    acc_col = schema["accuracy"]
    rt_col = schema["reaction_time"]

    proc_dir = Path(cfg["paths"]["data_processed"])

    # --- Load performance and emotion data robustly ---
    perf = _load_any_csv(proc_dir, ["performance_clean.csv", "performance_kpis.csv", "performance.csv"])
    emo = _load_any_csv(proc_dir, ["emotion_clean.csv", "emotion_features_by_task.csv", "emotion.csv"])

    print(f"After load: perf.empty={perf.empty}, emo.empty={emo.empty}")

    if perf.empty or emo.empty:
        print(f"Emotion step: perf.empty={perf.empty}, emo.empty={emo.empty} -> writing empty outputs")
        proc_dir.mkdir(parents=True, exist_ok=True)

        empty_feats = pd.DataFrame()
        empty_base = pd.DataFrame()
        empty_cmp = pd.DataFrame()
        empty_corr = pd.DataFrame()
        empty_linked = pd.DataFrame()

        empty_feats.to_csv(proc_dir / "emotion_features_by_task.csv", index=False)
        empty_base.to_csv(proc_dir / "emotion_perf_compare.csv", index=False)
        empty_cmp.to_csv(proc_dir / "comparison_emotion_vs_baseline.csv", index=False)
        empty_corr.to_csv(proc_dir / "emotion_perf_corr.csv", index=True)
        empty_linked.to_csv(proc_dir / "emotion_linked.csv", index=False)

        return {
            "features": str(proc_dir / "emotion_features_by_task.csv"),
            "linked": str(proc_dir / "emotion_linked.csv"),
            "compare": str(proc_dir / "emotion_perf_compare.csv"),
            "baseline_vs_emotion": str(proc_dir / "comparison_emotion_vs_baseline.csv"),
            "corr": str(proc_dir / "emotion_perf_corr.csv"),
        }

    # --- Ensure emotion has a usable timestamp ---
    if ts_col not in emo.columns:
        if "timestamp_str" in emo.columns:
            emo[ts_col] = pd.to_datetime(emo["timestamp_str"], errors="coerce", utc=True).dt.tz_localize(None)
            print("✅ Created timestamp from 'timestamp_str'")
        elif "start_time" in emo.columns:
            emo[ts_col] = pd.to_datetime(emo["start_time"], errors="coerce", utc=True).dt.tz_localize(None)
            print("✅ Created timestamp from 'start_time'")
        elif "end_time" in emo.columns:
            emo[ts_col] = pd.to_datetime(emo["end_time"], errors="coerce", utc=True).dt.tz_localize(None)
            print("✅ Created timestamp from 'end_time'")
        else:
            raise KeyError(f"No usable timestamp column found in emotion data: {list(emo.columns)}")

    # --- Convert numeric + categorical emotion columns ---
    emo_cols = ["emotion", "confidence", "x", "y", "w", "h"]
    for c in emo_cols:
        if c in emo.columns:
            if c == "emotion":
                # some pipelines export the literal string 'nan' or 'None' in the label
                # column; coerce those to real NaN first, then compute codes.
                emo["emotion_label"] = emo["emotion"].replace(["nan", "None", "NoneType"], np.nan)
                # strip / normalize whitespace
                emo.loc[emo["emotion_label"].notna(), "emotion_label"] = (
                    emo.loc[emo["emotion_label"].notna(), "emotion_label"].astype(str).str.strip()
                )
                # numeric labels -> numeric codes, otherwise categorical codes
                if pd.api.types.is_numeric_dtype(emo["emotion"]):
                    emo["emotion_code"] = pd.to_numeric(emo["emotion"], errors="coerce")
                    # keep original numeric label string as label
                    emo.loc[emo["emotion_label"].isna(), "emotion_label"] = (
                        emo.loc[emo["emotion_label"].isna(), "emotion"].astype(str)
                    )
                else:
                    # safe categorical coding: categories -> codes, -1 means NA
                    emo["emotion_code"] = pd.Categorical(emo["emotion_label"]).codes
                    emo.loc[emo["emotion_code"] == -1, "emotion_code"] = np.nan
            else:
                emo[c] = pd.to_numeric(emo[c], errors="coerce")

    if "emotion_label" in emo.columns:
        labels = emo["emotion_label"].dropna().unique().tolist()
        cnt = int(emo["emotion_label"].notna().sum())
        print(f"Emotion labels found: {labels[:30]} (non-missing={cnt})")

    if "emotion_code" in emo.columns:
        non_missing_codes = int(emo["emotion_code"].notna().sum())
        print(f"Emotion numeric codes non-missing={non_missing_codes}, "
              f"mean={emo['emotion_code'].mean() if non_missing_codes > 0 else np.nan}")

    # --- Standardize IDs, tasks, timestamps ---
    for df in [perf, emo]:
        df[id_col] = df[id_col].astype(str).str.upper().str.strip()
        if task_col in df.columns:
            df[task_col] = df[task_col].astype(str)

    perf[ts_col] = pd.to_datetime(perf[ts_col], errors="coerce", utc=True).dt.tz_localize(None)
    emo[ts_col] = pd.to_datetime(emo[ts_col], errors="coerce", utc=True).dt.tz_localize(None)

    perf = perf.dropna(subset=[ts_col])
    emo = emo.dropna(subset=[ts_col])

    # --- Link performance + emotion ---
    tol = cfg["merging"].get("time_tolerance_seconds", 5)
    linked = nearest_time_merge(perf, emo, on_id=id_col,
                                left_time=ts_col, right_time=ts_col,
                                tolerance_seconds=tol)
    print("linked shape:", getattr(linked, "shape", None))

    if ts_col not in linked.columns:
        for alt in ("timestamp_x", "timestamp_y", "timestamp_str", "start_time", "end_time"):
            if alt in linked.columns:
                linked[ts_col] = pd.to_datetime(linked[alt], errors="coerce", utc=True).dt.tz_localize(None)
                print(f"Using '{alt}' -> created '{ts_col}'")
                break

    if ts_col not in linked.columns:
        raise KeyError(f"Emotion analysis failed: no '{ts_col}' column in linked df. Got: {list(linked.columns)}")

    print("Final linked timestamp preview:\n", linked[[id_col, task_col, ts_col]].head())

    # --- Features ---
    emo_feats = compute_emotion_features(linked, id_col, task_col, emo_cols, ts_col)

    # --- Baseline KPIs ---
    base_kpis = compute_basic_kpis(perf, id_col, task_col, acc_col, rt_col)

    # --- Merge features into KPIs ---
    merged = base_kpis.merge(emo_feats, on=[id_col, task_col], how="left")

    # --- Compare baseline vs current ---
    metrics = ["avg_accuracy", "avg_reaction_time", "error_rate"]
    cmp = compare_with_baseline(
        base_kpis.groupby(id_col).mean(numeric_only=True),
        merged.groupby(id_col).mean(numeric_only=True),
        metrics,
    )

    # --- Correlation ---
    # Drop columns with all-NA or zero variance before computing correlation to
    # avoid huge NaN blocks in the output and make the matrix meaningful.
    corr_df = merged.drop(columns=[id_col, task_col])
    # drop all-NA columns
    all_na = corr_df.columns[corr_df.isna().all()].tolist()
    if all_na:
        print(f"Dropping all-NA columns before correlation: {all_na}")
        corr_df = corr_df.drop(columns=all_na)
    # drop zero-variance columns (std == 0 or nan)
    num = corr_df.select_dtypes(include=[np.number])
    zero_var = [c for c in num.columns if num[c].std(skipna=True) == 0 or np.isnan(num[c].std(skipna=True))]
    if zero_var:
        print(f"Dropping zero-variance columns before correlation: {zero_var}")
        corr_df = corr_df.drop(columns=zero_var)

    if corr_df.shape[1] == 0:
        corr = pd.DataFrame()
    else:
        corr = corr_df.corr(numeric_only=True)

    # --- Save outputs ---
    emo_feats.to_csv(proc_dir / "emotion_features_by_task.csv", index=False)
    merged.to_csv(proc_dir / "emotion_perf_compare.csv", index=False)
    cmp.to_csv(proc_dir / "comparison_emotion_vs_baseline.csv", index=False)
    corr.to_csv(proc_dir / "emotion_perf_corr.csv", index=True)
    linked.to_csv(proc_dir / "emotion_linked.csv", index=False)

    return {
        "features": str(proc_dir / "emotion_features_by_task.csv"),
        "linked": str(proc_dir / "emotion_linked.csv"),
        "compare": str(proc_dir / "emotion_perf_compare.csv"),
        "baseline_vs_emotion": str(proc_dir / "comparison_emotion_vs_baseline.csv"),
        "corr": str(proc_dir / "emotion_perf_corr.csv"),
    }
