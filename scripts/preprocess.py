import pandas as pd
import numpy as np
from pathlib import Path
from .utils import load_config, load_excel_sheets, standardize_id

# -----------------------------
# Helpers
# -----------------------------
def unify_timestamp(df, candidates, fmt="%d/%m/%Y %H:%M"):
    """Convert timestamp columns to datetime without timezone."""
    for c in candidates:
        if "time" in c.lower() or "timestamp" in c.lower():
            if "timestamp" not in df.columns:
                df = df.rename(columns={c: "timestamp"})
            df = df.loc[:, ~df.columns.duplicated()]
            # Convert to datetime and remove timezone
            df["timestamp"] = pd.to_datetime(df["timestamp"], format=fmt, errors="coerce", utc=True)
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # Remove timezone
            return df
    return df

def extract_performance(df, task, schema):
    """Standardize performance dataframe for one task."""
    df = df.copy()
    df["task"] = task.upper()

    # Standardize ID
    if "UserID" in df.columns:
        df[schema["id"]] = df["UserID"].apply(standardize_id)

    # Timestamps
    df = unify_timestamp(df, df.columns)

    # Reaction time mapping → reaction_time
    rt = None
    for col in ["ReactionTime_ms2", "ReactionTime_ms", "QAnswerRT_ms",
                "reaction_time_ms2", "ReactionTime"]:
        if col in df.columns:
            rt = pd.to_numeric(df[col], errors="coerce")
            if col == "ReactionTime":  # seconds → ms
                rt = rt * 1000
            break
    if rt is not None:
        df[schema["reaction_time"]] = rt

    # Correctness mapping → is_correct
    if "IsCorrect" in df.columns:
        df[schema["correct_flag"]] = pd.to_numeric(df["IsCorrect"], errors="coerce")
    elif "Outcome" in df.columns:
        df[schema["correct_flag"]] = df["Outcome"].str.lower().eq("correct").astype(int)

    # Accuracy
    if "Accuracy" in df.columns:
        df[schema["accuracy"]] = pd.to_numeric(df["Accuracy"], errors="coerce")
    elif schema["correct_flag"] in df.columns:
        df[schema["accuracy"]] = df[schema["correct_flag"]]

    # Build safe subset
    cols = [schema["id"], "timestamp", "task",
            schema["accuracy"], schema["reaction_time"], schema["correct_flag"]]
    cols_present = [c for c in cols if c in df.columns]
    df = df[cols_present]

    df = df.loc[:, ~df.columns.duplicated()]
    return df

def generate_missingness_report(df: pd.DataFrame, name: str):
    return pd.DataFrame({
        "file": name,
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "n_missing": [int(df[c].isna().sum()) for c in df.columns],
        "pct_missing": [float(df[c].isna().mean()) for c in df.columns]
    })

def drop_duplicates_report(df: pd.DataFrame, keys, name: str):
    before = len(df)
    df_clean = df.drop_duplicates(subset=keys) if set(keys).issubset(df.columns) else df
    after = len(df_clean)
    return df_clean, pd.DataFrame([{
        "file": name,
        "before": before,
        "after": after,
        "removed": before - after
    }])

def create_id_mapping(df, id_col="user_id"):
    uuids = sorted(df[id_col].dropna().unique())
    return {u: f"Student_{i+1}" for i, u in enumerate(uuids)}

def save_data_dictionary(df: pd.DataFrame, path: Path):
    dd = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "example_values": [", ".join(map(str, df[c].dropna().unique()[:5])) for c in df.columns]
    })
    dd.to_excel(path, index=False, engine='openpyxl')

# -----------------------------
# Main run
# -----------------------------
def run(config_path):
    cfg = load_config(config_path)
    dataset_file = Path(cfg["paths"]["data_raw"]) / cfg["files"]["excel_file"]
    sheets = cfg["files"]["sheets"]
    schema = cfg["schema"]

    raw = load_excel_sheets(str(dataset_file), sheets)
    out_dir = Path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Performance ---
    perf_frames = []
    for key in ["m1", "m2", "m3", "m4", "m5", "windows"]:
        df = raw.get(key, pd.DataFrame())
        if df is None or df.empty:
            continue
        perf_frames.append(extract_performance(df, key, schema))
    performance = pd.concat(perf_frames, ignore_index=True, sort=False)

    # --- ID mapping ---
    id_map = create_id_mapping(performance, schema["id"])
    performance["student_label"] = performance[schema["id"]].map(id_map)

    # Save mapping
    id_map_df = pd.DataFrame({
        "user_id": list(id_map.keys()),
        "label": list(id_map.values())
    })
    id_map_df.to_csv(out_dir / "id_mapping_secure.csv", index=False)

    # --- Emotion ---
    emo = raw.get("emotion", pd.DataFrame()).copy()
    if not emo.empty:
        if "UserID" in emo.columns:
            emo[schema["id"]] = emo["UserID"].apply(standardize_id)
        emo = unify_timestamp(emo, emo.columns)
        emo["student_label"] = emo[schema["id"]].map(id_map)

        # Baseline condition
        emo = emo.sort_values([schema["id"], "timestamp"])
        emo["condition"] = "current"
        for uid, g in emo.groupby(schema["id"]):
            n_baseline = max(1, int(len(g) * 0.1))
            emo.loc[g.head(n_baseline).index, "condition"] = "baseline"

        # Ensure numeric emotion columns
        for c in schema.get("emotion_cols", []):
            if c in emo.columns:
                emo[c] = pd.to_numeric(emo[c], errors="coerce").fillna(0)

        emo.to_csv(out_dir / "emotion_clean.csv", index=False)

    # --- Body ---
    body = raw.get("body", pd.DataFrame()).copy()
    if not body.empty:
        if "UserID" in body.columns:
            body[schema["id"]] = body["UserID"].apply(standardize_id)
        body = unify_timestamp(body, body.columns)
        body["student_label"] = body[schema["id"]].map(id_map)

        # Baseline condition
        body = body.sort_values([schema["id"], "timestamp"])
        body["condition"] = "current"
        for uid, g in body.groupby(schema["id"]):
            n_baseline = max(1, int(len(g) * 0.1))
            body.loc[g.head(n_baseline).index, "condition"] = "baseline"

        # Ensure numeric body columns
        for c in schema.get("body_cols", []):
            if c in body.columns:
                body[c] = pd.to_numeric(body[c], errors="coerce").fillna(0)

        body.to_csv(out_dir / "body_clean.csv", index=False)

    # --- Save performance ---
    performance.to_csv(out_dir / "performance_clean.csv", index=False)

    # --- Preprocessing reports ---
    reports, dups = [], []
    for df, name in [(performance, "performance"), (emo, "emotion"), (body, "body")]:
        if df is None or df.empty:
            continue
        reports.append(generate_missingness_report(df, name))
        df_clean, dup_rep = drop_duplicates_report(df, [schema["id"], "task", "timestamp"], name)
        dups.append(dup_rep)

    pd.concat(reports, ignore_index=True).to_csv(out_dir / "preprocessing_report.csv", index=False)
    pd.concat(dups, ignore_index=True).to_csv(out_dir / "duplicates_report.csv", index=False)

    # --- Data dictionary ---
    save_data_dictionary(performance, out_dir / "data_dictionary.xlsx")

    return {
        "performance_clean": str(out_dir / "performance_clean.csv"),
        "emotion_clean": str(out_dir / "emotion_clean.csv"),
        "body_clean": str(out_dir / "body_clean.csv"),
        "id_mapping": str(out_dir / "id_mapping_secure.csv"),
        "preprocessing_report": str(out_dir / "preprocessing_report.csv"),
        "data_dictionary": str(out_dir / "data_dictionary.xlsx"),
    }

# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
