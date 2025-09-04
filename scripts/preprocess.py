import pandas as pd
from pathlib import Path
from .utils import load_config, load_excel_sheets, standardize_id

def unify_timestamp(df, candidates, fmt="%d/%m/%Y %H:%M"):
    for c in candidates:
        if "time" in c.lower() or "timestamp" in c.lower():
            if "timestamp" not in df.columns:
                df = df.rename(columns={c: "timestamp"})
            df = df.loc[:, ~df.columns.duplicated()]
            df["timestamp"] = pd.to_datetime(df["timestamp"], format=fmt, errors="coerce", utc=True)
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

    # Reaction time mapping → rt_ms
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
    cols = [schema["id"], schema["timestamp"], "task",
            schema["accuracy"], schema["reaction_time"], schema["correct_flag"]]
    cols_present = [c for c in cols if c in df.columns]
    df = df[cols_present]

    # Ensure unique column names
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def run(config_path):
    cfg = load_config(config_path)
    dataset_file = Path(cfg["paths"]["data_raw"]) / cfg["files"]["excel_file"]
    sheets = cfg["files"]["sheets"]
    schema = cfg["schema"]

    raw = load_excel_sheets(str(dataset_file), sheets)
    out_dir = Path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Performance data ===
    perf_frames = []
    for key in ["m1", "m2", "m3", "m4", "m5", "windows"]:
        df = raw.get(key, pd.DataFrame())
        if df is None or df.empty:
            continue
        perf_frames.append(extract_performance(df, key, schema))

    performance = pd.concat(perf_frames, ignore_index=True, sort=False)
    performance.to_csv(out_dir / "performance_clean.csv", index=False)

    # === Emotion ===
    emo = raw.get("emotion", pd.DataFrame()).copy()
    if not emo.empty:
        if "UserID" in emo.columns:
            emo[schema["id"]] = emo["UserID"].apply(standardize_id)
        emo = unify_timestamp(emo, emo.columns)
        emo.to_csv(out_dir / "emotion_clean.csv", index=False)

    # === Body ===
    body = raw.get("body", pd.DataFrame()).copy()
    if not body.empty:
        if "UserID" in body.columns:
            body[schema["id"]] = body["UserID"].apply(standardize_id)
        body = unify_timestamp(body, body.columns)
        body.to_csv(out_dir / "body_clean.csv", index=False)

    return {
        "performance_clean": str(out_dir / "performance_clean.csv"),
        "emotion_clean": str(out_dir / "emotion_clean.csv"),
        "body_clean": str(out_dir / "body_clean.csv"),
    }
