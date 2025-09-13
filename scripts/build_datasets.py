
import os, yaml, pandas as pd
import logging

def run(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    schema = cfg["schema"]
    id_col = schema["id"]
    task_col = schema["task"]

    proc = paths["data_processed"]
    out_datasets = paths["outputs_datasets"]
    os.makedirs(out_datasets, exist_ok=True)

    perf_kpis = os.path.join(proc, "performance_kpis.csv")
    emo_link = os.path.join(proc, "emotion_linked.csv")
    body_link = os.path.join(proc, "body_linked.csv")

    logger = logging.getLogger(__name__)

    def csv_has_header(path: str) -> bool:
        """Return True if file exists and appears to have a non-empty header row.

        This guards against files that exist but are empty or contain only whitespace
        which would cause pandas.read_csv to raise "No columns to parse from file".
        """
        try:
            if not os.path.exists(path):
                return False
            if os.path.getsize(path) == 0:
                return False
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        # rudimentary check: header should contain at least one comma or tab
                        return ("," in line) or ("\t" in line) or (len(line.strip()) > 0)
            return False
        except Exception:
            return False

    games = pd.read_csv(perf_kpis) if os.path.exists(perf_kpis) else pd.DataFrame()
    games.to_csv(os.path.join(out_datasets, "games_summary.csv"), index=False)

    if csv_has_header(emo_link):
        try:
            emo = pd.read_csv(emo_link)
            ge = games.merge(emo.groupby([id_col, task_col]).mean(numeric_only=True).reset_index(),
                             on=[id_col, task_col], how="left")
            ge.to_csv(os.path.join(out_datasets, "games_emotion_summary.csv"), index=False)
        except Exception as e:
            logger.error("Failed to read/merge emotion_linked (%s): %s", emo_link, e)
    else:
        logger.info("Skipping emotion summary: %s is empty or has no header", emo_link)

    if csv_has_header(body_link):
        try:
            body = pd.read_csv(body_link)
            geb = games.copy()
            if 'ge' in locals():
                geb = ge
            geb = geb.merge(body.groupby([id_col, task_col]).mean(numeric_only=True).reset_index(),
                            on=[id_col, task_col], how="left")
            geb.to_csv(os.path.join(out_datasets, "games_emotion_body_summary.csv"), index=False)
        except Exception as e:
            logger.error("Failed to read/merge body_linked (%s): %s", body_link, e)
    else:
        logger.info("Skipping body summary: %s is empty or has no header", body_link)

    # return paths based on configured outputs folder when available
    out_folder = out_datasets if 'out_datasets' in locals() else "outputs/datasets"
    return {
        "games": os.path.join(out_folder, "games_summary.csv"),
        "games_emotion": os.path.join(out_folder, "games_emotion_summary.csv"),
        "games_emotion_body": os.path.join(out_folder, "games_emotion_body_summary.csv"),
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
