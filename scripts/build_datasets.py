
import os, yaml, pandas as pd

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

    games = pd.read_csv(perf_kpis) if os.path.exists(perf_kpis) else pd.DataFrame()
    games.to_csv(os.path.join(out_datasets, "games_summary.csv"), index=False)

    if os.path.exists(emo_link):
        emo = pd.read_csv(emo_link)
        ge = games.merge(emo.groupby([id_col, task_col]).mean(numeric_only=True).reset_index(),
                         on=[id_col, task_col], how="left")
        ge.to_csv(os.path.join(out_datasets, "games_emotion_summary.csv"), index=False)

    if os.path.exists(body_link):
        body = pd.read_csv(body_link)
        geb = games.copy()
        if 'ge' in locals():
            geb = ge
        geb = geb.merge(body.groupby([id_col, task_col]).mean(numeric_only=True).reset_index(),
                        on=[id_col, task_col], how="left")
        geb.to_csv(os.path.join(out_datasets, "games_emotion_body_summary.csv"), index=False)

    return {"games": "outputs/datasets/games_summary.csv",
            "games_emotion": "outputs/datasets/games_emotion_summary.csv",
            "games_emotion_body": "outputs/datasets/games_emotion_body_summary.csv"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(args.config))
