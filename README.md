# EF Integrated Analysis Project

This project performs an integrated analysis of **Games/Performance**, **Emotion**, and **Body Motion** data to build **Executive Function (EF)** profiles per student.

## How to Use

1. Place your raw CSV files in `data/raw/`:
   - Performance: `M1.csv`..`M6.csv` (or edit `config.yaml`)
   - Emotion: `Emotion.csv`
   - Body: `Body.csv`

2. Open and edit `config.yaml` to map **your actual column names** to the standardized names used by the pipeline.

3. Run the full pipeline:
   ```bash
   python -m scripts.run_all --config config.yaml
   ```

4. Outputs:
   - Cleaned & standardized tables in `data/processed/`
   - Datasets for delivery in `outputs/datasets/`
     - `games_summary.csv`
     - `games_emotion_summary.csv`
     - `games_emotion_body_summary.csv`
   - Figures in `outputs/figures/`
   - Final report (Markdown) in `outputs/reports/EF_Report.md`

## Contents
- `scripts/preprocess.py` — Cleans and standardizes data (IDs, timestamps, missing values, duplicates)
- `scripts/performance.py` — Computes KPIs (accuracy, reaction time, error rate) and EF-domain summaries
- `scripts/emotion.py` — Emotion change metrics and linkage to performance (+ with/without comparisons)
- `scripts/body.py` — Motion metrics, linkage to performance, and comparisons
- `scripts/integrate.py` — Emotion + Body + Performance integration & EF profiles; early indicators
- `scripts/visualize.py` — Matplotlib charts and heatmaps
- `scripts/build_datasets.py` — Exports delivery datasets per spec
- `scripts/report.py` — Generates a Markdown report with summaries and embeds
- `scripts/run_all.py` — Orchestrates all steps end-to-end

## Notes
- The pipeline is **config-driven**; update `config.yaml` rather than editing code.
- Visualizations use **matplotlib** only (no seaborn) and avoid specifying colors.
- If your accuracy is in percent, the code normalizes it to 0–1 automatically.
- Timestamp-based merges use a tolerance window (`time_tolerance_seconds` in config).