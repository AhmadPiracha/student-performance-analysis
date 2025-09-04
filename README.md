# EF Integrated Analysis Project

This project performs an integrated analysis of **Games/Performance**, **Emotion**, and **Body Motion** data to build **Executive Function (EF)** profiles per student.

---

## Setup Python Environment

1. **Create a virtual environment** (recommended):

```bash
python -m venv ef_env
```

2. **Activate the environment**:

* **Windows**:

```bash
ef_env\Scripts\activate
```

* **Linux / macOS**:

```bash
source ef_env/bin/activate
```

3. **Install required packages**:

```bash
pip install -r requirements.txt
```

> Make sure `requirements.txt` includes all dependencies like `pandas`, `numpy`, `matplotlib`, `pyyaml`, etc.

4. **Run the pipeline inside the environment**:

```bash
python -m scripts.run_all --config config.yaml
```

5. **Deactivate the environment** when done:

```bash
deactivate
```

---

## How to Use

1. Place your raw CSV files in `data/raw/`:

   * Performance: `M1.csv` .. `M6.csv` (or edit `config.yaml`)
   * Emotion: `Emotion.csv`
   * Body: `Body.csv`

2. Open and edit `config.yaml` to map **your actual column names** to the standardized names used by the pipeline.

3. Run the full pipeline (inside the Python environment):

```bash
python -m scripts.run_all --config config.yaml
```

---

## Outputs

* **Cleaned & standardized tables** in `data/processed/`

  * `performance_clean.csv`
  * `emotion_clean.csv`
  * `body_clean.csv`

* **Datasets for delivery** in `outputs/datasets/`

  * `games_summary.csv`
  * `games_emotion_summary.csv`
  * `games_emotion_body_summary.csv`

* **Figures** in `outputs/figures/`

  * Per-student KPI bar charts
  * Heatmaps of performance, emotion, body metrics
  * Correlation heatmaps

* **Final report** (Markdown) in `outputs/reports/EF_Report.md`

---

## Project Structure

| File                        | Description                                                                    |
| --------------------------- | ------------------------------------------------------------------------------ |
| `scripts/preprocess.py`     | Cleans and standardizes data (IDs, timestamps, missing values, duplicates)     |
| `scripts/performance.py`    | Computes KPIs (accuracy, reaction time, error rate) and EF-domain summaries    |
| `scripts/emotion.py`        | Emotion change metrics and linkage to performance (+ with/without comparisons) |
| `scripts/body.py`           | Motion metrics, linkage to performance, and comparisons                        |
| `scripts/integrate.py`      | Emotion + Body + Performance integration & EF profiles; early indicators       |
| `scripts/visualize.py`      | Matplotlib charts and heatmaps                                                 |
| `scripts/build_datasets.py` | Exports delivery datasets per spec                                             |
| `scripts/report.py`         | Generates a Markdown report with summaries and embeds                          |
| `scripts/run_all.py`        | Orchestrates all steps end-to-end                                              |

---

## Notes

* The pipeline is **config-driven**; update `config.yaml` rather than editing code.
* Visualizations use **matplotlib** only (no seaborn) and avoid specifying colors.
* Accuracy in percent is automatically normalized to 0–1.
* Timestamp-based merges use a tolerance window (`time_tolerance_seconds` in config).


I can also **create a concise “Client Delivery Instructions” section** that explains *exactly which files to send to the client*, so they can open it directly and understand the results. Do you want me to add that too?
