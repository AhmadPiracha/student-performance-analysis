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

> `requirements.txt` includes all dependencies like `pandas`, `numpy`, `matplotlib`, `pyyaml`, `scipy`, `openpyxl`, `pypandoc`, etc.
> ⚠️ **Pandoc Note**: You do **not** need to manually install pandoc. The pipeline automatically downloads pandoc if it’s not available on your system.

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

1. Place your raw files in `data/raw/`:

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

### 📊 Processed Data (`data/processed/`)

* `performance_clean.csv`
* `emotion_clean.csv`
* `body_clean.csv`
* `preprocessing_report.csv` – missing values, duplicates, normalization
* `data_dictionary.xlsx` – column names, types, descriptions, allowed values
* `performance_kpis.csv` – accuracy, RT, error rate per task
* `performance_domain_summary.csv` – KPIs rolled up by EF domain (WM/IC/CF)
* `comparison_emotion_vs_baseline.csv` – with vs. without emotion (stats + CIs)
* `comparison_body_vs_baseline.csv` – with vs. without body (stats + CIs)
* `emotion_perf_corr.csv`, `body_perf_corr.csv` – correlation tables
* `integrated_ef_table.csv` – combined emotion+body+performance
* `ef_profiles.csv` – numeric EF profiles
* `student_profiles.xlsx` – structured per-student EF report
* `student_summaries.txt` – 5–7 line narrative summaries per student
* `early_indicators_report.csv` – ADHD-like early indicator flags

### 📂 Delivery Datasets (`outputs/datasets/`)

* `games_summary.csv`
* `games_emotion_summary.csv`
* `games_emotion_body_summary.csv`

### 📈 Visualizations (`outputs/figures/`)

* Per-student KPI bar charts
* Heatmaps of performance, emotion, and body metrics
* Correlation heatmaps
* Index file: `figures_index.txt`

### 📝 Reports (`outputs/reports/`)

* `EF_Report.md` – Markdown version
* `EF_Final_Report.pdf` – auto-generated PDF with tables, summaries, and embedded figures

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
| `scripts/report.py`         | Generates Markdown + PDF report with summaries, figures, appendices            |
| `scripts/run_all.py`        | Orchestrates all steps end-to-end                                              |

---

## Notes

* The pipeline is **config-driven**; update `config.yaml` rather than editing code.
* Visualizations use **matplotlib** only (no seaborn) and avoid specifying colors.
* Accuracy in percent is automatically normalized to 0–1.
* Timestamp-based merges use a tolerance window (`time_tolerance_seconds` in config).
* Pandoc is auto-downloaded if not found, so PDF generation works out of the box.

---

## ✅ Client Checklist Coverage

This section maps the client’s requirements to deliverables produced by the pipeline:

1. **Core Performance Analysis (Games)**

   * `performance_kpis.csv`, `performance_domain_summary.csv`, `games_summary_explained.xlsx`
   * Outlier handling (IQR) documented in `games_summary_explained.xlsx`

2. **Emotion Analysis (Emotion.csv)**

   * `emotion_features_by_task.csv`
   * `comparison_emotion_vs_baseline.csv` (means, deltas, p-values, effect sizes, CIs)
   * `emotion_perf_corr.csv` (correlation with KPIs)

3. **Body Motion Analysis (Body.csv)**

   * `body_features_by_task.csv`
   * `comparison_body_vs_baseline.csv` (means, deltas, p-values, effect sizes, CIs)
   * `body_perf_corr.csv` (correlation with KPIs)

4. **Integrated Analysis**

   * `integrated_ef_table.csv`
   * `ef_profiles.csv` (numeric EF profiles)
   * `student_profiles.xlsx` (structured per-student report)
   * `student_summaries.txt` (5–7 line narratives per student)
   * `early_indicators_report.csv` (ADHD-like early flags)

5. **Visualizations**

   * Heatmaps, KPI plots, correlation maps in `outputs/figures/`
   * Index in `figures_index.txt`
   * Embedded into `EF_Final_Report.pdf`

6. **Data Delivery (Datasets)**

   * `games_summary.csv`
   * `games_emotion_summary.csv`
   * `games_emotion_body_summary.csv`

7. **Python Scripts & Reproducibility**

   * Scripts: `01_preprocess.py`, `02_kpis_games.py`, `03_emotion_compare.py`, `04_body_compare.py`, `05_integrated_profiles.py`
   * `requirements.txt` (all dependencies)
   * `README.md` (run instructions, reproducibility notes)

8. **Student IDs**

   * All outputs use simplified `Student #X` labels
   * Internal mapping in `id_mapping_secure.csv`

9. **Appendices**

   * Preprocessing appendix: `preprocessing_report.csv`
   * Data dictionary appendix: `data_dictionary.xlsx` (with preview embedded in PDF)
   * Statistical appendix: all test outputs (p-values, effect sizes, confidence intervals)
   * Figures appendix: all heatmaps embedded in PDF
