# Sierra — Snow Cover Forecasting (NARX + LSTM)

## Overview

Sierra is a research-grade toolkit to predict basin-scale snow-covered area and pixel-level snow probability from MODIS HDF satellite data and exogenous meteorological covariates. It provides end-to-end reproducible workflows for HDF processing, data cleaning & imputation, Optuna hyperparameter tuning, LSTM-based NARX modeling, and rich visual outputs (heatmaps and line plots) so reviewers can quickly evaluate model behavior and results.

> Predict snow-covered area and pixel-level snow probability using a NARX model built with LSTM layers and MODIS satellite inputs. Ideal for hydrology research and for showcasing interpretable ML models to recruiters.

---

## 🚀 Highlights

- **Problem:** Predict the snow-covered area per basin and per-pixel snow probability from MODIS HDF satellite files and exogenous covariates.
- **Model:** NARX (Nonlinear AutoRegressive with eXogenous inputs) implemented using LSTM-based RNNs.
- **Key features:** automated data cleaning & imputation, hyperparameter tuning with Optuna, GPU-ready training (TensorFlow 2.10), and visual outputs (heatmaps, per-basin line plots).
- **Evaluation:** Models are tuned using the **Nash–Sutcliffe Efficiency (NSE)** and output per-basin metrics in JSON.

---

## 🧭 Quick Start (short)

1. Clone the repo:

```bash
git clone <your-repo-url>
cd sierra-cc-master
```

2. Create the recommended environments:

```bash
# TensorFlow GPU workflows (training/evaluation)
conda env create -f tf210_gpu.yml
conda activate tf210_gpu

# HDF/heatmap processing
conda env create -f environment-hdf.yml
conda activate environment-hdf
```

3. Run example scripts:

```bash
python limpieza_datos.py            # data cleaning & CSV generation
python models/best_params.py        # Optuna hyperparameter search (NSE)
python models/create_load_model.py  # build or evaluate a model
python models/predictions.py        # generate predictions and plots
python heatmaps.py                  # create pixel-level snow probability maps
```

> Tip: Many scripts ask interactively for a basin name or scenario. See `images/ejemplo-ficheros.png` for the expected outputs.

---

## Data acquisition & preprocessing

Raw MODIS HDF files were obtained from NASA EarthData Search (https://search.earthdata.nasa.gov/search). Files were custom-downloaded for each basin, reprojected to geographic lat/lon, and processed to produce per-basin CSV summaries and per-pixel probability maps.

- See `datasets/` for raw CSV extracts and `datasets_imputed/` for cleaned datasets used for modeling.
- Reprojection and HDF handling are supported by `heatmaps.py`.


## Model & approach

- **Architecture:** NARX implemented with LSTM-based RNNs to model snow-area dynamics using exogenous predictors (meteorological covariates and pixel-level probabilities).
- **Tuning:** Hyperparameter search uses **Optuna** and optimizes the **Nash–Sutcliffe Efficiency (NSE)**.
- **Outputs:** per-basin models (`models/*.h5`), `metrics.json` files with metrics and hyperparameters, prediction CSVs, and visualization plots.
- **Typical workflow:** prepare HDFs → reproject & extract pixel-level probabilities → compute per-basin snow area → assemble datasets with exogenous covariates → clean & impute → tune (Optuna) → train NARX-LSTM → evaluate (NSE) → generate predictions and visualizations.

## Technologies & workflow

- **Languages & libraries:** Python (pandas, numpy, xarray), TensorFlow 2.10 (GPU-ready), Optuna, rasterio/GDAL (HDF & reprojection), scikit-learn, matplotlib/seaborn, jupyter (optional).
- **Reproducibility:** environment ymls (`tf210_gpu.yml`, `environment-hdf.yml`) pin key packages and make it easy for reviewers to reproduce experiments.
- **Data:** MODIS HDF files (NASA EarthData) — scripts include reprojection and per-pixel probability computation (`heatmaps.py`).
- **Execution notes:** GPU is recommended for training; many scripts are interactive but can be scripted for batch runs.


## Workflow

A short, practical workflow that describes how raw HDFs become model-ready datasets, models and visual reports:

1. **Data acquisition (HDFs)** — Download MODIS HDF files per basin from NASA EarthData; store raw files on the external disk (expected `data/` layout).
2. **HDF processing & reprojection** — Run `heatmaps.py` to reproject HDFs to lat/lon and compute pixel-level snow probability maps (GeoTIFFs / PNGs).
3. **Basin aggregation** — Use `limpieza_datos.py` → `process_basin()` to compute per-basin snow-area timeseries from processed pixel data; outputs saved to `datasets/`.
4. **Exogenous variables & scenarios** — Use `process_var_exog()` and `cleaning_future_series()` to prepare meteorological covariates and scenario CSVs (20 scenario/model combinations).
5. **ETL / dataset assembly** — `join_area_exog()` merges area timeseries and covariates into model-ready CSVs; imputed datasets are stored under `datasets_imputed/`.
6. **Data cleaning & imputation** — Apply `impute_outliers()` (IQR-based) and manual checks to guarantee training-quality inputs.
7. **Hyperparameter tuning & training** — Run `models/best_params.py` (Optuna) to find best hyperparameters, then `models/create_load_model.py` to train or evaluate models (GPU recommended).
8. **Predictions & visualization** — Generate model predictions with `models/predictions.py` and produce plots and CSVs; final visual outputs are in `images/` and `images/heatmaps/`.
9. **Review & iterate** — Inspect `models/*/metrics.json`, plots and heatmaps, refine preprocessing or model configuration and repeat.

> Tip: wrap interactive scripts with small wrappers or flags to enable repeatable, non-interactive runs for CI or demos.


## Project layout (concise)

- `datasets/` — raw and aggregated CSVs
- `datasets_imputed/` — cleaned & imputed datasets
- `images/` — all visual outputs (flowchart, heatmaps, lineplots)
- `models/` — scripts and model artifacts
- `limpieza_datos.py` — cleaning and imputation utilities (see functions below)
- `heatmaps.py` — HDF to probability map tools
- `tf210_gpu.yml` / `environment-hdf.yml` — conda environments


## Useful scripts (what each does)

- `limpieza_datos.py` — data cleaning helpers
  - `process_basin(basin)` — compute basin area and snow area timeseries from HDFs
  - `process_var_exog(...)` — convert aggregated Excel to per-basin CSV of exogenous covariates
  - `cleaning_future_series(...)` — produce exogenous series for future scenarios (20 CSVs total)
  - `join_area_exog(...)` — join area and exogenous variables to build model datasets
  - `impute_outliers(...)` — IQR-based outlier imputation

- `models/best_params.py` — Optuna hyperparameter search (interactive)
- `models/create_load_model.py` — build or evaluate a model using discovered hyperparameters
- `models/predictions.py` — generate per-basin predictions and save CSV + plots
- `heatmaps.py` — generate pixel-wise probability heatmaps from HDF files


## Results & where to look

- Visuals: `images/flowchart.png`, `images/heatmaps/*`, `images/lineplots/*`
- Trained artifacts and metrics: `models/*/` (look for `metrics.json` and `.h5` files)

## Contact

- **Kiké Merino (Author)** — LinkedIn: https://www.linkedin.com/in/kikemerino/


## Contributing

- Open issues or PRs for improvements. Consider adding automated tests and a `LICENSE` file (MIT / Apache-2.0 recommended).


## Known limitations

- Several scripts are interactive and assume local disk structure for raw HDF files; include README notes in the future to automate example runs.


---

Thanks for looking — contact the repo owner for a short demo or walkthrough GIF if desired ✨
