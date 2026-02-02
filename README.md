# Sierra – Snow Cover Forecasting with LSTM (Remote Sensing & Time Series ML)

## 🔎 Project Overview

**Sierra** is an end-to-end **Machine Learning project** focused on **time series forecasting of snow-covered area** at basin scale using **satellite imagery and meteorological data**.

The project combines **remote sensing (MODIS HDF data)** with **deep learning (LSTM-based NARX models)** to predict snow dynamics, generating both **basin-level forecasts** and **pixel-level snow probability maps**.  
It is designed to be **reproducible, interpretable and production-oriented**, making it suitable for both research and applied ML roles.

---

## 🎯 Problem & Motivation

Accurate snow cover forecasting is critical for:
- Hydrology and water resource management
- Climate analysis
- Environmental risk assessment

Traditional statistical approaches struggle to model the **non-linear temporal dynamics** of snow evolution.  
This project addresses the problem using **LSTM-based NARX models** that integrate historical snow cover and **exogenous meteorological variables**.

---

## 🧠 Machine Learning Approach

- **Task:** Time series forecasting (regression)  
- **Model:** NARX (Nonlinear AutoRegressive with eXogenous inputs) implemented with **LSTM layers**
- **Inputs:**
  - Historical snow-covered area
  - Meteorological exogenous variables
  - Pixel-level snow probabilities from MODIS data
- **Evaluation metric:** **Nash–Sutcliffe Efficiency (NSE)**
- **Hyperparameter tuning:** Automated with **Optuna**

---

## 🗺️ Data & Feature Engineering

- **Source:** MODIS satellite HDF files (NASA EarthData)
- **Processing:**
  - HDF reprojection to geographic coordinates
  - Pixel-level snow probability computation
  - Basin-level aggregation of snow-covered area
- **ETL pipeline:**
  - Data cleaning and IQR-based outlier imputation
  - Dataset assembly combining area and exogenous variables
  - Generation of historical and future scenario datasets

---

## 📊 Outputs & Results

- **Per-basin LSTM models** (`.h5`)
- **Metrics per basin** stored in `metrics.json`
- **Time series predictions** with uncertainty-aware evaluation
- **Visual outputs:**
  - Basin-level line plots
  - Pixel-level snow probability heatmaps

All outputs are designed to allow **quick qualitative and quantitative evaluation** of model performance.

---

## 🛠️ Technologies Used

- **Python:** pandas, numpy, xarray
- **Deep Learning:** TensorFlow 2.10 (GPU-ready), LSTM
- **Optimization:** Optuna
- **Geospatial:** rasterio, GDAL
- **ML utilities:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Reproducibility:** Conda environments (`tf210_gpu.yml`, `environment-hdf.yml`)

---

## 🚀 Reproducibility & Execution

The project provides pinned environments and modular scripts to ensure full reproducibility.

Typical workflow:
1. HDF processing & reprojection
2. Basin-level aggregation
3. Dataset assembly & cleaning
4. Hyperparameter tuning (Optuna)
5. Model training & evaluation
6. Prediction & visualization

GPU is recommended for training but not required for inference or analysis.

---

## 👤 Author

**Kike Merino**  
Data Scientist | Machine Learning | Time Series | Deep Learning  
LinkedIn: https://www.linkedin.com/in/kikemerino/
