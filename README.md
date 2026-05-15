# NASA Exoplanet Research & Machine Learning Analysis

A comprehensive exploratory data analysis and machine learning investigation of the NASA Exoplanet Archive dataset using Python, statistical modelling, dimensionality reduction, clustering, anomaly detection, and explainable AI techniques.

---

## Project Overview

This project analyses thousands of confirmed exoplanets from the NASA Exoplanet Archive to investigate:

* Planetary discovery trends over time
* Missing data structures and observational bias
* Scientific data validation and archival refinement
* Imputation strategy optimisation
* Statistical relationships between stellar and planetary properties
* Exoplanet population structure using unsupervised learning
* Predictive modelling of planetary equilibrium temperature
* Explainable AI feature interpretation using SHAP
* Anomaly detection for identifying unusual exoplanet systems

This project demonstrates an end-to-end data science workflow applied to astrophysical research data, integrating data engineering, exploratory analysis, machine learning, dimensionality reduction, anomaly detection, and explainable AI into a reproducible analytical pipeline.

---

# Key Features

## Data Engineering & Cleaning

* Automated dataset auditing and forensic analysis
* Missing value quantification and visualisation
* Scientific validation rules for astrophysical realism
* Duplicate archival consolidation
* Dynamic preprocessing pipeline selection

## Statistical Analysis

* Mean, median, and standard deviation analysis
* Correlation analysis across astrophysical variables
* Discovery-era trend analysis

## Data Visualisation

* High-quality Matplotlib and Seaborn visualisations
* Multi-variable scientific scatterplots
* PCA and t-SNE projections
* Cluster visualisation and anomaly mapping

## Machine Learning

* Adaptive KNN imputation optimisation
* Iterative and median imputation benchmarking
* XGBoost regression modelling
* Feature importance analysis
* SHAP explainability framework
* Isolation Forest anomaly detection

## Research Reporting

* Automated HTML research report generation
* Embedded plots and analytical commentary
* Styled tables and navigation system

---

# Technologies Used

| Category         | Libraries                            |
| ---------------- | ------------------------------------ |
| Data Processing  | `pandas`, `numpy`                    |
| Visualisation    | `matplotlib`, `seaborn`, `missingno` |
| Machine Learning | `scikit-learn`, `xgboost`            |
| Explainable AI   | `shap`                               |
| Utility          | `os`, `datetime`, `webbrowser`       |

---

# Machine Learning Pipeline

## 1. Dataset Forensics

* Missingness analysis
* Duplicate structure analysis
* Observational bias inspection
* Scientific validation filtering

## 2. Imputation Benchmarking

Three imputation strategies were benchmarked:

* Median Imputation
* Adaptive KNN Imputation
* Iterative Imputation

Evaluation criteria included:

* Reconstruction error
* Correlation preservation
* Downstream predictive performance

A weighted composite scoring system automatically selected the optimal imputation method.

---

## 3. Dimensionality Reduction

### PCA

Principal Component Analysis was used to identify dominant variance structures in the exoplanet feature space.

### t-SNE

t-SNE was applied to investigate potential nonlinear clustering structures within planetary populations.

---

## 4. Clustering & Anomaly Detection

### K-Means Clustering

Exoplanets were grouped into population clusters based on astrophysical similarity.

### Isolation Forest

Rare and potentially unusual planetary systems were identified using anomaly detection techniques.

---

## 5. Predictive Modelling

An `XGBoostRegressor` model was trained to predict planetary equilibrium temperature using:

* Stellar temperature
* Orbital radius
* Planetary mass
* Orbital eccentricity

Model performance was evaluated using:

* R² Score
* Mean Absolute Error (MAE)

---

## 6. Explainable AI

SHAP analysis was used to:

* Quantify feature importance
* Interpret model decision behaviour
* Visualise positive and negative feature contributions

---

# Example Outputs

The project automatically generates:

* Missing data heatmaps
* Correlation heatmaps
* PCA projections
* t-SNE embeddings
* Cluster visualisations
* SHAP summary plots
* Feature importance charts
* Automated HTML research reports

---

# Repository Structure

```text
NASA-Exoplanet-Research-FOSE1030/
│
├── nasa_exoplanet_archive.csv
├── ashley_selvin.py
├── README.md
│
├── research_report/
│   ├── exoplanet_report.html
│   └── figures/
│       ├── anomaly_detection.png
│       ├── clustering_projection.png
│       ├── correlation_heatmap.png
│       ├── imputation_mse.png
│       ├── knn_optimisation.png
│       ├── missingness_matrix.png
│       ├── pca_projection.png
│       ├── shap_summary.png
│       ├── tsne_projection.png
│       ├── xgb_importance.png
│       └── yearly_completeness.png
```

---

# Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Ashley-Selvin/NASA-Exoplanet-Research-FOSE1030.git
cd NASA-Exoplanet-Research-FOSE1030
```

## 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn missingno scikit-learn xgboost shap
```

## 3. Run the Script

```bash
python ashley_selvin.py
```

---

# Research Questions Explored

* How has exoplanet discovery evolved over time?
* Which discovery methods produce the most complete datasets?
* Which imputation strategy best preserves astrophysical structure?
* Can machine learning accurately estimate planetary equilibrium temperature?
* Do exoplanets naturally form distinguishable clusters?
* Can anomalous planetary systems be identified automatically?

---

# Skills Demonstrated

## Data Science

* Exploratory Data Analysis (EDA)
* Feature engineering
* Statistical analysis
* Data cleaning pipelines

## Machine Learning

* Regression modelling
* Hyperparameter optimisation
* Clustering
* Dimensionality reduction
* Anomaly detection

## AI & Explainability

* SHAP interpretability
* Model diagnostics
* Feature attribution analysis

## Software Engineering

* Modular analytical pipeline design
* Automated reporting systems
* Reproducible research workflows

---

# Future Improvements

Potential future extensions include:

* Deep learning models for planetary classification
* Time-series analysis of archival updates
* Bayesian uncertainty modelling
* Interactive dashboards using Plotly or Streamlit
* Automated astrophysical anomaly reporting
* Scientific paper-style PDF export pipeline

---

# Data Source

Dataset derived from the NASA Exoplanet Archive operated by the California Institute of Technology under contract with NASA:

[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/?utm_source=chatgpt.com)

---

# Author

Ashley Selvin
Economics & Business Analytics Student
Machine Learning | Data Analytics | Quantitative Research | AI Applications
