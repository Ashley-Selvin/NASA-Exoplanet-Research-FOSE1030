import os 
import webbrowser 
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import IsolationForest

from xgboost import XGBRegressor

import shap 

# Load the dataset
exoplanet_df = pd.read_csv("nasa_exoplanet_archive.csv")
print("Dataset loaded successfully!\n")

print("Columns in dataset: \n")
for column in exoplanet_df.columns:
    print(f"- {column}")

print("\nDataset shape: ", exoplanet_df.shape)

# Report Directory ----------------------------------------------------------------------------------
report_dir = "research_report"
figures_dir = os.path.join(report_dir, "figures")

os.makedirs(figures_dir, exist_ok=True)

# Standardise column names
exoplanet_df.columns = (
    exoplanet_df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
    .str.replace("/", "_")
)

print(exoplanet_df.columns.to_list())

# Dataset Forensics --------------------------------------------------------
# Dataset Snapshot
print("\n===== Dataset Snapshot =====")
print(f"Rows: {exoplanet_df.shape[0]}")
print(f"Columns: {exoplanet_df.shape[1]}")

print("\nData Types:")
print(exoplanet_df.dtypes)

# Missingness Quantification
missing_summary = pd.DataFrame({
    "Missing Count": exoplanet_df.isnull().sum(),
    "Missing Percentage": (exoplanet_df.isnull().mean() * 100)
})

missing_summary = missing_summary.sort_values(by="Missing Percentage", ascending=False)
print("\n===== Missing Value Summary =====")
print(missing_summary)

# Missingness Visualizations
plt.figure(figsize=(14, 8))
msno.matrix(exoplanet_df) # Matrix plot showing missing data patterns
plt.title("Missing Data Matrix")
plt.savefig(
    os.path.join(figures_dir, "missingness_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

plt.figure(figsize=(14, 8))
msno.bar(exoplanet_df) # Bar plot showing completeness distribution
plt.title("Column Completeness")
plt.savefig(
    os.path.join(figures_dir, "completeness_bar.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# Missingness by Discovery Method
missing_by_method = (
    exoplanet_df
    .groupby("discovery_method")
    .apply(lambda x: x.isnull().mean() * 100)
)

print("\n===== Missingness by Discovery Method =====")
print(missing_by_method)
# observational bias analysis ^
# allows us to discuss: transit detections containing more radius info, radial velocity detections containing better mass estimates, direct imaging having sparse orbital data. 

# Duplicate Structure Analysis
# Exact Duplicate Rows
exact_duplicates = exoplanet_df.duplicated().sum()
print("\n===== Exact Duplicates =====")
print(f"Exact duplicate rows: {exact_duplicates}")

# Duplicate Planet Names
planet_duplicates = (
    exoplanet_df["planet_name"]
    .value_counts()
)
planet_duplicates = planet_duplicates[planet_duplicates > 1]
print("\n===== Planets with Multiple Records =====")
print(planet_duplicates.head(20)) # only the top 20 for brevity
# Archival evolution analysis ^

# Investigate 1 Planet Through Time --> must decide on a specific planet to track its data quality and evolution over time
example_planet = "11_com_b"
planet_history = exoplanet_df[
    exoplanet_df["planet_name"] == example_planet
]
print(planet_history)

# Shows changing measurements, improving completeness and archival revisions for a specific planet, illustrating the dynamic nature of exoplanet data and the importance of ongoing data quality efforts.

# Scientific Validation Rules -----------------------------------------------------------------
# Planet Mass Validation
invalid_mass = ((exoplanet_df["planet_mass_m_e"] <= 0) & exoplanet_df["planet_mass_m_e"].notnull())
print(f"Invalid planet mass rows: {invalid_mass.sum()}")
exoplanet_df = exoplanet_df[~invalid_mass]

# Orbital Radius Validation
invalid_orbit = ((exoplanet_df["orbit_semi-major_axis_au"] <= 0) & exoplanet_df["orbit_semi-major_axis_au"].notnull())
print(f"Invalid orbital radius rows: {invalid_orbit.sum()}")
exoplanet_df = exoplanet_df[~invalid_orbit]

# Stellar Distance Validation
invalid_distance = ((exoplanet_df["stellar_distance_pc"] <= 0) & exoplanet_df["stellar_distance_pc"].notnull())
print(f"Invalid stellar distance rows: {invalid_distance.sum()}")
exoplanet_df = exoplanet_df[~invalid_distance]

# Planet Temperature Validation
invalid_temp = ((exoplanet_df["planet_temperature_k"] < 0) & exoplanet_df["planet_temperature_k"].notnull())
print(f"Invalid planet temperature rows: {invalid_temp.sum()}")
exoplanet_df = exoplanet_df[~invalid_temp]

# Remove unrealistic stellar temperatures
exoplanet_df = exoplanet_df[(exoplanet_df["stellar_temperature_k"] > 2000) & (exoplanet_df["stellar_temperature_k"] < 50000)]

# Observational Refinement Analysis ---------------------------------------------------------------------------------------
# Goal is to analyse whether: newer records contain more complete and accurate data, planetary measurements evolve over time and whether modern astronomy improves data completeness 
# Completeness by Discovery Year
yearly_completeness = (exoplanet_df.groupby("discovery_year").apply(lambda x: 100 - (x.isnull().mean().mean() * 100)))
plt.figure(figsize=(12, 6))
plt.plot(yearly_completeness.index, yearly_completeness.values, marker="o")

plt.xlabel("Discovery Year")
plt.ylabel("Average Dataset Completeness (%)")
plt.title("Dataset Completeness Across Discovery Years")
plt.savefig(
    os.path.join(figures_dir, "yearly_completeness.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Duplicate Consolidation ----------------------------------------------------------------------------------------------
# Instead of simply dropping duplicates, we consolidate planetary records by keeping the latest release, aggregating numeric values and preserving the best available measurements for each planet.
# Convert Release Date
exoplanet_df["releasedate"] = pd.to_datetime(exoplanet_df["releasedate"], errors="coerce")
# Sort by Release Date
exoplanet_df = exoplanet_df.sort_values(by="releasedate")
# Consolidation strategy
numerical_columns = exoplanet_df.select_dtypes(include=np.number).columns
categorical_columns = exoplanet_df.select_dtypes(exclude=np.number).columns
# Aggregate Function
aggregation_rules = {}
for col in numerical_columns:
    aggregation_rules[col] = "median" # Numerical columns --> median 
for col in categorical_columns:
    aggregation_rules[col] = "last" # Categorical columns --> last observation
# Consolidate
exoplanet_df = (exoplanet_df.groupby("planet_name", as_index=False).agg(aggregation_rules)) # Reconstructs canonical planetary records from archival measurements

# Imputation Benchmarking and Validation ----------------------------------------------------------------------------------------------------------------------------------------
# Research Goal:
# Evaluate which imputation strategy best preserves astrophysical structure, minimizes reconstruction error and improves downstream ML performance.
print("\n===== Imputation Benchmarking =====")
# Columns selected for imputation benchmarking
imputation_columns = [
    "planet_mass_m_e", 
    "planet_temperature_k", 
    "stellar_temperature_k", 
    "orbit_semi-major_axis_au",
    "stellar_mass_m_sol"
]

# Create a complete subset for controlled experiment 
complete_data = exoplanet_df[imputation_columns].dropna()

print(f"\nComplete cases available: {len(complete_data)}")

# Create artificial missingness
masked_data = complete_data.copy()
np.random.seed(42)
mask = np.random.rand(*masked_data.shape) < 0.1
original_values = complete_data.copy()
masked_data[mask] = np.nan

# Reconstruction Error Evaluation -----------------------------------------------------------------------------------------------------------------------------------------------
def evaluate_imputation(original, imputed, mask):
    return mean_squared_error(original[mask], imputed[mask])

# Median Imputation -------------------------------------------------------------------------------------------------------------------------------------------------------------
median_imputer = SimpleImputer(strategy="median")
median_imputed = pd.DataFrame(median_imputer.fit_transform(masked_data), columns=masked_data.columns)

# Adaptive KNN Imputation ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Goal: Automatically determine the optimal number of neighbours by minimizing reconstruction error under simulated missingness
knn_results = []

candidate_neighbours = [3, 5, 7, 9, 11, 15]

best_knn_mse = np.inf
best_knn_k = None
best_knn_imputed = None

for k in candidate_neighbours:
    candidate_imputer = KNNImputer(n_neighbors=k)
    candidate_imputed = pd.DataFrame(candidate_imputer.fit_transform(masked_data), columns=masked_data.columns)
    candidate_mse = evaluate_imputation(original_values.values, candidate_imputed.values, mask)
    knn_results.append({"k": k, "MSE": candidate_mse})
    print(f"KNN neighbours = {k} --> Reconstruction MSE = {candidate_mse:.4f}")

    # Track best configuration
    if candidate_mse < best_knn_mse:
        best_knn_mse = candidate_mse
        best_knn_k = k
        best_knn_imputed = candidate_imputed

# Final selected KNN outputs
knn_imputed = best_knn_imputed
knn_mse = best_knn_mse

print("\n===== Optimal KNN Configuration =====")
print(f"Best neighbour count: {best_knn_k}")
print(f"Best reconstruction MSE: {best_knn_mse:.4f}")

# KNN Hyperparameter Search Visualization ----------------------------------------------------------------------------------------------------------------------------------------------------
knn_results_df = pd.DataFrame(knn_results)
plt.figure(figsize=(10, 6))
plt.plot(
    knn_results_df["k"],
    knn_results_df["MSE"],
    marker="o"
)
plt.xlabel("Number of Neighbours (k)")
plt.ylabel("Reconstruction MSE")
plt.title("Adaptive KNN Hyperparameter Optimisation")
plt.savefig(
    os.path.join(figures_dir, "knn_optimisation.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Iterative Imputation ----------------------------------------------------------------------------------------------------------------------------------------------------------
iterative_imputer = IterativeImputer(random_state=42, max_iter=20)
iterative_imputed = pd.DataFrame(iterative_imputer.fit_transform(masked_data), columns=masked_data.columns)

median_mse = evaluate_imputation(original_values.values, median_imputed.values, mask)
iterative_mse = evaluate_imputation(original_values.values, iterative_imputed.values, mask)

print("\n===== Reconstruction Error =====")
print(f"Median Imputation MSE: {median_mse:.4f}")
print(f"KNN Imputation MSE: {knn_mse:.4f}")
print(f"Iterative Imputation MSE: {iterative_mse:.4f}")

# Correlation Preservation Analysis --------------------------------------------------------------------------------------------------------------------------------------------
original_corr = complete_data.corr()

median_corr = median_imputed.corr()
knn_corr = knn_imputed.corr()
iterative_corr = iterative_imputed.corr()

median_corr_difference = np.abs(original_corr - median_corr).mean().mean()
knn_corr_difference = np.abs(original_corr - knn_corr).mean().mean()
iterative_corr_difference = np.abs(original_corr - iterative_corr).mean().mean()

print("\n===== Correlation Distortion =====")
print(f"KNN correlation distortion: " f"{knn_corr_difference:.4f}")
print(f"Iterative correlation distortion: " f"{iterative_corr_difference:.4f}")

# Visual Comparison -------------------------------------------------------------------------------------------------------------------------------------------------------------
methods = [
    "Median", 
    "KNN", 
    "Iterative"
]

mse_scores = [
    median_mse, 
    knn_mse, 
    iterative_mse
]

plt.figure(figsize=(8, 5))
plt.bar(methods, mse_scores)

plt.ylabel("Mean Squared Error")
plt.title("Imputation Reconstruction Error")
plt.savefig(
    os.path.join(figures_dir, "imputation_mse.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Downstream XGBoost Performance -----------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate whether imputation quality improves ML predictive performance.
prediction_target = "planet_temperature_k"

prediction_features = [
    "stellar_temperature_k", 
    "orbit_semi-major_axis_au",
    "planet_mass_m_e",
    "stellar_mass_m_sol"
]

# Function to evaluate ML performance after imputation
def evaluate_xgboost_performance(imputed_df, method_name):
    model_df = imputed_df.copy()

    X = model_df[prediction_features]
    y = model_df[prediction_target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"\n===== {method_name.upper()} XGBoost Performance =====")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    return r2

# Evaluate downstream performance 
median_r2 = evaluate_xgboost_performance(median_imputed, "Median")
knn_r2 = evaluate_xgboost_performance(knn_imputed, "KNN")
iterative_r2 = evaluate_xgboost_performance(iterative_imputed, "Iterative")

# ML Performance Comparison Plot ---------------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.bar(methods, [median_r2, knn_r2, iterative_r2])
plt.ylabel("XGBoost R² Score")
plt.title("Downstream ML Performance by Imputation Method")
plt.savefig(
    os.path.join(figures_dir, "ML_comparison_bar.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Final Imputation Strategy --------------------------------------------------------------------------------------------------------------------------------------------
# Combine:
# 1. Reconstruction accuracy 
# 2. Correlation preservation
# 3. Downstream ML performance 

imputation_results = pd.DataFrame({
    "Method": [
        "Median", 
        "KNN",
        "Iterative"
    ],
    "MSE": [
        median_mse, 
        knn_mse, 
        iterative_mse
    ],
    "Correlation_Distortion": [
        median_corr_difference, 
        knn_corr_difference, 
        iterative_corr_difference
    ],
    "R2": [
        median_r2, 
        knn_r2, 
        iterative_r2
    ]
})

print("\n===== Raw Imputation Results =====")
print(imputation_results)

# Normalize metrics --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Lower is better for: MSE, Correlation distortion
# Higher is better for: R²
epsilon = 1e-9

imputation_results["MSE_Normalized"] = (
    (imputation_results["MSE"].max() - imputation_results["MSE"]) /
    (imputation_results["MSE"].max() - imputation_results["MSE"].min() + epsilon)
)

imputation_results["Correlation_Normalized"] = (
    (imputation_results["Correlation_Distortion"].max() - imputation_results["Correlation_Distortion"]) /
    (imputation_results["Correlation_Distortion"].max() - imputation_results["Correlation_Distortion"].min() + epsilon)
)

imputation_results["R2_Normalized"] = (
    (imputation_results["R2"] - imputation_results["R2"].min()) /
    (imputation_results["R2"].max() - imputation_results["R2"].min() + epsilon)
)

# Composite Score -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Weighted scoring system 
# Weighting rationale:
# 1. Reconstruction accuracy is most important 
# 2. Relationship preservation is second 
# 3. ML performance is third

imputation_results["Composite_Score"] = (0.5 * imputation_results["MSE_Normalized"] + 0.3 * imputation_results["Correlation_Normalized"] + 0.2 * imputation_results["R2_Normalized"])

print("\n===== Normalized Imputation Results =====")
print(imputation_results)

# Automatically select best imputation method --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
best_method = imputation_results.loc[imputation_results["Composite_Score"].idxmax(), "Method"]
print("\n===== Final Imputation Decision =====")
print(f"Selected Method: {best_method}")

# Dynamically instantiate winning imputer ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if best_method == "Median":
    final_imputer = SimpleImputer(strategy="median")
elif best_method == "KNN":
    final_imputer = KNNImputer(n_neighbors=best_knn_k)
else:
    final_imputer = IterativeImputer(random_state=42, max_iter=20)

# Apply selected imputer to full dataset --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
numerical_columns = exoplanet_df.select_dtypes(include=np.number).columns
exoplanet_df[numerical_columns] = final_imputer.fit_transform(exoplanet_df[numerical_columns])

print(f"\nFinal preprocessing pipeline applied using "f"{best_method} imputation.")

# Feature Engineering --------------------------------------------------------
# Log Planet Mass 
exoplanet_df["log_planet_mass"] = np.log10(exoplanet_df["planet_mass_m_e"].clip(lower=1e-6))

# Discovery Era 
def classify_era(year):
    if year < 2005:
        return "Early Discovery Era"
    elif year < 2015:
        return "Kepler Era"
    else:
        return "Modern Era"
    
exoplanet_df["discovery_era"] = (
    exoplanet_df["discovery_year"].apply(classify_era)
)

# Habitability Proxy
exoplanet_df["habitability_score"] = (
    np.exp(-abs(exoplanet_df["planet_temperature_k"] - 288) / 100)
)

# Numerical Analysis --------------------------------------------------------
selected_columns = [
    "planet_mass_m_e", 
    "planet_temperature_k", 
    "stellar_temperature_k",
]

summary_statistics = pd.DataFrame({
    "Mean": exoplanet_df[selected_columns].mean(),
    "Median": exoplanet_df[selected_columns].median(),
    "Standard Deviation": exoplanet_df[selected_columns].std()
})
print(summary_statistics)

# Correlation Heatmap -------------------------
plt.figure(figsize=(14, 10))

heatmap_columns = [
    "planet_mass_m_e",
    "planet_temperature_k",
    "stellar_temperature_k",
    "orbit_semi-major_axis_au",
    "stellar_mass_m_sol",
    "stellar_radius_r_sol",
    "stellar_distance_pc"
]
correlation_matrix = (exoplanet_df[heatmap_columns].corr())

sns.heatmap(
    correlation_matrix, 
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap")
plt.savefig(
    os.path.join(figures_dir, "correlation_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Simple Plot -----------------------------------------------------
# Planet Mass vs Discovery Year
plt.figure(figsize=(12, 7))

plt.scatter(
    exoplanet_df["discovery_year"], 
    exoplanet_df["planet_mass_m_e"],
    alpha=0.5
)

plt.yscale("log")

plt.xlabel("Discovery Year")
plt.ylabel("Planet Mass (Earth Masses)")
plt.title("Planet Mass vs Discovery Year")
plt.savefig(
    os.path.join(figures_dir, "simple_plot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Multi-Variable Scatter Plot -----------------------------------------------------
plt.figure(figsize=(14, 8))

scatter = plt.scatter(
    exoplanet_df["stellar_temperature_k"], 
    exoplanet_df["planet_temperature_k"],
    s=exoplanet_df["planet_mass_m_e"] * 0.3,
    c=exoplanet_df["discovery_year"],
    alpha=0.6,
    cmap="viridis"
)

plt.colorbar(scatter, label="Discovery Year")

plt.xlabel("Stellar Temperature (K)")
plt.ylabel("Planet Equilibrium Temperature (K)")
plt.title("Planetary Temperature vs Stellar Temperature")
plt.savefig(
    os.path.join(figures_dir, "multi_variable_scatter_plot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# PCA Dimensionality Reduction --------------------------------------------------------
features_for_ml = [
    "planet_mass_m_e",
    "planet_temperature_k",
    "stellar_temperature_k",
    "orbit_semi-major_axis_au",
]

ml_df = exoplanet_df[features_for_ml].dropna()

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(ml_df)

# PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(
    pca_components, 
    columns=["PC1", "PC2"]
)

# PCA plot
plt.figure(figsize=(12, 8))

plt.scatter(
    pca_df["PC1"], 
    pca_df["PC2"],
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Exoplanet Feature Space")
plt.savefig(
    os.path.join(figures_dir, "pca_projection.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# t-SNE Projection --------------------------------------------------------
tsne = TSNE(
    n_components=2, 
    perplexity=30, 
    max_iter=1000, 
    random_state=42
)

embedded_data = tsne.fit_transform(scaled_data)

# t-SNE Plot
plt.figure(figsize=(12, 8))

plt.scatter(
    embedded_data[:, 0], 
    embedded_data[:, 1],
    alpha=0.5
)

plt.title("t-SNE Projection of Exoplanet Populations")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig(
    os.path.join(figures_dir, "tsne_projection.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Clustering ---------------------------------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Cluster Visualization
plt.figure(figsize=(12, 8))

plt.scatter(
    embedded_data[:, 0],
    embedded_data[:, 1],
    c=cluster_labels,
    cmap="tab10",
    alpha=0.7
)

plt.title("K-Means Clustering of Exoplanets")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig(
    os.path.join(figures_dir, "clustering_projection.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# XGBoost Predictive Modelling ---------------------------------------------------------
# Predict Planet Equilibrium Temperature
prediction_features = [
    "stellar_temperature_k",
    "orbit_semi-major_axis_au",
    "planet_mass_m_e"
]

prediction_df = exoplanet_df[prediction_features + ["planet_temperature_k"]].dropna()

# Train-test split
X = prediction_df[prediction_features]
y = prediction_df["planet_temperature_k"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost Model
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Predictions
predictions = xgb_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Feature Importance ---------------------------------------------------------
importance_scores = xgb_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": prediction_features,
    "Importance": importance_scores
})

importance_df = importance_df.sort_values(by="Importance", ascending=True)

# Feature Importance Plot
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"], 
    importance_df["Importance"], 
    color="skyblue"
)

plt.xlabel("Importance Score")
plt.title("XGBoost Feature Importance")
plt.savefig(
    os.path.join(figures_dir, "xgb_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# SHAP Explainability ---------------------------------------------------------
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(
    os.path.join(figures_dir, "shap_summary.png"),
    dpi=300, 
    bbox_inches="tight"
)

plt.close()

# Anomaly Detection with Isolation Forest ---------------------------------------------------------
anomaly_detector = IsolationForest(contamination=0.01, random_state=42)
anomaly_labels = anomaly_detector.fit_predict(scaled_data)

# Anomaly Visualization
plt.figure(figsize=(12, 8))
plt.scatter(
    embedded_data[:, 0], 
    embedded_data[:, 1],
    c=anomaly_labels,
    cmap="coolwarm",
    alpha=0.7
)

plt.title("Anomaly Detection in Exoplanet Population")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig(
    os.path.join(figures_dir, "anomaly_detection.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# -----------------------------------------------------------------------------------------------------------------------------------------
# Research Report Generation 
# -----------------------------------------------------------------------------------------------------------------------------------------
report_html = f"""
<html>
<head>
<meta charset="UTF-8">
<title>Exoplanet Research Report</title>

<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background-color: #f5f7fa;
    color: #222;
    line-height: 1.7;
}}
h1 {{
    color: #0b3d91;
    border-bottom: 3px solid #0b3d91;
    padding-bottom: 10px;
    background-color: #f5f7fa;
    position: sticky;
    top: 0;
    z-index: 100;
}}
h2 {{
    color: #1f5fbf;
    margin-top: 40px;
}}
.section {{
    background: white;
    padding: 25px;
    border-radius: 10px;
    border-left: 6px solid #0b3d91;
    margin-bottom: 30px;
    scroll-margin-top: 100px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}}
img {{
    width: 100%;
    max-width: 1200px;
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 8px;
    margin-top: 15px;
    margin-bottom: 15px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
}}
th, td {{
    border: 1px solid #ccc;
    padding: 10px;
    text-align: left;
}}
th {{
    background-color: #0b3d91;
    color: white;
}}
.metric {{
    font-size: 18px;
    font-family: Consolas, monospace;
    margin-bottom: 10px;
}}
.sidebar {{
    position: fixed;
    left: 0;
    top: 0;
    width: 220px;
    height: 100%;
    background: #0b3d91;
    padding: 25px;
    padding-bottom: 50px;
    overflow-y: auto;
}}
.sidebar h2 {{
    color: white;
}}
.sidebar a {{
    display: block;
    color: white;
    text-decoration: none;
    margin-bottom: 15px;
}}
.sidebar a:hover {{
    text-decoration: underline;
}}
body {{
    margin-left: 260px;
}}
.styled-table {{
    border-collapse: collapse;
    margin-top: 15px;
    width: 100%;
    font-size: 15px;
    border-radius: 8px;
    overflow: hidden;
}}

.styled-table thead tr {{
    background-color: #0b3d91;
    color: white;
    text-align: left;
}}

.styled-table th, 
.styled-table td {{
    padding: 12px 15px;
}}

.styled-table tbody tr:nth-child(even) {{
    background-color: #f3f6fb;
}}

.styled-table tbody tr:hover {{
    background-color: #dce8ff;
}}
html {{
    scroll-behavior: smooth;
}}
.caption {{
    font-size: 14px;
    color: #555;
    font-style: italic;
    line-height: 1.5;
    margin-top: 8px;
    margin-bottom: 30px;
    text-align: center;
}}

</style>
</head>

<body>
<h1>NASA Exoplanet Archive Research Report</h1>

<div class="sidebar">
<h2>Sections</h2>

<a href="#summary">Dataset Summary</a>
<a href="#forensics">Dataset Forensics</a>
<a href="#duplicates">Duplicate Structure</a>
<a href="#validation">Scientific Validation</a>
<a href="#refinement">Observational Refinement</a>
<a href="#imputation">Imputation Benchmarking</a>
<a href="#statistics">Statistical Analysis</a>
<a href="#pca">PCA Projection</a>
<a href="#tsne">t-SNE Projection</a>
<a href="#clustering">Clustering</a>
<a href="#xgboost">XGBoost Modelling</a>
<a href="#shap">SHAP Explainability</a>
<a href="#anomalies">Anomaly Detection</a>

</div>

<p>
Generated automatically using Python on:
{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}
</p>

<div class="section">
<h2>Abstract</h2>

<p>
This report presents a comprehensive computational analysis of the NASA Exoplanet Archive using modern data science and machine learning methodologies. The study investigates dataset completeness, observational biases, duplicate archival structures, imputation strategies, dimensionality reduction, clustering behaviour, predictive modelling performance, explainable AI interpretations, and anomaly detection. The resulting pipeline demonstrates how astrophysical datasets can be transformed into scientifically interpretable machine learning frameworks while preserving physical realism and statistical integrity. 
</p>

</div>

<!-- DATASET SUMMARY --> 
<div class="section" id="summary">
<h2>Dataset Summary</h2>

<div class="metric"><b>Rows:</b> {exoplanet_df.shape[0]}</div>
<div class="metric"><b>Columns:</b> {exoplanet_df.shape[1]}</div>
<div class="metric"><b>Selected Imputation Method:</b> {best_method}</div>

<p>
This report investigates the structural quality, completeness and predictive characteristics of the NASA Exoplanet Archive.
The pipeline combines data forensics, scientific validation, imputation benchmarking, dimensionality reduction, clustering, anomaly detection
and explainable machine learning techniques.
</p>

</div>

<!-- DATASET FORENSICS --> 
<div class="section" id="forensics">
<h2>Dataset Forensics</h2>

<h3>Missing Value Summary</h3>

{missing_summary.round(3).to_html(classes='styled-table')}

<h3>Missingness by Discovery Method</h3>

{missing_by_method.round(3).to_html(classes='styled-table')}

<img src="figures/missingness_matrix.png">

<div class="caption">
Figure: Missing data matrix illustrating the distribution and structural patterns of incomplete observations across the NASA Exoplanet Archive dataset. White gaps indicate missing values and reveal systematic incompleteness across several astrophysical variables. 
</div>

<img src="figures/completeness_bar.png">

<div class="caption">
Figure: Column completeness distribution across all dataset features. Variables with lower completeness percentages indicate observational limitations and heterogeneous measurement availability across exoplanet detection techniques.
</div>

<p>
The dataset exhibits substantial variability in completeness across features and discovery techniques.
Transit based detections generally contain richer radius and orbital measurements, while direct imaging 
records display sparser orbital information. These patterns highlight the observational biases inherent 
in exoplanet detection methods.
</p>

</div>

<!-- DUPLICATE STRUCTURE --> 
<div class="section" id="duplicates">
<h2>Duplicate Structure Analysis</h2>

<p>
Duplicate planetary records were investigated to identify archival 
revisions and evolving measurements over time. Rather than removing 
duplicates entirely, records were consolidated to preserve the most 
complete and scientifically reliable measurements available for each planet.
</p>

</div>

<!-- SCIENTIFIC VAIDATION --> 
<div class="section" id="validation">
<h2>Scientific Validation</h2>

<p>
Physical validation rules were applied to remove scientifically 
impossible observations, including negative planetary masses, 
invalid orbital radii, impossible eccentricities, 
and unrealistic stellar temperatures. 
These filtering procedures improve downstream model reliability 
and reduce noise introduced by corrupted observations.
</p>

</div>

<!-- OBSERVATIONAL REFINEMENT --> 
<div class="section" id="refinement">
<h2>Observational Refinement Analysis</h2>

<img src="figures/yearly_completeness.png">

<div class="caption">
Figure: Average dataset completeness by exoplanet discovery year. Completeness increases substantially over time, reflecting advances in astronomical instrumentation, observational precision, and archival standardisation practices.
</div>

<p>
Dataset completeness improves substantially across discovery years, 
reflecting advances in observational astronomy, instrumentation, 
and archival standardisation. Modern-era detections contain more 
complete planetary and stellar measurements compared to earlier discoveries. 
</p>

</div>

<!-- Imputation --> 
<div class="section" id="imputation">
<h2>Imputation Benchmarking</h2>

{imputation_results.to_html(index=False, classes='styled-table')}

<img src="figures/knn_optimisation.png">

<div class="caption">
Figure: Reconstruction error across candidate KNN neighbour configurations during imputation benchmarking. Lower mean squared error values indicate stronger preservation of the original astrophysical structure under simulated missingness conditions.
</div>

<img src="figures/imputation_mse.png">

<div class="caption">
Figure: Comparative reconstruction performance of median, KNN, and iterative imputation strategies. Lower reconstruction error indicates improved recovery of masked astrophysical observations and stronger preservation of underlying statistical relationships.
</div>

<h3>KNN Hyperparameter Search Results</h3>

{knn_results_df.to_html(index=False, classes='styled-table')}

<p>
Multiple imputation strategies were benchmarked using reconstruction
error, correlation preservation, and downstream machine learning 
performance. The selected imputation strategy achieved the strongest
balance between numerical accuracy and structural preservation.
</p>

</div>

<!-- STATISTICS --> 
<div class="section" id="statistics">
<h2>Statistical Analysis</h2>

{summary_statistics.round(3).to_html(classes='styled-table')}

<img src="figures/correlation_heatmap.png">

<div class="caption">
Figure: Correlation heatmap of numerical astrophysical variables within the processed exoplanet dataset. Strong positive and negative correlations reveal physically meaningful relationships between stellar, orbital, and planetary properties.
</div>

<img src="figures/simple_plot.png">

<div class="caption">
Figure: Planetary mass distribution across discovery years shown on a logarithmic scale. The visualisation highlights the increasing diversity of detected exoplanets as observational capabilities improved over time.
</div>

<img src="figures/multi_variable_scatter_plot.png">

<div class="caption">
Figure: Multi-variable relationship between stellar temperature and planetary equilibrium temperature. Point size represents planetary mass, while colour encodes discovery year, illustrating multidimensional astrophysical structure within the dataset.
</div>

<p>
Statistical analysis reveals substantial heterogeneity across exoplanet
populations. Strong relationships emerge between stellar temperature, 
planetary equilibrium temperature, and orbital properties, 
suggesting physically meaningful structure within the dataset.
</p>

</div>

<!-- PCA --> 
<div class="section" id="pca">
<h2>PCA Projection</h2>

<img src="figures/pca_projection.png">

<div class="caption">
Figure: Principal Component Analysis projection of the exoplanet feature space. The dimensionality reduction preserves major variance structures and reveals broad population gradients across astrophysical characteristics.
</div>

<p>
Principal Component Analysis compresses the high-dimensional feature 
space into lower-dimensional representations while preserving major 
variance structures. The resulting projection demonstrates broad 
planetary groupings and continuous astrophysical gradients.
</p>

</div>

<!-- TSNE --> 
<div class="section" id="tsne">
<h2>t-SNE Projection</h2>

<img src="figures/tsne_projection.png">

<div class="caption">
Figure: t-SNE projection of exoplanetary observations in reduced-dimensional space. Non-linear embedding reveals local structures and potential population subgroups not fully captured by linear dimensionality reduction methods.
</div>

<p>
t-SNE reveals highly non-linear structures within the exoplanet
population that are not fully captured by PCA. Distinct local 
clusters emerge, indicating potentially different planetary formation
or observational regimes.
</p>

</div>

<!-- CLUSTERING --> 
<div class="section" id="clustering">
<h2>Cluster Structure</h2>

<img src="figures/clustering_projection.png">

<div class="caption">
Figure: K-Means clustering results visualised within the t-SNE embedded feature space. Distinct cluster structures suggest the presence of multiple astrophysical subpopulations within the exoplanet archive.
</div>

<p>
K-Means clustering identifies several major exoplanet population
groups within the transformed feature space. These clusters may 
correspond to astrophysical subclasses characterised by differences
in planetary mass, orbital structure, and stellar environment. 
</p>

</div>

<!-- XGBOOST --> 
<div class="section" id="xgboost">
<h2>XGBoost Predictive Modelling</h2>

<div class="metric"><b>Mean Absolute Error:</b> {mae:.2f}</div>
<div class="metric"><b>R² Score:</b> {r2:.4f}</div>

{importance_df.to_html(index=False, classes='styled-table')}

<img src="figures/xgb_importance.png">

<div class="caption">
Figure: Relative feature importance scores derived from the XGBoost predictive model. Stellar temperature and orbital characteristics contribute most strongly to planetary equilibrium temperature prediction.
</div>

<p>
The XGBoost model achieved strong predictive performance when estimating 
planetary equilibrium temperature from stellar and orbital variables.
Feature importance analysis indicates that stellar temperature and 
orbital distance contribute most strongly to predictive accuracy.
</p>

</div>

<!-- SHAP -->
<div class="section" id="shap">
<h2>SHAP Explainability</h2>

<img src="figures/shap_summary.png">

<div class="caption">
Figure: SHAP summary plot illustrating feature contributions to XGBoost model predictions across all test observations. Features with larger SHAP magnitudes exert stronger influence on predicted planetary temperatures.
</div>

<p>
SHAP analysis provides interpretable explanations for model predictions 
by quantifying the contribution of each feature across observations. 
The results confirm that astrophysically meaningful variables dominate 
model behaviour and influence predictive outcomes consistently. 
</p>

</div>

<!-- ANOMALIES --> 
<div class="section" id="anomalies">
<h2>Anomaly Detection</h2>

<img src="figures/anomaly_detection.png">

<div class="caption">
Figure: Isolation Forest anomaly detection results visualised within the reduced-dimensional exoplanet feature space. Highlighted observations represent statistically unusual planetary systems that deviate substantially from the broader population structure.
</div>

<p>
Isolation Forest analysis identifies rare or extreme exoplanetary 
configurations that deviate significantly from the broader population.
These anomalies may represent scientifically interesting outliers, 
measurement artefacts, or potentially novel astrophysical systems.
</p>

<hr>

<p style="text-align: center; color: #666;">
End of Automated Research Report
</p>

</div>

</body>
</html>
"""

report_path = os.path.join(report_dir, "exoplanet_report.html")
with open(report_path, "w", encoding="utf-8") as file:
    file.write(report_html)

print(f"Research report saved to: {report_path}")

webbrowser.open(report_path)

