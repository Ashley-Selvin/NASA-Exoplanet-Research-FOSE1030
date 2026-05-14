import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
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
plt.show()

plt.figure(figsize=(14, 8))
msno.bar(exoplanet_df) # Bar plot showing completeness distribution
plt.title("Column Completeness")
plt.show()


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

# Remove impossible eccentricity values
exoplanet_df = exoplanet_df[(exoplanet_df["pl_orbeccen"] >= 0) & (exoplanet_df["pl_orbeccen"] < 1)]

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

plt.show()

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

# Median Imputation -------------------------------------------------------------------------------------------------------------------------------------------------------------
median_imputer = SimpleImputer(strategy="median")
median_imputed = pd.DataFrame(median_imputer.fit_transform(masked_data), columns=masked_data.columns)

# KNN Imputation ----------------------------------------------------------------------------------------------------------------------------------------------------------------
knn_imputer = KNNImputer(n_neighbors=5)
knn_imputed = pd.DataFrame(knn_imputer.fit_transform(masked_data), columns=masked_data.columns)

# Iterative Imputation ----------------------------------------------------------------------------------------------------------------------------------------------------------
iterative_imputer = IterativeImputer(random_state=42, max_iter=20)
iterative_imputed = pd.DataFrame(iterative_imputer.fit_transform(masked_data), columns=masked_data.columns)

# Reconstruction Error Evaluation -----------------------------------------------------------------------------------------------------------------------------------------------
def evaluate_imputation(original, imputed, mask):
    return mean_squared_error(original[mask], imputed[mask])

median_mse = evaluate_imputation(original_values.values, median_imputed.values, mask)
knn_mse = evaluate_imputation(original_values.values, knn_imputed.values, mask)
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

median_corr_difference = np.abs(original_corr - knn_corr).mean().mean()
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

plt.show()

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
plt.show()

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
imputation_results["MSE_Normalized"] = ((imputation_results["MSE"].max() - imputation_results["MSE"]) / (imputation_results["MSE"].max() - imputation_results["MSE"].min()))
imputation_results["Correlation_Normalized"] = ((imputation_results["Correlation_Distortion"].max() - imputation_results["Correlation_Distortion"]) / (imputation_results["Correlation_Distortion"].max() - imputation_results["Correlation_Distortion"].min()))
imputation_results["R2_Normalized"] = ((imputation_results["R2"] - imputation_results["R2"].min()) / (imputation_results["R2"].max() - imputation_results["R2"].min()))

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
    final_imputer = KNNImputer(n_neighbors=5)
else:
    final_imputer = IterativeImputer(random_state=42, max_iter=20)

# Apply selected imputer to full dataset --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
numerical_columns = exoplanet_df.select_dtypes(include=np.number).columns
exoplanet_df[numerical_columns] = final_imputer.fit_transform(exoplanet_df[numerical_columns])

print(f"\nFinal preprocessing pipeline applied using "f"{best_method} imputation.")




# Feature Engineering --------------------------------------------------------
# Log Planet Mass 
exoplanet_df["log_planet_mass"] = np.log10(exoplanet_df["planet_mass_m_e"])

# Discovery Era 
def classify_era(year):
    if year < 2005:
        return "Early Discovery Era"
    elif year < 2015:
        return "Kepler Era"
    else:
        return "Modern Era"
    
exoplanet_df["discovery_era"] = (
    exoplanet_df["disc_year"].apply(classify_era)
)

# Habitability Proxy
exoplanet_df["habitability_score"] = (
    1 / (1 + abs(exoplanet_df["planet_temperature_k"] - 288))
)

# Numerical Analysis --------------------------------------------------------
selected_columns = [
    "planet_mass_m_e", 
    "planet_temperature_k", 
    "stellar_temperature_k",
    "pl_orbeccen",
]

summary_statistics = pd.DataFrame({
    "Mean": exoplanet_df[selected_columns].mean(),
    "Median": exoplanet_df[selected_columns].median(),
    "Standard Deviation": exoplanet_df[selected_columns].std()
})
print(summary_statistics)

# Correlation Heatmap -------------------------
plt.figure(figsize=(14, 10))

correlation_matrix = (exoplanet_df[numerical_columns].corr())

sns.heatmap(
    correlation_matrix, 
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap")
plt.show()

# Simple Plot -----------------------------------------------------
# Planet Mass vs Discovery Year
plt.figure(figsize=(12, 7))

plt.scatter(
    exoplanet_df["disc_year"], 
    exoplanet_df["planet_mass_m_e"],
    alpha=0.5
)

plt.yscale("log")

plt.xlabel("Discovery Year")
plt.ylabel("Planet Mass (Earth Masses)")
plt.title("Planet Mass vs Discovery Year")

plt.show()

# Multi-Variable Scatter Plot -----------------------------------------------------
plt.figure(figsize=(14, 8))

scatter = plt.scatter(
    exoplanet_df["stellar_temperature_k"], 
    exoplanet_df["planet_temperature_k"],
    s=exoplanet_df["planet_mass_m_e"] * 0.3,
    c=exoplanet_df["disc_year"],
    alpha=0.6,
    cmap="viridis"
)

plt.colorbar(scatter, label="Discovery Year")

plt.xlabel("Stellar Temperature (K)")
plt.ylabel("Planet Equilibrium Temperature (K)")
plt.title("Planetary Temperature vs Stellar Temperature")

plt.show()

# PCA Dimensionality Reduction --------------------------------------------------------
features_for_ml = [
    "planet_mass_m_e",
    "planet_temperature_k",
    "stellar_temperature_k",
    "pl_orbsmax",
    "pl_orbeccen",
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

plt.show()

# t-SNE Projection --------------------------------------------------------
tsne = TSNE(
    n_components=2, 
    perplexity=30, 
    n_iter=1000, 
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

plt.show()

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

plt.show()

# XGBoost Predictive Modelling ---------------------------------------------------------
# Predict Planet Equilibrium Temperature
prediction_features = [
    "stellar_temperature_k",
    "pl_orbsmax",
    "planet_mass_m_e",
    "pl_orbeccen"
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

plt.show()

# SHAP Explainability ---------------------------------------------------------
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test)

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
plt.show()

