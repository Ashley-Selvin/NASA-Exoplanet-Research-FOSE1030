import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
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

# Initial Dataset Inspection
print(exoplanet_df.head())
print(exoplanet_df.info())
print(exoplanet_df.describe())

# Missing Value Analysis ------------------------
# Missing Value Counts
missing_counts = exoplanet_df.isnull().sum()
missing_percentages = (
    exoplanet_df.isnull().mean() * 100
).sort_values(ascending=False)

print(missing_percentages)

# Missingness Heatmap
plt.figure(figsize=(14, 8))
msno.matrix(exoplanet_df)
plt.title("Missing Data Matrix")
plt.show()

# Missingness by Discovery Method
missing_by_method = (
    exoplanet_df
    .groupby("discoverymethod")
    .apply(lambda x: x.isnull().mean() * 100)
)

print(missing_by_method)

# Scientific Constraint Validation ------------------------
# Example constraints:
# Remove impossible eccentricity values
exoplanet_df = exoplanet_df[
    (exoplanet_df["pl_orbeccen"] >= 0) & 
    (exoplanet_df["pl_orbeccen"] < 1)
]

# Remove non-positive planet masses 
exoplanet_df = exoplanet_df[
    exoplanet_df["pl_bmasse"] > 0
]

# Remove unrealistic stellar temperatures
exoplanet_df = exoplanet_df[
    (exoplanet_df["st_teff"] > 2000) & 
    (exoplanet_df["st_teff"] < 50000)
]

# Duplicate Removal ------------------------
print("Duplicate rows:", exoplanet_df.duplicated().sum())
exoplanet_df = exoplanet_df.drop_duplicates()

# Imputation --------------------------------------------------------
# Numerical Columns
numerical_columns = exoplanet_df.select_dtypes(include=np.number).columns

# KNN Imputation
imputer = KNNImputer(n_neighbors=5)
exoplanet_df[numerical_columns] = imputer.fit_transform(
    exoplanet_df[numerical_columns]
)

# Feature Engineering --------------------------------------------------------
# Log Planet Mass 
exoplanet_df["log_planet_mass"] = np.log10(exoplanet_df["pl_bmasse"])

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
    1 / (1 + abs(exoplanet_df["pl_eqt"] - 288))
)

# Numerical Analysis --------------------------------------------------------
selected_columns = [
    "pl_bmasse", 
    "pl_eqt", 
    "st_teff",
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
    exoplanet_df["pl_bmasse"],
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
    exoplanet_df["st_teff"], 
    exoplanet_df["pl_eqt"],
    s=exoplanet_df["pl_bmasse"] * 0.3,
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
    "pl_bmasse",
    "pl_eqt",
    "st_teff",
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
    "st_teff",
    "pl_orbsmax",
    "pl_bmasse",
    "pl_orbeccen"
]

prediction_df = exoplanet_df[prediction_features + ["pl_eqt"]].dropna()

# Train-test split
X = prediction_df[prediction_features]
y = prediction_df["pl_eqt"]
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

