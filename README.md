# 🌍 TerraSense

**High-Resolution Spatial Downscaling of Soil Organic Carbon via Dual-Sensor Fusion and XGBoost Spatial Embedding**

## 📌 Overview

**TerraSense** is an applied machine learning framework designed to solve the resolution mismatch in global digital soil mapping. Traditional products like SoilGrids provide Soil Organic Carbon (SOC) data at a coarse 250m resolution, which is insufficient for precision agriculture in heterogeneous landscapes.

This project spatially downscales SOC estimates to a highly actionable **10m resolution** by fusing multi-temporal optical indices (Sentinel-2) with physical surface texture metrics (Sentinel-1 SAR). By explicitly embedding spatial autocorrelations (latitude/longitude) into an Extreme Gradient Boosting (XGBoost) architecture, TerraSense delivers robust soil intelligence even under dense canopy cover.

**Study Area:** Thanjavur District, Tamil Nadu, India.

-----

## 🚀 Key Features

  * **Dual-Sensor Fusion:** Combines optical "color" (NDVI, SWIR) with radar "structure" (VV/VH backscatter, GLCM textures) to bypass optical sensor saturation.
  * **Spatial Embedding:** Explicitly injects geospatial coordinates into the feature matrix to capture regional micro-climates and spatial autocorrelations.
  * **High Accuracy:** Outperforms baseline SVM and Random Forest models, achieving $R^2 = 0.7026$ and $\text{RMSE} = 1.6582$ g/kg.
  * **Explainable AI (XAI):** Integrates SHAP (SHapley Additive exPlanations) to validate the physical significance of radar backscatter in SOC prediction.
  * **Interactive DSS:** Includes a Streamlit-based Decision Support System for real-time 2D/3D visualization of downscaled maps.

-----

## 🧠 System Architecture

The TerraSense pipeline is structured into four distinct micro-phases:

1.  **Data Acquisition (GEE):** Cloud-side extraction of Sentinel-1 (C-Band SAR), Sentinel-2 (MSI), NASA SRTM DEM, and baseline SoilGrids250m target data.
2.  **Feature Engineering:** Cloud and local computation of the 16-Dimensional feature matrix, including vegetation indices and GLCM texture metrics (Contrast, Entropy).
3.  **AI Core:** An XGBoost regression engine trained on an 80/20 spatial split, evaluated with strict hold-out cross-validation.
4.  **Application Layer:** A localized inference engine connected to a Streamlit web dashboard.

-----

## ⚙️ Installation & Setup

### Prerequisites

  * Python 3.9 or higher
  * A registered [Google Earth Engine](https://earthengine.google.com/) account.

### 1\. Clone the Repository

```bash
git clone https://github.com/yourusername/TerraSense.git
cd TerraSense
```

### 2\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3\. Authenticate Google Earth Engine

Before running the extraction scripts or backend pipeline, authenticate your local environment with GEE:

```bash
earthengine authenticate
```

### 4\. Run the Streamlit Dashboard

To launch the interactive Geo-Soil Intelligence dashboard locally:

```bash
streamlit run app.py
```

-----

## 📊 Feature Matrix Definition

The model expects a strictly ordered 16-dimensional input vector for inference:

1.  **Optical:** `B2`, `B3`, `B4`, `B8`, `B11`, `B12`, `NDVI`
2.  **Radar:** `VV`, `VH`, `VV_Contrast`, `VV_Entropy`, `VH_Entropy`
3.  **Topography:** `Elevation` (DEM), `LandCover`
4.  **Spatial:** `Latitude`, `Longitude`

-----

## 🔬 Explainable AI Results

SHAP analysis conducted during model validation confirms that **Latitude** (spatial mapping) and **VH Backscatter** (radar volume scattering) are the most dominant drivers of the model's predictions, heavily validating the core hypothesis of the Dual-Sensor approach.

-----
