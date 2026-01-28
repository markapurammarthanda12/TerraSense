# ==============================================================================
# SOIL INTELLIGENCE PLATFORM (FINAL PRODUCTION VERSION)
# Features: Real Thanjavur Data (No Randoms), Firewall-Safe Viewer, Live AI
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import time

# 1. PAGE SETUP
st.set_page_config(page_title="TerraSense AI", layout="wide")
st.markdown("<style>.metric-card {background-color: #1E1E1E; padding: 20px; border-radius: 10px;}</style>", unsafe_allow_html=True)

# 2. DEFINE FEATURES (Exact match to your training)
EXPECTED_COLS = [
 'B11', 'B12', 'B2', 'B3', 'B4', 'B8', 'DEM', 'LandCover', 'NDVI', 
 'VH', 'VH_ent', 'VV', 'VV_contrast', 'VV_ent', 'latitude', 'longitude',
]

# 3. LOAD THE AI BRAIN
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    try:
        # Check if file exists to prevent hard crash
        if os.path.exists("xgb_soc_model_v1.json"):
            model.load_model("xgb_soc_model_v1.json")
            return model
        else:
            return None
    except:
        return None

ai_model = load_model()

# 4. DATA LOADER (FIXED: LOADS REAL CSV, NO RANDOM POINTS)
def load_and_predict(uploaded_file):
    if uploaded_file is not None:
        # CASE A: User Uploads a File
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'Predicted_SOC' in df.columns:
                return df
            else:
                # Check for columns needed for AI Prediction
                missing = [c for c in EXPECTED_COLS if c not in df.columns]
                
                if not missing and ai_model:
                    with st.spinner('🤖 AI Model Running... Predicting SOC...'):
                        X = df[EXPECTED_COLS]
                        df['Predicted_SOC'] = ai_model.predict(X)
                    return df
                else:
                    if not ai_model:
                        st.error("⚠️ Model file (json) missing. Cannot run inference.")
                    if missing:
                        st.error(f"⚠️ Uploaded file missing columns: {missing}")
                    return pd.DataFrame()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return pd.DataFrame()
    else:
        # CASE B: DEMO MODE (FIXED)
        # Strictly load the Thanjavur CSV. Do NOT generate random points.
        csv_path = 'Final_Predicted_SOC_Thanjavur.csv'
        
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                st.error(f"Error loading Demo Data: {e}")
                return pd.DataFrame()
        else:
            # Only show this if the CSV is actually missing from the folder
            st.warning("⚠️ 'Final_Predicted_SOC_Thanjavur.csv' not found in folder. Upload a file to begin.")
            return pd.DataFrame()

# 5. IMAGE GENERATOR
def save_map_image(df_to_plot, col_name, vmin, vmax, filename):
    try:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=75)
        fig.patch.set_facecolor('white')
        ax.axis('off')
        ax.scatter(
            df_to_plot['longitude'], 
            df_to_plot['latitude'], 
            c=df_to_plot[col_name], 
            cmap='RdYlGn', 
            vmin=vmin, 
            vmax=vmax, 
            s=25, 
            marker='s'
        )
        plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return True
    except Exception as e:
        return False

# ==============================================================================
# MAIN APP UI
# ==============================================================================
with st.sidebar:
    st.title("📂 TerraSense Manager")
    uploaded_file = st.file_uploader("Upload Region CSV", type=["csv"])
    st.markdown("---")
    st.header("⚙️ Scenario Simulator")
    rain_impact = st.slider("Rainfall Change (%)", -50, 50, 0)
    veg_impact = st.slider("Vegetation Change (%)", -50, 50, 0)

# Load Data
df = load_and_predict(uploaded_file)

if not df.empty and 'Predicted_SOC' in df.columns:
    # 1. Simulation Logic
    # Ensure numeric types
    df['Predicted_SOC'] = pd.to_numeric(df['Predicted_SOC'], errors='coerce')
    df['Simulated_SOC'] = df['Predicted_SOC'] * (1 + (rain_impact*0.01) + (veg_impact*0.02))

    # 2. Dynamic Title
    dataset_name = "Thanjavur District (Demo)" if uploaded_file is None else "Custom Region Analysis"
    st.title(f"🌍 TerraSense: {dataset_name}")

    # 3. Tabs
    # Note: We use the "Offline-Safe" Side-by-Side view in Tab 3 to prevent Firewall crashes
    tab1, tab2, tab3 = st.tabs(["📊 Live Map", "🏔️ 3D Twin", "⚡ Impact Visualizer"])

    with tab1:
        c_min, c_max = df['Predicted_SOC'].min(), df['Predicted_SOC'].max()
        try:
            fig = px.scatter_map(
                df, lat="latitude", lon="longitude", color="Simulated_SOC", 
                range_color=[c_min, c_max], color_continuous_scale="RdYlGn", 
                size_max=15, zoom=9, map_style="carto-darkmatter", height=550
            )
        except:
            fig = px.scatter_mapbox(
                df, lat="latitude", lon="longitude", color="Simulated_SOC", 
                range_color=[c_min, c_max], color_continuous_scale="RdYlGn", 
                size_max=15, zoom=9, mapbox_style="carto-darkmatter", height=550
            )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        view_state = pdk.ViewState(latitude=df['latitude'].mean(), longitude=df['longitude'].mean(), zoom=9, pitch=45)
        layer = pdk.Layer("ColumnLayer", data=df, get_position=["longitude", "latitude"],
                          get_elevation="DEM", elevation_scale=50, radius=200,
                          get_fill_color="[255-(Predicted_SOC*5), Predicted_SOC*6, 50, 200]", pickable=True)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Predicted_SOC}"}))

    with tab3:
        # FIREWALL-SAFE VISUALIZATION (Side-by-Side)
        st.write("### 🔮 Impact Visualization")
        
        if rain_impact == 0 and veg_impact == 0:
            st.info("ℹ️ Move the sliders in the sidebar (Left) to simulate a change.")
        
        with st.status("🚀 Generating Comparison...", expanded=True) as status:
            # Timestamp busts browser cache
            timestamp = int(time.time())
            file_before = f"map_before_{timestamp}.png"
            file_after = f"map_after_{timestamp}.png"
            
            vmin, vmax = df['Predicted_SOC'].min(), df['Predicted_SOC'].max()
            save_map_image(df, 'Predicted_SOC', vmin, vmax, file_before)
            save_map_image(df, 'Simulated_SOC', vmin, vmax, file_after)
            
            status.update(label="✅ Complete!", state="complete", expanded=False)
        
        # Two columns for side-by-side view (Reliable)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Current Reality")
            st.image(file_before, use_container_width=True)
            
        with col2:
            st.markdown("#### 🔮 Future Prediction")
            st.image(file_after, use_container_width=True)
            
        # Add a metric to show the difference
        avg_change = df['Simulated_SOC'].mean() - df['Predicted_SOC'].mean()
        st.metric(label="Net Carbon Change", value=f"{avg_change:.2f} g/kg", delta_color="normal")

else:
    # If Thanjavur CSV is missing and no upload, show specific error
    st.info("👈 Please upload a CSV file to begin analysis.")









































# # ==============================================================================
# # SOIL INTELLIGENCE PLATFORM (LIVE AI EDITION)
# # Features: Loads Trained Model -> Predicts on New Data Live
# # ==============================================================================
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import pydeck as pdk
# import numpy as np
# import xgboost as xgb  # NEW: We need XGBoost inside the app now
# import matplotlib.pyplot as plt
# from streamlit_image_comparison import image_comparison
# import io
# from PIL import Image

# # 1. PAGE SETUP
# st.set_page_config(page_title="Geo-Soil Analytics Engine", layout="wide")
# st.markdown("<style>.metric-card {background-color: #1E1E1E; padding: 20px; border-radius: 10px;}</style>", unsafe_allow_html=True)

# # 2. DEFINE FEATURES (MUST MATCH YOUR TRAINING EXACTLY)
# # Copy this list from your Notebook Phase 17 output to be safe
# MODEL_FEATURES = [
#   'B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI',
#   'VV', 'VH', 'VV_contrast', 'VV_ent', 'VH_ent',
#   'NDVI_StdDev', 'DEM', 'LandCover', 'SOC', # Note: 'SOC' might be target, check if input
#   'longitude', 'latitude' 
# ]
# # NOTE: If 'SOC' (Target) was in your features list by mistake in the notebook, remove it here!
# # The model inputs should usually be just the bands + coords.
# # A safe standard list (Update based on your notebook printout):
# # 2. DEFINE FEATURES (MUST MATCH TRAINING EXACTLY)
# # I have removed 'NDVI_StdDev' because the model was trained without it.
# EXPECTED_COLS = [
#  'B11', 'B12', 'B2', 'B3', 'B4', 'B8', 'DEM', 'LandCover', 'NDVI', 'VH', 'VH_ent', 'VV', 'VV_contrast', 'VV_ent', 'latitude', 'longitude',
# ]

# # 3. LOAD THE AI BRAIN
# @st.cache_resource
# def load_model():
#     model = xgb.XGBRegressor()
#     try:
#         model.load_model("xgb_soc_model_v1.json")
#         return model
#     except:
#         st.error("⚠️ Model file 'xgb_soc_model_v1.json' not found.")
#         return None

# ai_model = load_model()

# # 4. SMART DATA LOADER (Handles Raw vs Processed Data)
# def load_and_predict(uploaded_file):
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
            
#             # CASE A: File already has predictions (Fast Load)
#             if 'Predicted_SOC' in df.columns:
#                 return df
            
#             # CASE B: File is RAW data (Needs AI Prediction)
#             else:
#                 # Check if it has the required columns to run the model
#                 missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
                
#                 if not missing_cols:
#                     with st.spinner('🤖 AI Model Running... Predicting Soil Carbon on new data...'):
#                         # Filter to only the columns the model knows
#                         X_input = df[EXPECTED_COLS]
#                         # RUN PREDICTION LIVE
#                         df['Predicted_SOC'] = ai_model.predict(X_input)
#                         st.success("✅ AI Inference Complete!")
#                     return df
#                 else:
#                     st.error(f"⚠️ Cannot predict. Uploaded file is missing columns: {missing_cols}")
#                     return pd.DataFrame()
                    
#         except Exception as e:
#             st.error(f"Error: {e}")
#             return pd.DataFrame()
#     else:
#         # Default Demo File
#         try:
#             return pd.read_csv('Final_Predicted_SOC_Thanjavur.csv')
#         except:
#             return pd.DataFrame()

# # Helper for Image Generation
# @st.cache_data(show_spinner=False)
# def get_static_map_image(df_to_plot, col_name, vmin, vmax):
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
#     ax.axis('off')
#     ax.scatter(df_to_plot['longitude'], df_to_plot['latitude'], c=df_to_plot[col_name], 
#                cmap='RdYlGn', vmin=vmin, vmax=vmax, s=15, marker='s')
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
#     plt.close(fig)
#     buf.seek(0)
#     return Image.open(buf)

# # ==============================================================================
# # MAIN APP UI
# # ==============================================================================
# with st.sidebar:
#     st.title("📂 Universal AI Analyst")
#     uploaded_file = st.file_uploader("Upload District Grid (CSV)", type=["csv"])
#     st.markdown("---")
#     st.header("⚙️ Simulation")
#     rain_impact = st.slider("Rainfall Change (%)", -50, 50, 0)
#     veg_impact = st.slider("Vegetation Change (%)", -50, 50, 0)

# df = load_and_predict(uploaded_file)

# if not df.empty and 'Predicted_SOC' in df.columns:
#     # 1. Risk Logic
#     conditions = [(df['Predicted_SOC']<20), (df['Predicted_SOC']>=20)&(df['Predicted_SOC']<30), (df['Predicted_SOC']>=30)]
#     df['Risk_Level'] = np.select(conditions, ['CRITICAL', 'MODERATE', 'HEALTHY'], default='Unknown')

#     # 2. Simulation Logic
#     df['Simulated_SOC'] = df['Predicted_SOC'] * (1 + (rain_impact*0.01) + (veg_impact*0.02))

#     # 3. Dynamic Title
#     title_text = "Thanjavur (Demo)" if uploaded_file is None else "Custom Region Analysis"
#     st.title(f"🌍 Geo-Soil Intelligence: {title_text}")

#     # 4. Tabs
#     tab1, tab2, tab3 = st.tabs(["📊 Live Map", "🏔️ 3D Twin", "⚡ Impact Slider"])

#     with tab1:
#         c_min, c_max = df['Predicted_SOC'].min(), df['Predicted_SOC'].max()
#         fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="Simulated_SOC", 
#                                 range_color=[c_min, c_max], color_continuous_scale="RdYlGn", 
#                                 size_max=15, zoom=9, mapbox_style="carto-darkmatter", height=550)
#         st.plotly_chart(fig, use_container_width=True)

#     with tab2:
#         view_state = pdk.ViewState(latitude=df['latitude'].mean(), longitude=df['longitude'].mean(), zoom=9, pitch=45)
#         layer = pdk.Layer("ColumnLayer", data=df, get_position=["longitude", "latitude"],
#                           get_elevation="Predicted_SOC", elevation_scale=50, radius=200,
#                           get_fill_color="[255-(Predicted_SOC*5), Predicted_SOC*6, 50, 200]", pickable=True)
#         st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Predicted_SOC}"}))

#     with tab3:
#         if rain_impact == 0 and veg_impact == 0:
#             st.info("Move sliders to compare.")
#         else:
#             vmin, vmax = df['Predicted_SOC'].min(), df['Predicted_SOC'].max()
#             img1 = get_static_map_image(df, 'Predicted_SOC', vmin, vmax)
#             img2 = get_static_map_image(df, 'Simulated_SOC', vmin, vmax)
#             image_comparison(img1=img1, img2=img2, label1="Current", label2="Predicted", width=700, in_memory=True)

# else:
#     st.info("👈 Upload a CSV file to run the AI Model.")