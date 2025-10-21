# ---------------------------------
# WINE QUALITY STREAMLIT DASHBOARD
# ---------------------------------
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
import plotly.express as px
from pathlib import Path

# ---------------------------------
# LOAD MODEL
# ---------------------------------
MODEL_PATH = Path("/content/MLOps/Labs/API_Labs/FastAPI_Labs/model/wine_model.pkl")
st.set_page_config(page_title="Wine Quality Prediction Demo", page_icon="🍷", layout="wide")

try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# ---------------------------------
# PAGE TITLE
# ---------------------------------
st.title("🍷 Wine Quality Prediction Dashboard")
st.write("Predict the **wine cultivar (Class 0 / 1 / 2)** using chemical properties and explore the dataset interactively.")

# ---------------------------------
# TABS
# ---------------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Prediction", "📊 Data Exploration", "ℹ️ About"])

# ---------------------------------
# TAB 1: PREDICTION
# ---------------------------------
with tab1:
    st.header("Wine Class Prediction")

    # Sidebar sliders
    st.sidebar.header("Input Features")
    alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 12.0)
    malic_acid = st.sidebar.slider("Malic Acid", 0.5, 5.0, 2.0)
    ash = st.sidebar.slider("Ash", 1.0, 3.5, 2.3)
    alcalinity = st.sidebar.slider("Alcalinity of Ash", 10.0, 30.0, 15.0)
    magnesium = st.sidebar.slider("Magnesium", 70, 160, 100)
    total_phenols = st.sidebar.slider("Total Phenols", 0.5, 4.0, 2.0)
    flavanoids = st.sidebar.slider("Flavanoids", 0.0, 6.0, 3.0)
    nonflavanoid_phenols = st.sidebar.slider("Nonflavanoid Phenols", 0.0, 1.0, 0.3)
    proanthocyanins = st.sidebar.slider("Proanthocyanins", 0.5, 4.0, 1.5)
    color_intensity = st.sidebar.slider("Color Intensity", 1.0, 15.0, 5.0)
    hue = st.sidebar.slider("Hue", 0.5, 2.0, 1.0)
    od_ratio = st.sidebar.slider("OD280/OD315", 1.0, 4.0, 2.5)
    proline = st.sidebar.slider("Proline", 300, 1800, 800)

    # Summary table
    st.markdown("### 🧾 Your Input Summary")
    input_df = pd.DataFrame({
        "Alcohol":[alcohol],"Malic Acid":[malic_acid],"Ash":[ash],"Alcalinity":[alcalinity],
        "Magnesium":[magnesium],"Total Phenols":[total_phenols],"Flavanoids":[flavanoids],
        "Nonflavanoid Phenols":[nonflavanoid_phenols],"Proanthocyanins":[proanthocyanins],
        "Color Intensity":[color_intensity],"Hue":[hue],"OD280/OD315":[od_ratio],"Proline":[proline]
    })
    st.dataframe(input_df)

    # Prediction
    if st.button("🔍 Predict Wine Class"):
        features = np.array([[alcohol, malic_acid, ash, alcalinity, magnesium,
                              total_phenols, flavanoids, nonflavanoid_phenols,
                              proanthocyanins, color_intensity, hue, od_ratio, proline]])
        try:
            prediction = model.predict(features)[0]
            probs = model.predict_proba(features)[0]

            class_map = {0:"Cultivar 1",1:"Cultivar 2",2:"Cultivar 3"}
            st.success(f"**Predicted Wine Class:** {class_map[prediction]}")

            if prediction==0:
                st.info("🍇 Cultivar 1 – higher alcohol & lower ash.")
            elif prediction==1:
                st.info("🍷 Cultivar 2 – balanced phenols & medium hue.")
            else:
                st.info("🍾 Cultivar 3 – rich color & high proline.")

            probs_df = pd.DataFrame({
                "Cultivar":["Cultivar 1","Cultivar 2","Cultivar 3"],
                "Probability":probs
            })
            fig = px.bar(probs_df, x="Cultivar", y="Probability", color="Cultivar",
                         color_discrete_sequence=["#b5179e","#7209b7","#560bad"],
                         title="Prediction Confidence")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------------
# TAB 2: DATA EXPLORATION
# ---------------------------------
with tab2:
    st.header("📊 Explore the Wine Dataset")
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target

    feature = st.selectbox("Select a feature to visualize", df.columns[:-1])
    fig = px.histogram(df, x=feature, color=df["target"].astype(str),
                       nbins=20, barmode="overlay",
                       color_discrete_sequence=["#b5179e","#7209b7","#560bad"],
                       title=f"Distribution of {feature.title()} by Cultivar")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# TAB 3: ABOUT
# ---------------------------------
with tab3:
    st.header("ℹ️ About This App")
    st.write("""
    - **Author:** Mihir Parab  
    - **Course:** MLOps Streamlit Lab  
    - **Dataset:** scikit-learn Wine Dataset  
    - **Model:** Random Forest Classifier  
    - **Built with:** Python · Streamlit · Plotly  
    """)
    st.caption("Developed for MLOps Streamlit Lab — modified by Mihir Parab 🍇")
