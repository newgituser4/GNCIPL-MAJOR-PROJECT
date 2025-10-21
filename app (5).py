
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- Page config ----------------
st.set_page_config(page_title="Ecommerce Purchase Behaviour", page_icon="🛍️", layout="centered")

st.title("Ecommerce Purchase Behaviour")
st.write("Enter customer details to predict the segment")

# ---------------- Model paths (relative) ----------------
MODEL_DIR = Path(__file__).parent / "models"
KMEANS_PATH = MODEL_DIR / "kmeans_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# ---------------- Cached loaders ----------------
@st.cache_resource(show_spinner=True)
def load_kmeans(path: Path):
    return joblib.load(path)

@st.cache_resource(show_spinner=True)
def load_scaler(path: Path):
    return joblib.load(path)

def get_artifacts():
    # Prefer repo models/, fall back to user uploads
    if KMEANS_PATH.exists() and SCALER_PATH.exists():
        return load_kmeans(KMEANS_PATH), load_scaler(SCALER_PATH)
    st.warning("Model files not found in ./models — upload them below or add them to your repo for seamless deploys.", icon="⚠️")
    kmeans_file = st.file_uploader("Upload kmeans_model.pkl", type=["pkl"], key="kmeans")
    scaler_file = st.file_uploader("Upload scaler.pkl", type=["pkl"], key="scaler")
    if kmeans_file and scaler_file:
        return joblib.load(kmeans_file), joblib.load(scaler_file)
    return None, None

# ---------------- Inputs ----------------
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age of Customer", min_value=18, max_value=100, value=40, step=1)
    income_label = st.selectbox("Income level of Customer", ["Middle", "High"])
    high_value = st.checkbox("High Value Customer", value=False)
with col2:
    purchase_amount = st.number_input("Purchase amount", min_value=0, max_value=10_000_000, value=100_000, step=1_000)
    frequency = st.number_input("Frequency of Purchase", min_value=0, max_value=1_000, value=10, step=1)
    overall_satisfaction = st.number_input("Overall Satisfaction (0–10)", min_value=0, max_value=10, value=4, step=1)

income_level = 0 if income_label == "Middle" else 1
high_value_int = 1 if high_value else 0

columns = [
    "Age",
    "Income_Level",
    "Purchase_Amount",
    "Frequency_of_Purchase",
    "High_Value_Customer",
    "Overall_Satisfaction",
]
input_df = pd.DataFrame(
    [{
        "Age": int(age),
        "Income_Level": int(income_level),
        "Purchase_Amount": float(purchase_amount),
        "Frequency_of_Purchase": int(frequency),
        "High_Value_Customer": int(high_value_int),
        "Overall_Satisfaction": int(overall_satisfaction),
    }],
    columns=columns,
)

# ---------------- Predict ----------------
kmeans, scaler = get_artifacts()

if kmeans is not None and scaler is not None:
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Scaling failed: {e}")
        st.stop()

    if st.button("Predict Segment"):
        try:
            cluster = kmeans.predict(input_scaled)
            label = int(cluster[0]) if hasattr(cluster, "__len__") else int(cluster)
            st.success(f"Predicted segment: {label}")
            st.json({"segment": label, "inputs": input_df.to_dict(orient="records")[0]})
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload kmeans_model.pkl and scaler.pkl or place them in a models/ folder next to this app.")
