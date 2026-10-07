from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Supply Chain Intelligence Dashboard", layout="wide")
st.title("📦 Supply Chain Intelligence & Risk Dashboard")
st.caption("Explore historical delivery performance, regional exposure, and order-level delay risk.")

# Load project assets relative to this file, not the shell's current directory.
df = pd.read_csv(DATA_DIR / "cleaned_data.csv")
region_clusters = pd.read_csv(DATA_DIR / "region_clusters.csv")
rf_model = joblib.load(MODEL_DIR / "rf_delay_model.pkl")
model_columns = list(joblib.load(MODEL_DIR / "model_columns.pkl"))

required_columns = {
    "Late_delivery_risk",
    "Sales",
    "Order Id",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Order Region",
    "Category Name",
    "Order Item Quantity",
    "Order Item Discount Rate",
}
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    st.error(f"The cleaned dataset is missing required columns: {', '.join(missing_columns)}")
    st.stop()

model_features = list(getattr(rf_model, "feature_names_in_", []))
if model_features and model_features != model_columns:
    st.error("The saved model and model_columns.pkl do not have the same feature order.")
    st.stop()

# ---- KPI overview ----
st.header("📊 Key Performance Indicators")
on_time_rate = (df["Late_delivery_risk"] == 0).mean() * 100
total_sales = df["Sales"].sum()
total_orders = df["Order Id"].nunique()
mean_schedule_gap = (
    df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
).mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")
col2.metric("Total Sales", f"${total_sales / 1_000_000:.1f}M")
col3.metric("Unique Orders", f"{total_orders:,}")
col4.metric("Average Gap vs. Schedule", f"{mean_schedule_gap:+.2f} days")
st.caption("A positive schedule gap means shipments were late on average; a negative value means they arrived early.")
