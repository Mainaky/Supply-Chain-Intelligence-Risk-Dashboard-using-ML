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

# ---- Delivery and regional analysis ----
st.header("📈 Delivery Performance Analysis")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("On-Time Rate by Shipping Mode")
    shipping_otd = (
        df.groupby("Shipping Mode")["Late_delivery_risk"]
        .apply(lambda values: (values == 0).mean() * 100)
        .sort_values(ascending=False)
    )
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    shipping_otd.plot(kind="bar", ax=ax1, color="steelblue")
    ax1.set_ylabel("On-time orders (%)")
    ax1.axhline(on_time_rate, color="firebrick", linestyle="--", linewidth=1, label="Overall")
    ax1.legend()
    ax1.tick_params(axis="x", labelrotation=25)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with chart_col2:
    st.subheader("Regional Risk Segments")
    cluster_labels = {
        0: "Low risk, low volume",
        1: "Standard performance",
        2: "High-priority risk",
        3: "High risk, low volume",
    }
    cluster_colors = {
        "Low risk, low volume": "seagreen",
        "Standard performance": "slategray",
        "High-priority risk": "firebrick",
        "High risk, low volume": "darkorange",
    }
    regional_view = region_clusters.copy()
    regional_view["Risk Tier"] = regional_view["Cluster"].map(cluster_labels)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for tier, subset in regional_view.groupby("Risk Tier", dropna=False):
        ax2.scatter(
            subset["on_time_rate"],
            subset["total_sales"],
            label=tier or "Unclassified",
            color=cluster_colors.get(tier, "black"),
            s=55,
        )
    ax2.set_xlabel("On-time orders (%)")
    ax2.set_ylabel("Sales ($)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.header("🌍 Regional Risk Breakdown")
regional_view = regional_view.rename(columns={"on_time_rate": "On-time rate (%)"})
st.dataframe(
    regional_view[
        ["Order Region", "Risk Tier", "On-time rate (%)", "total_sales", "order_count"]
    ]
    .sort_values("total_sales", ascending=False)
    .style.format(
        {"On-time rate (%)": "{:.1f}%", "total_sales": "${:,.0f}", "order_count": "{:,.0f}"}
    ),
    height=320,
    use_container_width=True,
)

# ---- Interactive delay-risk prediction ----
st.header("🔮 Delay Risk Predictor")
st.write("Enter order details to estimate the chance of a late delivery.")

categorical_specs = {
    "Shipping Mode": "Shipping Mode_",
    "Order Region": "Order Region_",
    "Category Name": "Category Name_",
}
categorical_values = {
    field: set(df[field].dropna().astype(str).unique()) for field in categorical_specs
}
categorical_baselines = {}
for field, prefix in categorical_specs.items():
    encoded_values = {name[len(prefix):] for name in model_columns if name.startswith(prefix)}
    unencoded_values = categorical_values[field] - encoded_values
    unknown_encoded_values = encoded_values - categorical_values[field]
    if len(unencoded_values) != 1 or unknown_encoded_values:
        st.error(
            f"The saved model categories for {field} do not match the cleaned dataset. "
            "Refresh the model and feature schema together."
        )
        st.stop()
    categorical_baselines[field] = next(iter(unencoded_values))

numeric_features = [
    "Order Item Quantity",
    "Sales",
    "Days for shipment (scheduled)",
    "Order Item Discount Rate",
]
missing_model_inputs = sorted(set(numeric_features).difference(model_columns))
if missing_model_inputs:
    st.error(f"The saved model is missing input features: {', '.join(missing_model_inputs)}")
    st.stop()

pred_col1, pred_col2, pred_col3 = st.columns(3)
with pred_col1:
    shipping_mode = st.selectbox("Shipping Mode", sorted(categorical_values["Shipping Mode"]))
    order_region = st.selectbox("Order Region", sorted(categorical_values["Order Region"]))
with pred_col2:
    category = st.selectbox("Category", sorted(categorical_values["Category Name"]))
    scheduled_days = st.slider("Scheduled shipping days", min_value=0, max_value=5, value=2)
with pred_col3:
    quantity = st.number_input("Item quantity", min_value=1, max_value=10, value=1)
    sales = st.number_input("Sales ($)", min_value=0.0, value=200.0, step=10.0)
    discount = st.slider("Discount rate", min_value=0.0, max_value=0.5, value=0.1)

st.caption(
    "The displayed probability reflects this model's output for historical data; it is not a calibrated guarantee."
)

if st.button("Predict Delay Risk", type="primary"):
    input_df = pd.DataFrame(0.0, index=[0], columns=model_columns)
    input_df.loc[0, numeric_features] = [quantity, sales, scheduled_days, discount]

    selected_categories = {
        "Shipping Mode": shipping_mode,
        "Order Region": order_region,
        "Category Name": category,
    }
    for field, value in selected_categories.items():
        if value == categorical_baselines[field]:
            continue  # The training data used drop-first encoding for this baseline.
        encoded_column = f"{categorical_specs[field]}{value}"
        if encoded_column not in model_columns:
            st.error(f"The selected {field.lower()} is not represented by the saved model.")
            st.stop()
        input_df.loc[0, encoded_column] = 1.0

    classes = list(rf_model.classes_)
    if 1 not in classes:
        st.error("The saved model does not contain the late-delivery class (1).")
        st.stop()

    prediction = int(rf_model.predict(input_df)[0])
    late_probability = float(rf_model.predict_proba(input_df)[0][classes.index(1)])
    st.metric("Estimated chance of late delivery", f"{late_probability:.1%}")
    if prediction == 1:
        st.error("The model classifies this order as higher risk for late delivery.")
    else:
        st.success("The model classifies this order as lower risk for late delivery.")
