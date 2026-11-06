"""Streamlit presentation layer with user authentication, analytics, and ML inference."""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Intelligence & Risk Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_configuration() -> str:
    """Return backend base URL; sessions use short-lived user JWTs."""
    explicit_url = os.getenv("API_BASE_URL")
    if explicit_url:
        base_url = explicit_url.rstrip("/")
    else:
        host = os.getenv("API_HOST", "127.0.0.1")
        port = os.getenv("API_PORT", "8000")
        base_url = f"http://{host}:{port}"
    return base_url


BASE_URL = api_configuration()


# ── Session State Initialization ──────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = None
if "user_info" not in st.session_state:
    st.session_state["user_info"] = {}


def logout():
    """Clear session authentication state."""
    st.session_state["authenticated"] = False
    st.session_state["auth_token"] = None
    st.session_state["user_info"] = {}
    st.rerun()


# ── Authentication View ───────────────────────────────────────────────────────
def render_auth_page():
    """Renders the Login and Sign-Up authentication portal."""
    st.markdown("""
        <div style="text-align: center; margin-top: 20px; margin-bottom: 25px;">
            <h1 style="font-size: 2.4rem; font-weight: 700; color: #1E293B;">📦 Supply Chain Intelligence Console</h1>
            <p style="font-size: 1.1rem; color: #64748B;">
                Production-grade delay prediction, delivery risk analytics, and global operations monitoring
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        auth_tab1, auth_tab2 = st.tabs(["🔐 Sign In", "📝 Create Account (Sign Up)"])

        # ── TAB 1: LOGIN ──────────────────────────────────────────────
        with auth_tab1:
            st.markdown("### Access Operations Portal")
            login_user = st.text_input("Username or Email", key="login_username_input", placeholder="e.g. mainak")
            login_pass = st.text_input("Password", type="password", key="login_pass_input", placeholder="Enter your password")

            if st.button("Sign In", type="primary", use_container_width=True):
                    if not login_user or not login_pass:
                        st.error("Please provide both username/email and password.")
                    else:
                        with st.spinner("Authenticating..."):
                            try:
                                resp = requests.post(
                                    f"{BASE_URL}/api/v1/auth/login",
                                    json={"username": login_user, "password": login_pass},
                                    timeout=(30, 30),
                                )
                                if resp.status_code == 200:
                                    data = resp.json()
                                    st.session_state["authenticated"] = True
                                    st.session_state["auth_token"] = data["access_token"]
                                    st.session_state["user_info"] = {
                                        "username": data["username"],
                                        "email": data.get("email", ""),
                                        "role": data.get("role", "Supply Chain Analyst"),
                                    }
                                    st.rerun()
                                else:
                                    st.error("Login failed. Check your username/email and password.")
                            except requests.RequestException:
                                st.error("Cannot reach the authentication service. Please try again shortly.")

        # ── TAB 2: SIGN UP ────────────────────────────────────────────
        with auth_tab2:
            st.markdown("### Register New Analyst Account")
            new_user = st.text_input("Choose Username", key="signup_user_input", placeholder="alphanumeric, min 3 chars")
            new_email = st.text_input("Work Email", key="signup_email_input", placeholder="name@company.com")
            new_pass = st.text_input("Create Password", type="password", key="signup_pass_input", placeholder="at least 12 characters")
            confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm_input", placeholder="re-enter password")

            if st.button("Create Account & Access", type="primary", use_container_width=True):
                if not new_user or not new_email or not new_pass:
                    st.error("All fields are required.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 12:
                    st.error("Password must be at least 12 characters.")
                else:
                    with st.spinner("Creating account..."):
                        try:
                            resp = requests.post(
                                f"{BASE_URL}/api/v1/auth/signup",
                                json={"username": new_user, "email": new_email, "password": new_pass},
                                timeout=(30, 30),
                            )
                            if resp.status_code == 201:
                                data = resp.json()
                                st.session_state["authenticated"] = True
                                st.session_state["auth_token"] = data["access_token"]
                                st.session_state["user_info"] = {
                                    "username": data["username"],
                                    "email": "",
                                    "role": data.get("role", "Supply Chain Analyst"),
                                }
                                st.rerun()
                            else:
                                err_msg = resp.json().get("detail", "Signup could not be completed.")
                                st.error(f"Registration error: {err_msg}")
                        except requests.RequestException:
                            st.error("Cannot reach registration service. Please verify backend connectivity.")


# ── Dashboard Data Fetching ───────────────────────────────────────────────────
def fetch_dashboard(base_url: str, token: str) -> dict:
    response = requests.get(
        f"{base_url}/api/v1/dashboard",
        headers={"Authorization": f"Bearer {token}"},
        timeout=(3.05, 25),
    )
    response.raise_for_status()
    return response.json()


# ── Main Dashboard Application View ───────────────────────────────────────────
def render_dashboard():
    user = st.session_state.get("user_info", {})
    username = user.get("username", "Analyst")
    role = user.get("role", "Supply Chain Analyst")
    active_token = st.session_state.get("auth_token")
    if not active_token:
        logout()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👤 Active Session")
        st.markdown(f"**User:** `{username}`")
        st.caption(f"Role: {role}")
        st.markdown("---")

        st.markdown("### 🌐 System Status")
        st.success("🟢 API Connected")
        st.markdown("---")

        if st.button("🚪 Sign Out", use_container_width=True):
            logout()

        st.markdown("---")
        st.markdown("""
        **About This Platform:**  
        • **Dataset:** 180,516 DataCo Orders  
        • **Model:** Random Forest (78 Features)  
        • **Accuracy / ROC-AUC:** 0.73  
        • **Target:** `Late_delivery_risk`
        """)

    # ── Top Banner ────────────────────────────────────────────────────────────
    st.markdown(f"""
        <div style="background-color: #F8FAFC; border-left: 4px solid #3B82F6; padding: 12px 18px; border-radius: 6px; margin-bottom: 20px;">
            <span style="font-size: 1.3rem; font-weight: 600; color: #1E293B;">📦 Supply Chain Intelligence & Risk Operations</span>
            <span style="float: right; color: #64748B; font-size: 0.95rem;">Authenticated as: <b>{username}</b> ({role})</span>
        </div>
    """, unsafe_allow_html=True)

    # ── Data Ingestion from FastAPI ────────────────────────────────────────────
    try:
        dashboard = fetch_dashboard(BASE_URL, active_token)
    except requests.RequestException:
        st.error(
            "⚠️ The analytics backend service is not responding. "
            "If deployed on Render (free tier), the web service may be waking from standby. "
            "Please refresh in 30 seconds."
        )
        if st.button("🔄 Retry Connection"):
            st.rerun()
        st.stop()

    # ── Section 1: Executive KPI Cards ────────────────────────────────────────
    kpis = dashboard["kpis"]
    st.subheader("📊 Executive Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("On-Time Delivery Rate", f"{kpis['on_time_rate']:.1f}%", help="Percentage of total historical orders delivered on or before scheduled date")
    col2.metric("Total Revenue Exposure", f"${kpis['total_sales'] / 1_000_000:.1f}M", help="Aggregated historical sales volume processed")
    col3.metric("Distinct Orders Analyzed", f"{kpis['unique_orders']:,}", help="Unique global purchase transactions evaluated")
    col4.metric("Avg. Schedule Gap", f"{kpis['average_schedule_gap_days']:+.2f} days", delta_color="inverse", help="Positive value denotes real shipping duration exceeded scheduled commitment")

    st.caption(f"Evaluated on {kpis['dataset_rows']:,} verified order records.")
    st.divider()

    # ── Section 2: Visual Performance Analytics ───────────────────────────────
    st.subheader("📈 Delivery Performance & Regional Risk Analytics")
    chart_col1, chart_col2 = st.columns(2)

    shipping = pd.DataFrame(dashboard["shipping_performance"])
    regions = pd.DataFrame(dashboard["regions"])
    colors = {
        "Low risk, low volume": "#10B981",    # Emerald
        "Standard performance": "#64748B",    # Slate
        "High-priority risk": "#EF4444",      # Crimson
        "High risk, low volume": "#F59E0B",   # Amber
        "Unclassified": "#000000",
    }

    with chart_col1:
        st.markdown("**On-Time Delivery Rate by Shipping Mode**")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        bars = ax.bar(shipping["shipping_mode"], shipping["on_time_rate"], color="#3B82F6", edgecolor="#1D4ED8", width=0.55)
        ax.axhline(kpis["on_time_rate"], color="#EF4444", linestyle="--", linewidth=1.5, label=f"Average ({kpis['on_time_rate']:.1f}%)")
        ax.set_ylabel("On-time Rate (%)", fontsize=10)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", labelrotation=20, labelsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", loc="upper right")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        st.markdown("**Regional Risk Clusters: Sales Exposure vs. On-Time Reliability**")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for tier, group in regions.groupby("risk_tier"):
            ax.scatter(
                group["on_time_rate"], group["total_sales"] / 1_000_000,
                label=tier, color=colors.get(tier, "black"), s=70, alpha=0.85, edgecolors="white", linewidth=0.5
            )
        ax.set_xlabel("On-Time Rate (%)", fontsize=10)
        ax.set_ylabel("Sales ($ Millions)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, frameon=True, facecolor="white", loc="upper right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ── Section 3: Regional Risk Explorer ─────────────────────────────────────
    st.subheader("🌍 Regional Exposure Breakdown Table")
    st.dataframe(
        regions.rename(
            columns={
                "order_region": "Region",
                "risk_tier": "Risk Segment",
                "on_time_rate": "On-Time Rate (%)",
                "total_sales": "Sales ($)",
                "order_count": "Order Rows",
            }
        ).sort_values("Sales ($)", ascending=False).style.format(
            {"On-Time Rate (%)": "{:.1f}%", "Sales ($)": "${:,.0f}", "Order Rows": "{:,.0f}"}
        ),
        height=300,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Section 4: Real-Time Delay Risk Predictor ──────────────────────────────
    st.subheader("🔮 Real-Time Order Delay Risk Predictor (ML Inference)")
    st.write("Configure prospective order attributes to evaluate late-delivery risk using the production Random Forest classifier.")

    categories = dashboard["categories"]
    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        shipping_mode = st.selectbox("Shipping Mode", categories["shipping_mode"], help="Logistics priority selected for transit")
        order_region = st.selectbox("Destination Region", categories["order_region"], help="Geographic target territory")

    with col_in2:
        category = st.selectbox("Product Line", categories["category"], help="Inventory catalog classification")
        scheduled_days = st.slider("Scheduled Transit Days", 0, 5, 2, help="Promised delivery schedule window")

    with col_in3:
        quantity = st.number_input("Item Quantity", min_value=1, max_value=20, value=1)
        sales_amount = st.number_input("Order Sales Value ($)", min_value=5.0, max_value=50000.0, value=250.0, step=25.0)
        discount = st.slider("Applied Discount Rate", 0.0, 0.5, 0.1, step=0.01)

    if st.button("🚀 Predict Shipment Delay Risk", type="primary", use_container_width=True):
        payload = {
            "shipping_mode": shipping_mode,
            "order_region": order_region,
            "category": category,
            "scheduled_days": int(scheduled_days),
            "quantity": int(quantity),
            "sales": float(sales_amount),
            "discount_rate": float(discount),
        }

        with st.spinner("Invoking Random Forest Model via FastAPI..."):
            try:
                response = requests.post(
                    f"{BASE_URL}/api/v1/predict",
                    json=payload,
                    headers={"Authorization": f"Bearer {active_token}"},
                    timeout=(3.05, 30),
                )
                if response.status_code == 422:
                    st.error("Validation error: Invalid order parameter payload.")
                    return
                response.raise_for_status()
                result = response.json()

                # Display Results
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric("Estimated Probability of Late Delivery", f"{result['late_probability']:.1%}")

                with res_col2:
                    if result["prediction"] == 1:
                        st.error("⚠️ **HIGHER RISK OF DELAY DETECTED**\n\nThe classifier estimates this order has an elevated likelihood of exceeding scheduled shipping days. Consider upgrading to Expedited Carrier dispatch or prioritizing warehouse fulfillment.")
                    else:
                        st.success("✅ **LOWER RISK (ON-TIME EXPECTED)**\n\nOrder parameters align with historical patterns that complete within scheduled shipping delivery commitments.")

                st.caption(f"Model: {result.get('model', 'Random Forest')} | {result.get('caveat', '')}")

            except requests.RequestException as err:
                st.error("Inference request failed. Please verify API availability.")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    if not st.session_state["authenticated"]:
        render_auth_page()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
