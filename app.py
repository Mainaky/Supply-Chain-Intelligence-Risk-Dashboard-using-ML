"""Streamlit presentation layer with user authentication, analytics, and ML inference."""

from __future__ import annotations

import os
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

st.markdown("""
<style>
:root { --ink:#152238; --muted:#718096; --line:#e5eaf1; --paper:#ffffff; --navy:#101d32; --teal:#13a89e; }
html, body, [class*="css"] { font-family:Inter, ui-sans-serif, system-ui, sans-serif; }
[data-testid="stAppViewContainer"] { background:#f4f7fb; }
[data-testid="stHeader"] { background:rgba(244,247,251,.84); }
.block-container { max-width:1440px; padding-top:2.2rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#101d32; border-right:1px solid rgba(255,255,255,.06); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:#dce5f1; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#94a3b8; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.12); }
[data-testid="stMetric"] { box-sizing:border-box; height:100%; min-height:126px; background:#fff; padding:1.15rem 1.25rem; border:1px solid #e7edf4; border-radius:16px; box-shadow:0 8px 24px rgba(21,34,56,.035); }
[data-testid="stMetricLabel"] { color:#718096; font-weight:600; min-height:2.5em; }
[data-testid="stMetricValue"] { color:#152238; font-family:Inter,ui-sans-serif,system-ui,sans-serif; font-weight:750; font-size:clamp(1.35rem,2.1vw,2rem); line-height:1.15; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; overflow-wrap:anywhere; }
[data-testid="stMetricValue"] > div, [data-testid="stMetricValue"] p { white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }
[data-testid="stMetricDelta"] { font-size:.82rem; }
button[kind="primary"] { background:linear-gradient(100deg,#147d82,#13a89e); border:0; border-radius:10px; }
[data-testid="stBaseButton-secondary"] { border-radius:10px; }
[data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #e5eaf1; }
[data-baseweb="tab"] { border-radius:8px 8px 0 0; }
[data-testid="stDataFrame"] { border:1px solid #e5eaf1; border-radius:14px; overflow:hidden; }
[data-testid="stForm"] { background:#fff; border:1px solid #e5eaf1; border-radius:16px; padding:24px; }
.brand { display:flex; align-items:center; gap:12px; padding:.35rem 0 1.4rem; }
.brand-icon { display:grid; place-items:center; width:42px; height:42px; border-radius:13px; color:#fff; font-size:21px; background:linear-gradient(140deg,#15b8a6,#3764b8); box-shadow:0 8px 20px rgba(19,168,158,.24); }
.brand-name { color:#f8fafc; font:800 15px Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:-.3px; line-height:1.2; }
.brand-sub { color:#8fa1b9; font-size:10px; letter-spacing:1.35px; text-transform:uppercase; margin-top:4px; }
.page-hero { position:relative; overflow:hidden; padding:28px 32px; margin:0 0 24px; border-radius:20px; color:white; background:linear-gradient(112deg,#10243a 0%,#173b51 58%,#126c6b 100%); box-shadow:0 16px 42px rgba(16,39,61,.12); }
.page-hero:after { content:''; position:absolute; width:270px; height:270px; right:-55px; top:-145px; border:1px solid rgba(255,255,255,.13); border-radius:50%; box-shadow:0 0 0 34px rgba(255,255,255,.035),0 0 0 70px rgba(255,255,255,.025); }
.hero-kicker { color:#8be1d2; text-transform:uppercase; letter-spacing:1.7px; font-size:10px; font-weight:700; margin-bottom:8px; }
.hero-title { color:#fff; font:800 29px Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:-.8px; margin:0; }
.hero-copy { color:#cfdae4; font-size:14px; margin:7px 0 0; max-width:720px; }
.section-title { color:#152238; font:750 18px Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:-.3px; margin:.2rem 0 .35rem; }
.section-copy { color:#718096; font-size:13px; margin-bottom:1rem; }
.info-card { height:100%; min-height:120px; background:white; border:1px solid #e7edf4; border-radius:15px; padding:18px 20px; box-shadow:0 8px 22px rgba(21,34,56,.03); }
.info-label { color:#8290a3; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:1.1px; }
.info-value { color:#152238; font:750 17px Inter,ui-sans-serif,system-ui,sans-serif; margin-top:9px; }
.info-note { color:#718096; font-size:12px; margin-top:5px; }
.auth-wrap { text-align:center; padding:2.1rem 1rem .6rem; }
.auth-kicker { color:#128f91; font-size:10px; letter-spacing:1.8px; font-weight:800; text-transform:uppercase; }
.auth-title { color:#152238; font:800 31px Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:-1px; margin:9px 0 6px; }
.auth-copy { color:#718096; font-size:14px; margin:0 auto; max-width:530px; }
.status-pill { color:#b7f2df; background:rgba(35,190,145,.12); border:1px solid rgba(105,226,187,.18); border-radius:100px; padding:5px 10px; font-size:11px; }
@media(max-width:720px){.block-container{padding:1.1rem 1rem 2rem}.page-hero{padding:22px 20px}.hero-title{font-size:23px}}
</style>
""", unsafe_allow_html=True)


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
    st.session_state.pop("last_risk_prediction", None)
    st.rerun()


# ── Authentication View ───────────────────────────────────────────────────────
def render_auth_page():
    """Renders the Login and Sign-Up authentication portal."""
    st.markdown("""
        <div class="auth-wrap">
            <div class="auth-kicker">Operations intelligence platform</div>
            <h1 class="auth-title">Supply Chain Intelligence</h1>
            <p class="auth-copy">A focused workspace for delivery performance, regional exposure, and machine-learning risk estimates.</p>
        </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        auth_tab1, auth_tab2 = st.tabs(["🔐 Sign In", "📝 Create Account (Sign Up)"])

        # ── TAB 1: LOGIN ──────────────────────────────────────────────
        with auth_tab1:
            st.markdown("### Welcome back")
            st.caption("Sign in to open your supply chain operations workspace.")
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
                                    st.session_state["workspace_navigation"] = "New shipment assessment"
                                    st.rerun()
                                else:
                                    st.error("Login failed. Check your username/email and password.")
                            except requests.RequestException:
                                st.error("Cannot reach the authentication service. Please try again shortly.")

        # ── TAB 2: SIGN UP ────────────────────────────────────────────
        with auth_tab2:
            st.markdown("### Create your workspace account")
            st.caption("Register an analyst account to access historical analytics and the risk predictor.")
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
                                st.session_state["workspace_navigation"] = "New shipment assessment"
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
        timeout=(30, 30),
    )
    response.raise_for_status()
    return response.json()


# ── Workspace shell and visual pages ──────────────────────────────────────────
def render_hero(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""<section class="page-hero"><div class="hero-kicker">{eyebrow}</div>
        <h1 class="hero-title">{title}</h1><p class="hero-copy">{description}</p></section>""",
        unsafe_allow_html=True,
    )


def render_sidebar(username: str, role: str) -> str:
    with st.sidebar:
        st.markdown(
            """<div class="brand"><div class="brand-icon">↗</div><div>
            <div class="brand-name">NEXUS / SUPPLY</div><div class="brand-sub">Intelligence workspace</div>
            </div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<span class='brand-sub'>WORKSPACE</span>", unsafe_allow_html=True)
        page = st.radio(
            "Workspace navigation",
            ["New shipment assessment", "Executive overview", "Delivery performance", "Regional intelligence"],
            label_visibility="collapsed",
            key="workspace_navigation",
        )
        st.divider()
        st.markdown("<span class='brand-sub'>SERVICE STATUS</span>", unsafe_allow_html=True)
        st.markdown("<span class='status-pill'>● &nbsp; Analytics API connected</span>", unsafe_allow_html=True)
        st.write("")
        st.divider()
        st.markdown(f"**{username}**")
        st.caption(role)
        if st.button("Sign out", use_container_width=True):
            logout()
        st.caption("Historical DataCo orders · classification + duration models")
    return page


def render_kpis(kpis: dict) -> None:
    columns = st.columns(4)
    metrics = [
        ("On-time delivery", f"{kpis['on_time_rate']:.1f}%", "Historical order-item delivery rate"),
        ("Sales represented", f"${kpis['total_sales'] / 1_000_000:.1f}M", "Recorded sales value in the dataset"),
        ("Distinct orders", f"{kpis['unique_orders']:,}", "Unique order IDs represented"),
        ("Average schedule gap", f"{kpis['average_schedule_gap_days']:+.2f} days", "Actual shipping days minus scheduled days"),
    ]
    for column, (label, value, help_text) in zip(columns, metrics):
        column.metric(label, value, help=help_text)


def render_shipping_chart(shipping: pd.DataFrame, overall_rate: float) -> None:
    ordered = shipping.sort_values("on_time_rate", ascending=True)
    st.bar_chart(
        ordered,
        x="shipping_mode",
        y="on_time_rate",
        horizontal=True,
        color="#128f91",
        x_label="Orders delivered on time (%)",
        y_label="",
        height=330,
        width="stretch",
    )
    st.caption(f"Network-wide historical benchmark: {overall_rate:.1f}% on time.")


def render_region_scatter(regions: pd.DataFrame) -> None:
    chart_data = regions.assign(sales_millions=regions["total_sales"] / 1_000_000)
    st.scatter_chart(
        chart_data,
        x="on_time_rate",
        y="sales_millions",
        color="risk_tier",
        size="order_count",
        x_label="On-time delivery rate (%)",
        y_label="Sales represented ($M)",
        height=460,
        width="stretch",
    )


def render_region_table(regions: pd.DataFrame, *, height: int = 440) -> None:
    table = regions.rename(
        columns={
            "order_region": "Region",
            "risk_tier": "Risk segment",
            "on_time_rate": "On-time rate (%)",
            "total_sales": "Sales ($)",
            "order_count": "Order rows",
        }
    ).sort_values("Sales ($)", ascending=False)
    st.dataframe(
        table.style.format(
            {"On-time rate (%)": "{:.1f}%", "Sales ($)": "${:,.0f}", "Order rows": "{:,.0f}"}
        ),
        height=height,
        width="stretch",
        hide_index=True,
    )


def render_overview(dashboard: dict) -> None:
    kpis = dashboard["kpis"]
    shipping = pd.DataFrame(dashboard["shipping_performance"])
    regions = pd.DataFrame(dashboard["regions"])
    render_hero(
        "SUPPLY CHAIN INTELLIGENCE · OPERATIONS BRIEF",
        "Executive overview",
        "A clear snapshot of historical delivery reliability, sales exposure, and shipment risk across the network.",
    )
    render_kpis(kpis)
    st.caption(f"Based on {kpis['dataset_rows']:,} historical order-item records · {len(regions)} regions · {shipping.shape[0]} shipping modes")
    st.markdown("<div class='section-title'>Delivery reliability by shipping mode</div><div class='section-copy'>Compare each service level with the network-wide historical on-time rate.</div>", unsafe_allow_html=True)
    render_shipping_chart(shipping, kpis["on_time_rate"])

    top_region = regions.sort_values("total_sales", ascending=False).iloc[0]
    top_mode = shipping.sort_values("on_time_rate", ascending=False).iloc[0]
    st.markdown("<div class='section-title'>Signals to review</div>", unsafe_allow_html=True)
    cards = st.columns(3)
    insights = [
        ("Strongest service level", str(top_mode["shipping_mode"]), f"{top_mode['on_time_rate']:.1f}% historical on-time rate"),
        ("Largest sales region", str(top_region["order_region"]), f"${top_region['total_sales'] / 1_000_000:.2f}M in recorded sales"),
        ("Schedule variance", f"{kpis['average_schedule_gap_days']:+.2f} days", "Positive values mean actual shipping took longer on average"),
    ]
    for column, (label, value, note) in zip(cards, insights):
        column.markdown(f"<div class='info-card'><div class='info-label'>{label}</div><div class='info-value'>{value}</div><div class='info-note'>{note}</div></div>", unsafe_allow_html=True)


def render_delivery_page(dashboard: dict) -> None:
    kpis = dashboard["kpis"]
    shipping = pd.DataFrame(dashboard["shipping_performance"])
    highest = shipping.sort_values("on_time_rate", ascending=False).iloc[0]
    lowest = shipping.sort_values("on_time_rate").iloc[0]
    render_hero(
        "SERVICE-LEVEL ANALYSIS",
        "Delivery performance",
        "Review how shipping modes have performed against promised delivery schedules in the historical dataset.",
    )
    metrics = st.columns(3)
    metrics[0].metric("Network on-time rate", f"{kpis['on_time_rate']:.1f}%")
    metrics[1].metric("Highest mode rate", f"{highest['on_time_rate']:.1f}%", str(highest["shipping_mode"]))
    metrics[2].metric("Lowest mode rate", f"{lowest['on_time_rate']:.1f}%", str(lowest["shipping_mode"]))
    st.markdown("<div class='section-title'>On-time rate by service level</div><div class='section-copy'>Compare historical service-mode performance with the network benchmark shown below the chart.</div>", unsafe_allow_html=True)
    render_shipping_chart(shipping, kpis["on_time_rate"])
    st.markdown("<div class='section-title'>Service-level detail</div>", unsafe_allow_html=True)
    table = shipping.rename(columns={"shipping_mode": "Shipping mode", "on_time_rate": "On-time rate (%)"}).sort_values("On-time rate (%)", ascending=False)
    st.dataframe(table.style.format({"On-time rate (%)": "{:.1f}%"}), width="stretch", hide_index=True)
    st.info("These are descriptive historical results. They do not establish that the shipping mode caused the delivery outcome.")


def render_regions_page(dashboard: dict) -> None:
    regions = pd.DataFrame(dashboard["regions"])
    render_hero(
        "GEOGRAPHIC EXPOSURE · RISK SEGMENTS",
        "Regional intelligence",
        "Explore the relationship between recorded sales exposure and delivery reliability across destination regions.",
    )
    st.markdown("<div class='section-title'>Sales exposure and delivery reliability</div><div class='section-copy'>Each point represents a region; color indicates the existing cluster segment.</div>", unsafe_allow_html=True)
    render_region_scatter(regions)
    all_tiers = sorted(regions["risk_tier"].dropna().unique().tolist())
    selected_tiers = st.multiselect("Filter risk segments", all_tiers, default=all_tiers)
    filtered = regions[regions["risk_tier"].isin(selected_tiers)]
    st.markdown(f"<div class='section-title'>Regional breakdown <span style='color:#8290a3;font-size:13px;font-weight:500'>· {len(filtered)} of {len(regions)} regions</span></div>", unsafe_allow_html=True)
    render_region_table(filtered)


def render_predictor_page(dashboard: dict, token: str) -> None:
    render_hero(
        "NEW SHIPMENT · CLASSIFICATION + REGRESSION",
        "Shipment assessment",
        "Enter the person requesting the assessment and the order profile. The system returns a late-risk class and an estimated shipping duration.",
    )
    categories = dashboard["categories"]
    with st.form("risk_prediction_form", border=True):
        st.markdown("<div class='section-title'>Assessment details</div><div class='section-copy'>The requester name labels this assessment. It is not used as a model feature.</div>", unsafe_allow_html=True)
        requester_name = st.text_input(
            "Person requesting this assessment",
            placeholder="Enter the analyst or planner's name",
            max_chars=80,
            key="prediction_requester_name",
        )
        st.markdown("<div class='section-title'>Order profile</div><div class='section-copy'>Enter values within the ranges represented in the historical training data.</div>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        shipping_mode = col_a.selectbox("Shipping mode", categories["shipping_mode"], key="predict_shipping")
        order_region = col_a.selectbox("Destination region", categories["order_region"], key="predict_region")
        category = col_b.selectbox("Product category", categories["category"], key="predict_category")
        scheduled_options = dashboard["scheduled_day_options"]
        default_schedule_index = scheduled_options.index(4) if 4 in scheduled_options else 0
        scheduled_days = col_b.selectbox("Scheduled shipping days", scheduled_options, index=default_schedule_index, key="predict_schedule")
        quantity = col_c.number_input("Item quantity", min_value=1, max_value=5, value=1, key="predict_quantity")
        sales_amount = col_c.number_input("Sales value ($)", min_value=10.0, max_value=2000.0, value=250.0, step=10.0, key="predict_sales")
        discount = st.slider("Item discount rate", 0.0, 0.25, 0.1, step=0.01, key="predict_discount")
        submitted = st.form_submit_button("Run classification + duration estimate", type="primary", use_container_width=True)

    if submitted:
        if not requester_name.strip():
            st.error("Enter the person requesting this assessment before running the models.")
            st.session_state.pop("last_risk_prediction", None)
        else:
            payload = {
                "shipping_mode": shipping_mode,
                "order_region": order_region,
                "category": category,
                "scheduled_days": int(scheduled_days),
                "quantity": int(quantity),
                "sales": float(sales_amount),
                "discount_rate": float(discount),
            }
            try:
                with st.spinner("Running the classification and duration models…"):
                    response = requests.post(
                        f"{BASE_URL}/api/v1/predict",
                        json=payload,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=(30, 30),
                    )
                response.raise_for_status()
                st.session_state["last_risk_prediction"] = {
                    **response.json(),
                    "requester_name": requester_name.strip(),
                }
            except requests.RequestException as exc:
                st.session_state.pop("last_risk_prediction", None)
                detail = "The prediction service could not complete this request. Please try again."
                if getattr(exc, "response", None) is not None and exc.response is not None:
                    try:
                        detail = exc.response.json().get("detail", detail)
                    except ValueError:
                        pass
                st.error(detail)

    result = st.session_state.get("last_risk_prediction")
    if result:
        st.markdown("<div class='section-title'>Assessment result</div>", unsafe_allow_html=True)
        st.write(f"Prepared for **{result['requester_name']}**")
        cols = st.columns(4)
        class_text = "Class 1 — Higher risk" if result["prediction"] == 1 else "Class 0 — Lower risk"
        schedule_difference = float(result["schedule_variance_days"])
        if schedule_difference > 0.05:
            schedule_text = f"{schedule_difference:.1f} days later"
        elif schedule_difference < -0.05:
            schedule_text = f"{abs(schedule_difference):.1f} days earlier"
        else:
            schedule_text = "On schedule"
        cols[0].metric("Late-delivery class", class_text)
        cols[1].metric("Late-risk probability", f"{result['late_probability']:.1%}")
        cols[2].metric("Estimated shipping duration", f"{result['estimated_shipping_days']:.1f} days")
        cols[3].metric("Schedule comparison", schedule_text)
        if result["prediction"] == 1:
            st.warning("The classifier assigned **Class 1**: the historical `Late_delivery_risk` target is flagged.")
        else:
            st.success("The classifier assigned **Class 0**: the historical `Late_delivery_risk` target is not flagged.")
        st.info(
            f"Stored destination risk segment: **{result['regional_risk_segment']}**. "
            "This four-segment regional profile comes from the project's precomputed region clustering; it is separate from the per-order binary classifier."
        )
        st.caption(
            f"Classification: {result['classification_model']} · Regression: {result['regression_model']} · "
            f"Grouped holdout MAE: {result['regression_holdout_mae_days']:.2f} days. "
            "The regression value estimates actual shipping duration; the schedule comparison describes it relative to the promised duration. "
            "The class and duration models can disagree; the probability is not calibrated and neither output is an operational guarantee."
        )


def render_dashboard() -> None:
    user = st.session_state.get("user_info", {})
    username = user.get("username", "Analyst")
    role = user.get("role", "Supply Chain Analyst")
    token = st.session_state.get("auth_token")
    if not token:
        logout()

    page = render_sidebar(username, role)
    try:
        dashboard = fetch_dashboard(BASE_URL, token)
    except requests.RequestException:
        st.error("The analytics API is not responding. Retry after checking that the backend is running.")
        if st.button("Retry connection", type="primary"):
            st.rerun()
        st.stop()

    pages = {
        "New shipment assessment": lambda: render_predictor_page(dashboard, token),
        "Executive overview": lambda: render_overview(dashboard),
        "Delivery performance": lambda: render_delivery_page(dashboard),
        "Regional intelligence": lambda: render_regions_page(dashboard),
    }
    pages[page]()


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    if not st.session_state["authenticated"]:
        render_auth_page()
    else:
        token = st.session_state.get("auth_token")
        try:
            response = requests.get(
                f"{BASE_URL}/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"},
                timeout=(30, 10),
            )
        except requests.RequestException:
            st.error("Could not verify your sign-in session. Please retry shortly.")
            st.stop()
        if response.status_code == 401:
            st.session_state["authenticated"] = False
            st.session_state["auth_token"] = None
            st.session_state["user_info"] = {}
            st.info("Your session expired. Please sign in again.")
            render_auth_page()
            return
        if not response.ok:
            st.error("Could not verify your sign-in session.")
            st.stop()
        st.session_state["user_info"]["username"] = response.json()["username"]
        render_dashboard()


if __name__ == "__main__":
    main()
