"""FastAPI backend for authentication, dashboard data, and risk inference."""

from __future__ import annotations

import logging
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from backend.auth import (
    AuthResponse,
    LoginRequest,
    SignUpRequest,
    authenticate_user,
    decode_token,
    enforce_rate_limit,
    init_db,
    register_user,
    reset_rate_limit,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
MODEL_DIR = ROOT_DIR / "models"
LOGGER = logging.getLogger("supply_chain_api")
BEARER = HTTPBearer(auto_error=False)
RISK_TIERS = {
    0: "Low risk, low volume",
    1: "Standard performance",
    2: "High-priority risk",
    3: "High risk, low volume",
}


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shipping_mode: str = Field(min_length=1, max_length=80)
    order_region: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=100)
    scheduled_days: int = Field(ge=0, le=4)
    quantity: int = Field(ge=1, le=5)
    sales: float = Field(ge=10, le=2_000)
    discount_rate: float = Field(ge=0, le=0.25)


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    label: Literal["Lower risk", "Higher risk"]
    regional_risk_segment: Literal[
        "Low risk, low volume",
        "Standard performance",
        "High-priority risk",
        "High risk, low volume",
        "Unclassified",
    ]
    late_probability: float = Field(ge=0, le=1)
    estimated_shipping_days: float = Field(ge=0, le=6)
    schedule_variance_days: float = Field(ge=-4, le=6)
    regression_holdout_mae_days: float = Field(ge=0)
    classification_model: str
    regression_model: str
    caveat: str


@dataclass(frozen=True)
class Assets:
    orders: pd.DataFrame
    regions: pd.DataFrame
    model: object
    model_columns: list[str]
    duration_model: dict
    duration_metrics: dict
    scheduled_day_options: list[int]
    categories: dict[str, list[str]]
    baselines: dict[str, str]


@lru_cache(maxsize=1)
def load_assets() -> Assets:
    """Load and validate the packaged data and classifier once per process."""
    assets_to_check = {
        "cleaned orders": DATA_DIR / "cleaned_data.csv",
        "regional summary": DATA_DIR / "region_clusters.csv",
        "classifier": MODEL_DIR / "rf_delay_model.pkl",
        "classifier feature schema": MODEL_DIR / "model_columns.pkl",
        "duration regressor": MODEL_DIR / "ship_days_regressor.pkl",
        "duration metrics": MODEL_DIR / "ship_days_regressor_metrics.json",
    }
    missing = [name for name, path in assets_to_check.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required project assets: {', '.join(missing)}")

    orders = pd.read_csv(assets_to_check["cleaned orders"])
    regions = pd.read_csv(assets_to_check["regional summary"])
    model = joblib.load(assets_to_check["classifier"])
    model_columns = list(joblib.load(assets_to_check["classifier feature schema"]))
    duration_model = joblib.load(assets_to_check["duration regressor"])
    duration_metrics = json.loads(assets_to_check["duration metrics"].read_text(encoding="utf-8"))

    order_fields = {
        "Late_delivery_risk", "Sales", "Order Id", "Days for shipping (real)",
        "Days for shipment (scheduled)", "Shipping Mode", "Order Region",
        "Category Name", "Order Item Quantity", "Order Item Discount Rate",
    }
    region_fields = {"Order Region", "on_time_rate", "total_sales", "order_count", "Cluster"}
    missing_order_fields = order_fields.difference(orders.columns)
    missing_region_fields = region_fields.difference(regions.columns)
    if missing_order_fields or missing_region_fields:
        raise RuntimeError(
            "Data schema mismatch: "
            f"orders missing={sorted(missing_order_fields)}, "
            f"regions missing={sorted(missing_region_fields)}"
        )
    estimator_columns = list(getattr(model, "feature_names_in_", []))
    if estimator_columns and estimator_columns != model_columns:
        raise RuntimeError("Classifier feature order differs from the saved feature schema")
    if getattr(model, "n_features_in_", len(model_columns)) != len(model_columns):
        raise RuntimeError("Classifier feature count differs from the saved feature schema")
    if 1 not in list(getattr(model, "classes_", [])):
        raise RuntimeError("Classifier is missing the late-risk target class")
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    duration_features = [
        "Shipping Mode", "Order Region", "Category Name",
        "Days for shipment (scheduled)", "Order Item Quantity", "Sales",
        "Order Item Discount Rate",
    ]
    if duration_model.get("features") != duration_features or duration_model.get("target") != "Days for shipping (real)":
        raise RuntimeError("Duration regressor schema does not match its training contract")
    duration_estimator = duration_model["pipeline"].named_steps["regressor"]
    if hasattr(duration_estimator, "n_jobs"):
        duration_estimator.n_jobs = 1

    category_definitions = {
        "shipping_mode": ("Shipping Mode", "Shipping Mode_"),
        "order_region": ("Order Region", "Order Region_"),
        "category": ("Category Name", "Category Name_"),
    }
    categories: dict[str, list[str]] = {}
    baselines: dict[str, str] = {}
    for request_name, (column, feature_prefix) in category_definitions.items():
        values = set(orders[column].dropna().astype(str).unique())
        encoded = {
            feature[len(feature_prefix):]
            for feature in model_columns
            if feature.startswith(feature_prefix)
        }
        baseline = values - encoded
        if len(baseline) != 1 or encoded - values:
            raise RuntimeError(f"Classifier categorical feature mismatch for {column}")
        categories[request_name] = sorted(values)
        baselines[request_name] = next(iter(baseline))

    required_numeric = {
        "Order Item Quantity", "Sales", "Days for shipment (scheduled)",
        "Order Item Discount Rate",
    }
    if not required_numeric.issubset(model_columns):
        raise RuntimeError("Classifier feature schema is missing numeric inputs")
    scheduled_day_options = sorted(
        orders["Days for shipment (scheduled)"].dropna().astype(int).unique().tolist()
    )
    return Assets(
        orders=orders,
        regions=regions,
        model=model,
        model_columns=model_columns,
        duration_model=duration_model,
        duration_metrics=duration_metrics,
        scheduled_day_options=scheduled_day_options,
        categories=categories,
        baselines=baselines,
    )


async def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(BEARER),
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to continue",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(status_code=401, detail="Invalid sign-in session")
    return subject


app = FastAPI(
    title="Supply Chain Risk API",
    version="1.0.0",
    docs_url=None if os.getenv("RENDER") else "/docs",
    redoc_url=None if os.getenv("RENDER") else "/redoc",
    openapi_url=None if os.getenv("RENDER") else "/openapi.json",
)


@app.on_event("startup")
def startup_validation() -> None:
    init_db()
    assets = load_assets()
    LOGGER.info(
        "API ready: %s order rows; %s regions; %s classifier features",
        len(assets.orders), len(assets.regions), len(assets.model_columns),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "supply-chain-api"}


@app.post("/api/v1/auth/signup", response_model=AuthResponse, status_code=201)
def signup(request: SignUpRequest) -> AuthResponse:
    enforce_rate_limit(
        "signup-email",
        str(request.email),
        max_attempts=3,
    )
    enforce_rate_limit("signup-username", request.username, max_attempts=3)
    return register_user(request)


@app.post("/api/v1/auth/login", response_model=AuthResponse)
def login(request: LoginRequest) -> AuthResponse:
    identifier = request.username.strip().lower()
    enforce_rate_limit("login-identity", identifier, max_attempts=5)
    result = authenticate_user(request)
    reset_rate_limit("login-identity", identifier)
    return result


@app.get("/api/v1/auth/me")
async def auth_me(username: str = Depends(current_user)) -> dict[str, str]:
    return {"username": username}


@app.get("/api/v1/dashboard")
async def dashboard_data(_: str = Depends(current_user)) -> dict[str, object]:
    assets = load_assets()
    orders = assets.orders
    on_time_rate = float((orders["Late_delivery_risk"] == 0).mean() * 100)
    schedule_gap = orders["Days for shipping (real)"] - orders["Days for shipment (scheduled)"]
    regions = assets.regions.copy()
    regions["risk_tier"] = regions["Cluster"].map(RISK_TIERS).fillna("Unclassified")
    shipping = (
        orders.groupby("Shipping Mode")["Late_delivery_risk"]
        .apply(lambda values: float((values == 0).mean() * 100))
        .sort_values(ascending=False)
    )
    return {
        "kpis": {
            "on_time_rate": on_time_rate,
            "total_sales": float(orders["Sales"].sum()),
            "unique_orders": int(orders["Order Id"].nunique()),
            "average_schedule_gap_days": float(schedule_gap.mean()),
            "dataset_rows": int(len(orders)),
        },
        "shipping_performance": [
            {"shipping_mode": str(name), "on_time_rate": float(value)}
            for name, value in shipping.items()
        ],
        "regions": [
            {
                "order_region": str(row["Order Region"]),
                "on_time_rate": float(row["on_time_rate"]),
                "total_sales": float(row["total_sales"]),
                "order_count": int(row["order_count"]),
                "risk_tier": str(row["risk_tier"]),
            }
            for _, row in regions.iterrows()
        ],
        "categories": assets.categories,
        "scheduled_day_options": assets.scheduled_day_options,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_delay(
    request: PredictionRequest,
    username: str = Depends(current_user),
) -> PredictionResponse:
    assets = load_assets()
    selections = {
        "shipping_mode": request.shipping_mode,
        "order_region": request.order_region,
        "category": request.category,
    }
    for field, value in selections.items():
        if value not in assets.categories[field]:
            raise HTTPException(status_code=422, detail=f"Unknown {field.replace('_', ' ')}")
    if request.scheduled_days not in assets.scheduled_day_options:
        raise HTTPException(status_code=422, detail="Scheduled transit days are outside the training data values")
    matching_regions = assets.regions.loc[
        assets.regions["Order Region"].astype(str) == request.order_region,
        "Cluster",
    ]
    if matching_regions.empty:
        raise HTTPException(status_code=422, detail="No historical risk segment exists for this region")
    regional_risk_segment = RISK_TIERS.get(int(matching_regions.iloc[0]), "Unclassified")

    features = pd.DataFrame(0.0, index=[0], columns=assets.model_columns)
    numeric_values = [request.quantity, request.sales, request.scheduled_days, request.discount_rate]
    features.loc[0, [
        "Order Item Quantity", "Sales", "Days for shipment (scheduled)",
        "Order Item Discount Rate",
    ]] = numeric_values
    prefixes = {
        "shipping_mode": "Shipping Mode_",
        "order_region": "Order Region_",
        "category": "Category Name_",
    }
    for field, value in selections.items():
        if value == assets.baselines[field]:
            continue
        feature = prefixes[field] + value
        if feature not in features.columns:
            raise HTTPException(status_code=422, detail=f"Unsupported {field.replace('_', ' ')}")
        features.loc[0, feature] = 1.0

    try:
        prediction = int(assets.model.predict(features)[0])
        classes = list(assets.model.classes_)
        probability = float(assets.model.predict_proba(features)[0][classes.index(1)])
        regression_input = pd.DataFrame([{
            "Shipping Mode": request.shipping_mode,
            "Order Region": request.order_region,
            "Category Name": request.category,
            "Days for shipment (scheduled)": request.scheduled_days,
            "Order Item Quantity": request.quantity,
            "Sales": request.sales,
            "Order Item Discount Rate": request.discount_rate,
        }])
        estimated_shipping_days = float(
            assets.duration_model["pipeline"].predict(regression_input)[0]
        )
    except Exception as exc:
        LOGGER.exception("Model inference failed for authenticated request")
        raise HTTPException(status_code=500, detail="Prediction could not be completed") from exc
    LOGGER.info("Delay-risk prediction completed")
    return PredictionResponse(
        prediction=prediction,
        label="Higher risk" if prediction else "Lower risk",
        regional_risk_segment=regional_risk_segment,
        late_probability=probability,
        estimated_shipping_days=estimated_shipping_days,
        schedule_variance_days=estimated_shipping_days - request.scheduled_days,
        regression_holdout_mae_days=float(assets.duration_metrics["mae_days"]),
        classification_model="Random Forest classifier",
        regression_model="Random Forest regressor",
        caveat=(
            "Both values are estimates from historical data. The classifier probability is not calibrated; "
            f"the duration model's grouped holdout MAE was {assets.duration_metrics['mae_days']:.2f} days."
        ),
    )
