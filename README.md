# Supply Chain Intelligence & Risk Dashboard

A two-service supply-chain analytics application. The Streamlit workspace opens on a shipment assessment form, while executive and regional analytics remain separate navigation pages. FastAPI returns a binary late-risk classification and a per-order shipping-duration regression estimate. User accounts use Argon2 password hashes and signed, expiring JWT sessions; Render Postgres stores accounts persistently in deployment.

## Architecture

```text
Browser
  └── Streamlit dashboard
        ├── signup/sign-in ──> FastAPI authentication endpoints
        ├── dashboard data ──> FastAPI analytics endpoint
        └── shipment assessment ──> FastAPI prediction endpoint
                                      ├── Random Forest late-risk classifier
                                      ├── Random Forest shipping-duration regressor
                                      └── Render Postgres user store
```

The Streamlit server calls the API over Render's private network. The FastAPI service is configured as a private service, so it has no public URL; the browser does not call the model API directly. Dashboard and prediction routes require a signed bearer token. `/health` is available on the internal network; interactive API docs are enabled locally and disabled on Render.

## What it shows

- A shipment assessment form is the first workspace screen after sign-in; the executive overview, delivery performance, and regional intelligence remain separate navigation pages.
- On-time delivery rate, total sales, unique orders, and average difference between actual and scheduled shipping days.
- On-time performance by shipping mode and regional risk segments compared with sales exposure.
- Order-level classification of historical `Late_delivery_risk` as class 0 or 1, a separate stored destination segment from the four regional clusters, and regression of `Days for shipping (real)` with its signed difference from schedule.
- Account creation, sign-in, sign-out, and 30-minute signed access tokens.

The included data contains 180,516 order-item rows and 65,749 distinct order IDs. It is historical DataCo data, not a live order feed. The classifier predicts a binary late-risk label; the separate regressor estimates actual shipping duration in days. A destination's four-way risk segment is looked up from the repository's precomputed regional clusters; it is not a multi-class order prediction. The duration model's order-grouped holdout MAE is 0.98 days (R² 0.40). The classifier probability is not documented as calibrated.

## Run locally

Use Python 3.11.10. Create a virtual environment and install all dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:JWT_SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"
$env:LOCAL_AUTH_DB_PATH = "$PWD\Data\auth_local.db"
```

Start the API in terminal 1:

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Start the dashboard in terminal 2:

```powershell
$env:API_HOST = "127.0.0.1"
$env:API_PORT = "8000"
streamlit run app.py
```

Open `http://localhost:8501`. After sign-in, the first page is the shipment assessment form; use the sidebar to open the analytics dashboard. Local accounts use SQLite at the ignored `Data/auth_local.db`; when running on Render, the API requires `DATABASE_URL` and uses managed Postgres instead. Do not reuse a production JWT secret locally or commit one to Git.

## API routes

- `POST /api/v1/auth/signup` — create a username/email account; passwords must be at least 12 characters.
- `POST /api/v1/auth/login` — verify username or email and password; returns a signed JWT with a 30-minute expiry.
- `GET /api/v1/auth/me` — verify the current token and return its user identity.
- `GET /api/v1/dashboard` — KPIs, shipping-mode performance, regional data, and allowed prediction categories.
- `POST /api/v1/predict` — validate order inputs and return the binary classifier class/probability, stored regional risk segment, estimated shipping days, and schedule variance.

## Deploy on Render

[`render.yaml`](render.yaml) defines two Starter web services in Singapore plus a paid Postgres database:

1. `supply-chain-api` runs FastAPI, generates a JWT signing secret, and connects to Postgres.
2. `supply-chain-dashboard` runs Streamlit and receives the API's private hostname and port from Render.
3. `supply-chain-users` stores account records using a managed database connection string.

To deploy, connect the GitHub repository to **New → Blueprint** in Render, inspect the service/database plans and current charges, then apply the Blueprint. The API is private and the dashboard connects to its internal hostname. Render's free Postgres databases expire after 30 days and do not provide backups, so the Blueprint uses a paid persistent database. Review the costs in Render before applying; no Render service has been created by this repository change. ([Blueprint reference](https://render.com/docs/blueprint-spec), [private services](https://render.com/docs/private-services), [Postgres connections](https://render.com/docs/postgresql-creating-connecting))

## Important files

```text
backend/main.py              FastAPI routes, validation, model loading, prediction
backend/auth.py              Argon2 password hashing, JWT issue/verification, DB access
backend/train_duration_model.py  Grouped train/holdout pipeline for duration regression
app.py                       Streamlit UI and account/session handling
Data/cleaned_data.csv        Cleaned historical order rows
Data/region_clusters.csv     Regional summaries and precomputed cluster IDs
models/rf_delay_model.pkl    Saved Random Forest classifier
models/ship_days_regressor.pkl  Per-order shipping-duration regression model
models/ship_days_regressor_metrics.json  Grouped holdout MAE, RMSE, and R²
models/region_clusters.csv     Precomputed four-segment regional classifications
models/model_columns.pkl     Ordered classifier feature schema
models/prophet_model.pkl     Forecast artifact; not used in the running app
models/kmeans_model (1).pkl  Clustering artifact; cluster results are precomputed
models/scaler.pkl            Scaler artifact; not used in the running app
requirements*.txt             Pinned API, dashboard, and common dependencies
render.yaml                  Render services, Postgres, secrets, and health checks
.python-version              Python runtime version
.streamlit/config.toml       Streamlit server settings
```

## Model and data limitations

- The source is the [DataCo Smart Supply Chain dataset on Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis). The repository includes a reduced historical extract, not the raw dataset or live operational data.
- The original classifier's training script and split/evaluation process are absent, so its reported 0.73 ROC-AUC cannot be reproduced from the repository alone. The new duration regressor is reproducible from the packaged CSV with an 80/20 split grouped by `Order Id`.
- The new per-order regressor estimates `Days for shipping (real)` with about 0.98 days MAE on that grouped holdout. Treat it as a rough historical estimate: the available features do not explain all carrier, route, or operational variability.
- The per-order classifier remains binary. The four regional segments are contextual metadata for the selected destination and should not be described as four classes predicted for an individual order.
- The Prophet artifact is not used by the per-order assessment; it represents a separate aggregate forecast workflow. The running UI does not currently render that forecast or SHAP explanations.
- User accounts are stored persistently in Render Postgres; a new deployment does not silently fall back to an ephemeral local database.

## Security and production boundary

This version uses Argon2 password hashing, a required random JWT secret, issuer/audience/expiry/signature checks, input validation, database-backed signup/login throttles, and persistent Postgres on Render. Sign-up does not verify email ownership; there is no password reset, MFA, account administration, or edge-level abuse monitoring. Add those controls and independent security review before using this as an operational system with real employees or sensitive data. The model itself is also not validated for current operational decisions.

## Screenshots

### Dashboard overview
![Dashboard overview](screenshots/dashboard_overview.png)

### Delivery performance
![Delivery analysis](screenshots/delivery_analysis.png)

### Delay risk predictor
![Delay predictor](screenshots/delay_predictor_1.png)

## Author

Vinayak Nair
