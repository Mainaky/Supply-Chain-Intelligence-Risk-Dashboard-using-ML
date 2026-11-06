# Supply Chain Intelligence & Risk Management Platform

An enterprise-grade, full-stack predictive supply chain analytics application. Featuring a modern **Streamlit frontend** with complete user authentication (Sign Up / Sign In / Demo access) and a high-performance **FastAPI microservice backend** powering real-time delivery performance analytics and Random Forest delay risk inference.

---

## 🏛️ System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │              CLIENT / WEB BROWSER                      │
               │  Streamlit UI (Port 8501 / Render Web Service)         │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  HTTPS Requests / Bearer JWT Authentication
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │              BACKEND REST MICROSERVICE                 │
               │  FastAPI (Port 8000 / Render Web Service)              │
               ├──────────────────────────┬─────────────────────────────┤
               │  • /api/v1/auth/signup   │  • /api/v1/dashboard        │
               │  • /api/v1/auth/login    │  • /api/v1/predict          │
               │  • /api/v1/auth/me       │  • /health & /docs          │
               └─────────────┬────────────────────────────┬─────────────┘
                             │                            │
                             ▼                            ▼
               ┌──────────────────────────┐  ┌──────────────────────────┐
               │    SQLITE AUTH STORE     │  │   MACHINE LEARNING ENGINE│
               │   Data/users.db (PBKDF2) │  │   Random Forest Classifier│
               │   Salted Passwords & JWT │  │   DataCo (180,516 records)│
               └──────────────────────────┘  └──────────────────────────┘
```

---

## 🌟 Key Features

1. **Enterprise User Authentication:**
   - **User Sign-Up:** Secure registration with email validation and PBKDF2-HMAC-SHA256 salted password hashing.
   - **User Login:** Issues tamper-evident signed JWT bearer tokens with 7-day expiration.
   - **1-Click Recruiter Demo:** Instant guest review access without manual form entry.
   - **Persistent SQLite Database:** Auto-initialized user storage with zero external database setup required.
2. **Executive KPI Dashboard:**
   - On-time delivery rate, total revenue exposure ($36.7M), unique order counts, and schedule gap days.
3. **Delivery Performance Analytics:**
   - On-time delivery comparisons across shipping modes (First Class, Second Class, Standard, Same Day).
   - Regional risk segmentation (Sales Exposure vs. Reliability) with K-Means risk clustering.
4. **Real-Time ML Delay Predictor:**
   - Real-time scoring using a 78-feature Random Forest model trained on the DataCo Supply Chain dataset.
   - Returns continuous late probability percentage and actionable operational mitigation strategies.
5. **Render Cloud Deployment Ready:**
   - Multi-service `render.yaml` blueprint with automated dependency installation, health checks, and cross-service communication.

---

## 🚀 Local Quickstart

### 1. Prerequisites
- Python 3.11.x installed.

### 2. Setup Virtual Environment
```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start Backend (Terminal 1)
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
* Interactive API Documentation (Swagger): `http://127.0.0.1:8000/docs`
* Health Endpoint: `http://127.0.0.1:8000/health`

### 4. Start Frontend (Terminal 2)
```powershell
streamlit run app.py
```
* Dashboard will automatically open at `http://localhost:8501`.

---

## ☁️ Deploying to Render

This repository includes a pre-configured `render.yaml` Blueprint.

### Option A: 1-Click Render Blueprint (Recommended)
1. Push this repository to your **GitHub** account.
2. Log into [Render](https://render.com).
3. Click **New +** $\rightarrow$ **Blueprint**.
4. Connect your GitHub repository.
5. Render will automatically detect `render.yaml` and provision both:
   - `supply-chain-api` (FastAPI backend)
   - `supply-chain-dashboard` (Streamlit UI)
6. Once deployed, open the public URL of `supply-chain-dashboard` in your browser!

### Option B: Deploying Manually on Render (Free Tier)

#### 1. Deploy the API:
- **Service Type:** Web Service
- **Runtime:** Python 3
- **Build Command:** `pip install -r requirements-common.txt -r requirements-api.txt`
- **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`
- **Environment Variables:**
  - `PYTHON_VERSION`: `3.11.10`
  - `API_SHARED_TOKEN`: *(click generate or enter any secret key)*
  - `JWT_SECRET_KEY`: *(click generate or enter any secret key)*

#### 2. Deploy the Dashboard:
- **Service Type:** Web Service
- **Runtime:** Python 3
- **Build Command:** `pip install -r requirements-common.txt -r requirements-dashboard.txt`
- **Start Command:** `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
- **Health Check Path:** `/_stcore/health`
- **Environment Variables:**
  - `PYTHON_VERSION`: `3.11.10`
  - `API_BASE_URL`: `https://<YOUR-API-SERVICE-NAME>.onrender.com`
  - `API_SHARED_TOKEN`: *(same token value as API)*

> **Note on Render Free Tier Cold Starts:**  
> Free instances spin down after 15 minutes of inactivity. When accessed after idle, the service takes 30–50 seconds to wake up. The UI features friendly loading state detection and automated retry triggers.

---

## 📊 Dataset & Model Details

- **Dataset:** DataCo Smart Supply Chain Dataset (180,516 orders, 65,749 distinct order transactions).
- **Target Variable:** `Late_delivery_risk` (Binary: 0 = On-Time, 1 = Late).
- **Model Architecture:** Random Forest Classifier serialized with `scikit-learn==1.6.1`.
- **Validation Score:** 0.73 ROC-AUC on historical test holdout.
