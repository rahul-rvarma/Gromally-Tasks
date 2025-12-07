# 📊 Demand Prediction API for Gromally Grocery Supplies

This document provides a comprehensive overview of the demand prediction module built for a street-vendor-like setup. The solution uses machine learning, robust data handling, and critical business logic exposed via a single, easily deployable FastAPI service.

## Project Deliverables

| File Name | Description |
| :--- | :--- |
| `main.py` | FastAPI backend logic, including data preprocessing, feature engineering, prediction logic, and serving the static frontend. |
| `demand_prediction_rf_model.pkl` | The trained **Random Forest Regressor** model. |
| `vendor_daily_sales.csv` | The historical sales dataset used for training and providing fallback data. |
| `index.html` | The interactive web dashboard (frontend) built with HTML, Bootstrap, and pure JavaScript. |
| `requirements.txt` | Python dependencies for the FastAPI environment. |

---

## 💻 Tech Stack & Architecture

The application adopts a minimalist, unified architecture for simplified deployment:

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend / API** | Python, **FastAPI** | Core service. It handles the prediction logic (`/predict`) and serves the entire frontend (`/`). |
| **Model** | **Scikit-learn (Random Forest Regressor)** | The primary demand forecasting engine, chosen for its robustness with diverse features. |
| **Frontend** | HTML, CSS (Bootstrap), Pure JavaScript | Interactive dashboard for user input and result interpretation. |
| **Deployment** | **Render** | Single Web Service deployment hosting both the API and the static files. |

---

## 🧠 Model & Core Logic Explanation

### 1. Model Approach & Features

The task is treated as a **Supervised Machine Learning Regression** problem. The **Random Forest Regressor** uses a comprehensive set of features to predict demand:

| Feature Type | Examples | Rationale |
| :--- | :--- | :--- |
| **Time Series** | `units_sold_lag_7`, `units_sold_rolling_mean_30` | Captures weekly seasonality and longer-term sales trends. |
| **Contextual** | `temperature_c`, `precipitation_mm`, `market_footfall` | Accounts for external factors impacting customer traffic. |
| **Categorical** | `item`, `promotion_type`, `day_of_week` | Accounts for product-specific demand and marketing effects. |

### 2. Key Robustness Mechanisms (Critical Business Logic)

The API is engineered to handle two critical real-world challenges: missing data and supply constraints.

#### A. Data Fallback (Handling Missing Inputs)

To ensure the API never fails due to missing data (especially when predicting far into the future), the system provides **robust fallback values** for any `null` or missing numerical input:

* **Financial/Stock Features (e.g., price, stock):** Filled with the **historical item average**.
* **Time Series Features (e.g., Lag/Rolling values):** Filled with **zero (0.0)** for future dates beyond the training data period.
* **Contextual Features (e.g., footfall):** Filled with the **Day-of-Year average** from historical data.

#### B. Post-Prediction Cap (Handling Supply)

This logic is crucial for operational decision-making. The core ML model predicts **Unconstrained Demand** (what customers *want*). The API then applies the actual supply constraint to predict **Sales** (what the vendor *can sell*).

The final prediction is capped by the available stock:

$$\text{Final Units Sold} = \min(\text{Model's Unconstrained Demand}, \text{Available Stock})$$


[Image of supply and demand curve intersection showing price and quantity equilibrium]


This enables the vendor to operate the tool in two distinct, meaningful modes:

| Mode | Stock Input | Output | Purpose |
| :--- | :--- | :--- | :--- |
| **1. Stocking Decision** | Leave **BLANK** (or set high) | **Unconstrained Demand** | Tells the vendor **how much to purchase/stock** to maximize sales and meet full customer interest. |
| **2. Sales Forecast** | Enter **ACTUAL** stock (e.g., 30) | **Constrained Sales** | Tells the vendor **how much they will sell** given their limited supply, highlighting potential lost sales. |

---

## 🚀 Deployment and Usage

The entire application runs as a single FastAPI Web Service, deployed on Render.

### Live Application Link
You can access the live, interactive dashboard here:
**[https://demand-prediction-57ke.onrender.com](https://demand-prediction-57ke.onrender.com)**

### 1. Local Setup

1.  **Install dependencies:** `pip install -r requirements.txt`
2.  **Run the application:** `uvicorn main:app --reload`
3.  **Access the dashboard:** `http://127.0.0.1:8000/`

### 2. Render Deployment

1.  Configure a **single Render Web Service** pointing to your repository.
2.  Set the **Root Directory** to blank (repository root).
3.  **Build Command:** `pip install -r requirements.txt`
4.  **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 3. Usage Notes

* In the deployed `index.html`, the `API_URL` is set to the **relative path** (`/predict`) to ensure seamless communication within the single Render service environment.
* The dashboard provides clear, dynamic interpretations of the result based on whether the stock field was used to constrain the sales.

---

## 💡 Assumptions

1.  **Item-Specific Averages:** Historical averages for price and stock are reliable proxies for missing future data.
2.  **Lagged Features:** The demand for an item is influenced by its sales in the immediate past (1, 7, 30 days prior).
3.  **Future Data Handling:** For predictions far beyond the historical data, all sales lag features are safely assumed to be zero, allowing the prediction to rely on item and seasonal averages.
4.  **Data Quality:** The provided `vendor_daily_sales.csv` is assumed to be representative of the vendor's actual market environment.