from fastapi.responses import FileResponse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Union 
from fastapi.middleware.cors import CORSMiddleware
# The incorrect import that caused the ImportError is REMOVED:
# from sklearn.exceptions import UserWarning 
# We rely on the core logic fix (passing a DataFrame) to handle the warning.


# --- Configuration and File Paths ---
MODEL_PATH = "demand_prediction_rf_model.pkl"
DATA_PATH = "vendor_daily_sales.csv"

# --- Corrected Feature Mapping (55 features) ---
MODEL_FEATURE_COLUMNS = [
    'price_per_unit', 'stock', 'is_weekend', 'is_holiday', 'temperature_c',
    'precipitation_mm', 'market_footfall', 'competitor_price_per_unit', 'month',
    'week', 'day_of_month', 'price_diff', 'price_ratio', 'day_of_year', 'day_sin',
    'day_cos', 'month_sin', 'month_cos', 'is_cheaper', 'is_rainy', 'is_heavy_rain',
    'is_hot', 'is_mild', 'stock_ratio', 'footfall_ratio',
    'units_sold_lag_1', 'units_sold_lag_2', 'units_sold_lag_3', 'units_sold_lag_7',
    'units_sold_rolling_mean_3', 'units_sold_rolling_std_3',
    'units_sold_rolling_mean_7', 'units_sold_rolling_std_7',
    'item_apple', 'item_banana', 'item_cucumber', 'item_mango', 'item_onion',
    'item_potato', 'item_spinach', 'item_tomato', 'category_fruit',
    'category_vegetable', 'promotion_type_bundle', 'promotion_type_discount',
    'promotion_type_loyalty', 'promotion_type_none', 'supply_disruption_none',
    'supply_disruption_quality_issue', 'supply_disruption_supplier_shortage',
    'supply_disruption_traffic_delay', 'event_festival', 'event_local_fair',
    'event_match_day', 'event_none'
]

# --- Global Variables ---
model = None
historical_df = None
# Stores averages by Day of Year (DOY) for date-specific features
doy_averages_df = None
# Stores averages by Item for item-specific features
item_averages_df = None
# Global fallback averages (calculated during startup)
global_avg_dict: Dict[str, Any] = {}


# --- Helper Functions (Omitted for brevity, assumed to be correct) ---

def load_historical_data():
    """Loads and preprocesses the historical sales data for lag calculations."""
    try:
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        df['is_weekend'] = df['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        # Sort by date and item for correct lag/rolling calculations
        df = df.sort_values(by=['date', 'item']).reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"ERROR: Historical data file not found at {DATA_PATH}. Lag/Rolling features will fail.")
        return None

def calculate_all_averages(df: pd.DataFrame):
    """Calculates all necessary historical averages (DOY, Item, and Global)."""
    
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # 1. Day of Year (DOY) Averages (for weather, footfall, holiday)
    doy_averages = df.groupby('day_of_year').agg({
        'temperature_c': 'mean',
        'precipitation_mm': 'mean',
        'is_holiday': 'max',
        'market_footfall': 'mean'
    }).reset_index()
    doy_averages['is_holiday'] = (doy_averages['is_holiday'] > 0).astype(int)
    
    # 2. Item Averages (for price and stock)
    item_averages = df.groupby('item').agg({
        'price_per_unit': 'mean',
        'competitor_price_per_unit': 'mean',
        'stock': 'mean'
    }).reset_index()

    # 3. Global Averages (Fallback for everything)
    global_avg = df.agg({
        'temperature_c': 'mean',
        'precipitation_mm': 'mean',
        'is_holiday': 'mean',
        'market_footfall': 'mean',
        'price_per_unit': 'mean',
        'competitor_price_per_unit': 'mean',
        'stock': 'mean'
    }).to_dict()
    global_avg['is_holiday'] = int(round(global_avg['is_holiday']))
    
    return doy_averages, item_averages, global_avg

# --- Pydantic Models for API ---

class PredictionInput(BaseModel):
    """
    Schema for the prediction request body. 
    Numerical fields are now OPTIONAL (Union[float, None]) to allow user override, 
    falling back to historical averages if None is provided.
    """
    date: str  # YYYY-MM-DD format
    item: str
    promotion_type: str
    supply_disruption: str
    event: str
    # Fields that can be overridden by the user (or left blank/None)
    price_per_unit: Union[float, None] = None
    competitor_price_per_unit: Union[float, None] = None
    stock: Union[float, None] = None
    market_footfall: Union[float, None] = None

class PredictionOutput(BaseModel):
    """Schema for the prediction response."""
    item: str
    prediction_date: str
    predicted_units_sold: float
    model_used: str = "Random Forest Regressor"

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Gromally Demand Prediction API (Robust Fallback)",
    description="REST API endpoint where price, stock, and footfall can be overridden by user input, or auto-filled using historical averages."
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------

# --- Feature Engineering Functions (The core logic) ---

def engineer_features(raw_data: PredictionInput, historical_data: pd.DataFrame, doy_avg_df: pd.DataFrame, item_avg_df: pd.DataFrame, global_avg: Dict):
    """
    Transforms the raw input data into the full feature vector (55 features),
    using user inputs where available, and falling back to averages otherwise.
    """

    input_df = pd.DataFrame([raw_data.model_dump(exclude_none=True)]) # exclude_none ensures only provided fields are kept initially
    input_df['date'] = pd.to_datetime(input_df['date'])

    doy = input_df['date'].dt.dayofyear.iloc[0]

    # --- 1. Determine Item-Specific Averages (Fallback for Price/Stock) ---
    try:
        lookup_item = item_avg_df[item_avg_df['item'] == raw_data.item].iloc[0]
        avg_price = lookup_item['price_per_unit']
        avg_comp_price = lookup_item['competitor_price_per_unit']
        avg_stock = lookup_item['stock']
    except IndexError:
        avg_price = global_avg['price_per_unit']
        avg_comp_price = global_avg['competitor_price_per_unit']
        avg_stock = global_avg['stock']

    # Apply Price/Stock: Use user input (from input_df if present) OR the item average
    if 'price_per_unit' not in input_df.columns:
        input_df['price_per_unit'] = avg_price
    if 'competitor_price_per_unit' not in input_df.columns:
        input_df['competitor_price_per_unit'] = avg_comp_price
    if 'stock' not in input_df.columns:
        input_df['stock'] = avg_stock


    # --- 2. Determine Date-Specific Averages (Fallback for Footfall, Weather, Holiday) ---
    try:
        lookup_doy = doy_avg_df[doy_avg_df['day_of_year'] == doy].iloc[0]
        avg_temp = lookup_doy['temperature_c']
        avg_precip = lookup_doy['precipitation_mm']
        avg_holiday = lookup_doy['is_holiday'] 
        avg_footfall = lookup_doy['market_footfall'] 
    except IndexError:
        print(f"Warning: Day of Year {doy} not found in historical data. Using global averages for weather/holiday/footfall.")
        avg_temp = global_avg['temperature_c']
        avg_precip = global_avg['precipitation_mm']
        avg_holiday = global_avg['is_holiday']
        avg_footfall = global_avg['market_footfall'] 

    # Apply Footfall: Use user input OR the DOY average
    if 'market_footfall' not in input_df.columns:
        input_df['market_footfall'] = avg_footfall
        
    # Apply Weather/Holiday: These are ALWAYS auto-filled based on date, ignoring any hypothetical user input.
    input_df['temperature_c'] = avg_temp
    input_df['precipitation_mm'] = avg_precip
    input_df['is_holiday'] = avg_holiday
        
    # 2.1 Calculate 'is_weekend'
    input_df['is_weekend'] = input_df['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)


    # 3. Combine with historical data for time-series features
    item_history = historical_data[historical_data['item'] == raw_data.item].copy()

    # Category is derived from item
    categories = {'apple': 'fruit', 'banana': 'fruit', 'mango': 'fruit', 
                  'tomato': 'vegetable', 'onion': 'vegetable', 'potato': 'vegetable', 
                  'spinach': 'vegetable', 'cucumber': 'vegetable'}
    input_df['category'] = categories.get(raw_data.item, 'unknown')
    input_df['units_sold'] = np.nan 

    # Sort and combine the latest historical data with the new prediction row
    combined_df = pd.concat([item_history, input_df], ignore_index=True)
    combined_df = combined_df.sort_values(by='date').drop_duplicates(subset=['date', 'item'], keep='last').reset_index(drop=True)

    # 4. Time-based Features (same as before)
    combined_df['month'] = combined_df['date'].dt.month
    combined_df['week'] = combined_df['date'].dt.isocalendar().week.astype(int)
    combined_df['day_of_month'] = combined_df['date'].dt.day
    combined_df['day_of_year'] = combined_df['date'].dt.dayofyear

    # Sine/Cosine Encodings 
    combined_df['day_sin'] = np.sin(2 * np.pi * combined_df['day_of_year'] / 366)
    combined_df['day_cos'] = np.cos(2 * np.pi * combined_df['day_of_year'] / 366)
    combined_df['month_sin'] = np.sin(2 * np.pi * combined_df['month'] / 12)
    combined_df['month_cos'] = np.cos(2 * np.pi * combined_df['month'] / 12)
    
    # 5. Calculated and Boolean Features (using potentially user-provided values)
    combined_df['price_diff'] = combined_df['price_per_unit'] - combined_df['competitor_price_per_unit']
    combined_df['price_ratio'] = combined_df['price_per_unit'] / combined_df['competitor_price_per_unit']
    combined_df['is_cheaper'] = (combined_df['price_per_unit'] < combined_df['competitor_price_per_unit']).astype(int)

    # Weather Flags (using DOY average values)
    combined_df['is_rainy'] = (combined_df['precipitation_mm'] > 0).astype(int)
    combined_df['is_heavy_rain'] = (combined_df['precipitation_mm'] >= 10).astype(int)
    combined_df['is_hot'] = (combined_df['temperature_c'] >= 28).astype(int)
    combined_df['is_mild'] = ((combined_df['temperature_c'] >= 20) & (combined_df['temperature_c'] < 28)).astype(int)

    # Ratio Features
    if not item_history.empty:
        max_stock = item_history['stock'].max()
    else:
        max_stock = global_avg['stock'] 

    combined_df['stock_ratio'] = combined_df['stock'] / max_stock 
    combined_df['footfall_ratio'] = combined_df['market_footfall'] / 1000 

    # 6. Lag and Rolling Mean/Std Features (same as before)
    for lag in [1, 2, 3, 7]:
        combined_df[f'units_sold_lag_{lag}'] = combined_df['units_sold'].shift(lag)

    for window in [3, 7]:
        combined_df[f'units_sold_rolling_mean_{window}'] = combined_df['units_sold'].shift(1).rolling(window=window).mean()
        combined_df[f'units_sold_rolling_std_{window}'] = combined_df['units_sold'].shift(1).rolling(window=window).std().fillna(0)

    combined_df.fillna(0, inplace=True)

    # 7. One-Hot Encoding (same as before)
    data_to_encode = combined_df.tail(1).copy()
    
    # Item OHE 
    for item in ['apple', 'banana', 'cucumber', 'mango', 'onion', 'potato', 'spinach', 'tomato']:
        data_to_encode[f'item_{item}'] = (data_to_encode['item'] == item).astype(int)

    # Category OHE 
    for cat in ['fruit', 'vegetable']:
        data_to_encode[f'category_{cat}'] = (data_to_encode['category'] == cat).astype(int)

    # Promotion OHE 
    for promo in ['bundle', 'discount', 'loyalty', 'none']:
        data_to_encode[f'promotion_type_{promo}'] = (data_to_encode['promotion_type'] == promo).astype(int)

    # Supply Disruption OHE
    for sd in ['none', 'quality_issue', 'supplier_shortage', 'traffic_delay']:
        data_to_encode[f'supply_disruption_{sd}'] = (data_to_encode['supply_disruption'] == sd).astype(int)

    # Event OHE
    for event in ['none', 'festival', 'match_day', 'local_fair']:
        data_to_encode[f'event_{event}'] = (data_to_encode['event'] == event).astype(int)

    # 8. Final preparation
    final_feature_data = data_to_encode[MODEL_FEATURE_COLUMNS].tail(1)
    # 9. Extract the Stock value used for this prediction (either user-supplied or fallback)
    # We use .iloc[0] because final_feature_data is a single-row DataFrame
    stock_for_prediction = final_feature_data['stock'].iloc[0]

    return final_feature_data, stock_for_prediction


# --- API Lifecycle and Endpoints (Omitted for brevity) ---

@app.on_event("startup")
async def load_assets():
    """Load the model and historical data, and calculate all averages on application startup."""
    global model, historical_df, doy_averages_df, item_averages_df, global_avg_dict
    try:
        # Load the Random Forest model
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")

        # Load historical data 
        historical_df = load_historical_data()
        if historical_df is not None:
             print(f"Successfully loaded {len(historical_df)} historical records for feature engineering.")
             
             # Calculate all averages (DOY, Item, and Global)
             doy_averages_df, item_averages_df, global_avg_dict = calculate_all_averages(historical_df)
             
             print("Successfully pre-calculated historical DOY, Item, and Global fallbacks.")

    except FileNotFoundError as e:
        print(f"Error during startup: {e}")
        raise HTTPException(status_code=500, detail=f"Required file not found: {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        raise HTTPException(status_code=500, detail="Model or data loading failed.")


# @app.get("/")
# def read_root():
#     """Simple health check endpoint."""
#     return {"status": "ok", "model_loaded": model is not None, "data_available": historical_df is not None, "doy_averages_calculated": doy_averages_df is not None}


@app.get("/", include_in_schema=False)
def serve_index():
    """Serves the main HTML page (the frontend dashboard)."""
    return FileResponse("index.html")

@app.post("/predict", response_model=PredictionOutput)
def predict_demand(data: PredictionInput):
    """
    Endpoint to predict 'units_sold'. Includes a post-prediction cap based on stock.
    """
    if model is None or historical_df is None or doy_averages_df is None or item_averages_df is None:
        raise HTTPException(status_code=500, detail="Model or data assets not loaded.")

    try:
        # 1. Engineer features and retrieve the stock value
        feature_data, stock_level = engineer_features(data, historical_df, doy_averages_df, item_averages_df, global_avg_dict)

        # 2. Make Prediction (passing DataFrame with feature names)
        unconstrained_prediction = model.predict(feature_data)[0]

        # 3. CRITICAL BUSINESS LOGIC: Post-Prediction Cap
        # Sales cannot exceed the available stock.
        final_prediction = min(float(unconstrained_prediction), float(stock_level))
        
        # 4. Ensure non-negative and round
        final_prediction = max(0.0, float(np.round(final_prediction, 1)))

        # 5. Format and Return Result
        return PredictionOutput(
            item=data.item,
            prediction_date=data.date,
            predicted_units_sold=final_prediction 
        )

    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction processing error: {e}")