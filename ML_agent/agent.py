# c:\Users\Rahul\Desktop\Hackathon\ML_agent\agent.py
"""
ADK Agent using LlmAgent to run Prophet and Regression forecasts sequentially.
Results are stored in BigQuery tables/views and plots saved locally.
"""

import traceback
from typing import Dict, List, Any, Optional
import datetime
import uuid
import time
import os # For creating plot directory

import numpy as np
import pandas as pd
from google.adk.agents.llm_agent import LlmAgent
# REMOVED: from google.adk.agents.parallel_agent import ParallelAgent
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

project_id = "us-con-gcp-sbx-0000489-032025"

# --- ML Model Imports ---
# (Keep ML imports as they were)
try:
    from prophet import Prophet
except ImportError:
    print("Prophet library not found. Please install it: pip install prophet")
    class Prophet: # Dummy class
        def __init__(self, *args, **kwargs): pass
        def fit(self, df): pass
        def make_future_dataframe(self, periods): return pd.DataFrame({'ds': pd.to_datetime(['2024-01-01'])})
        def predict(self, future): return pd.DataFrame({'ds': future['ds'], 'yhat': [0]*len(future), 'yhat_lower': [0]*len(future), 'yhat_upper': [0]*len(future)})
        def add_country_holidays(self, country_name): print(f"Dummy: Adding holidays for {country_name}")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb
except ImportError:
    print("scikit-learn or xgboost not found. Please install them: pip install scikit-learn xgboost")
    class RandomForestRegressor: # Dummy class
         def __init__(self, *args, **kwargs): pass
         def fit(self, X, y): pass
         def predict(self, X): return np.zeros(len(X))
    class xgb: # Dummy class
        class XGBRegressor:
            def __init__(self, *args, **kwargs): pass
            def fit(self, X, y): pass
            def predict(self, X): return np.zeros(len(X))

# --- Visualization Import ---
# (Keep visualization import as it was)
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
except ImportError:
    print("matplotlib not found. Please install it: pip install matplotlib")
    plt = None

# --- Constants ---
# (Keep constants as they were)
GEMINI_MODEL: str = "gemini-2.0-flash"
DEFAULT_BQ_LOCATION: str = "US"
ANALYTICS_DATASET_ID: str = "analytics"
BI_DATASET_ID: str = "BI"
PLOT_SAVE_DIR: str = "forecast_plots"

# --- Helper Functions ---
# (Keep helper functions: ensure_dataset_exists, create_time_features, _save_forecast_plot as they were)
def ensure_dataset_exists(client: bigquery.Client, project_id: str, dataset_id: str, location: str):
    """Checks if a dataset exists, creates it if not."""
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        print(f"[Dataset Check] Dataset '{project_id}.{dataset_id}' already exists.")
    except NotFound:
        print(f"[Dataset Check] Dataset '{project_id}.{dataset_id}' not found. Creating...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset, exists_ok=True)
        print(f"[Dataset Check] Dataset '{project_id}.{dataset_id}' created.")
    except Exception as e:
        print(f"[Dataset Check] Error checking/creating dataset '{project_id}.{dataset_id}': {e}")
        raise

def create_time_features(df, date_col='ds'):
    """Creates time-based features from a datetime column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['quarter'] = df[date_col].dt.quarter
    return df

def _save_forecast_plot(
    historical_df: pd.DataFrame, # Contains 'ds' and 'y'
    forecast_df: pd.DataFrame,   # Contains 'forecast_date', 'forecasted_sales', optionally bounds
    model_name: str,
    view_name: str, # Use view name for uniqueness in filename
    plot_dir: str = PLOT_SAVE_DIR
) -> Optional[str]:
    """Generates and saves a forecast plot locally."""
    if plt is None:
        print("[Plotting] Matplotlib not installed. Skipping plot generation.")
        return None
    try:
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_view_name = "".join(c if c.isalnum() else "_" for c in view_name)
        plot_filename = os.path.join(plot_dir, f"{safe_view_name}_{model_name}_{timestamp}.png")

        print(f"[Plotting] Generating plot: {plot_filename}")
        plt.figure(figsize=(12, 6))
        plt.plot(historical_df['ds'], historical_df['y'], 'k.', label='Historical Actuals')
        plt.plot(forecast_df['forecast_date'], forecast_df['forecasted_sales'], ls='-', c='blue', label=f'{model_name} Forecast')
        if 'forecast_lower_bound' in forecast_df.columns and 'forecast_upper_bound' in forecast_df.columns:
            plt.fill_between(forecast_df['forecast_date'], forecast_df['forecast_lower_bound'], forecast_df['forecast_upper_bound'], color='blue', alpha=0.2, label='Confidence Interval')
        plt.title(f'Sales Forecast ({model_name}) - {view_name}')
        plt.xlabel('Date'); plt.ylabel('Sales'); plt.legend(); plt.grid(True); plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        print(f"[Plotting] Plot saved successfully.")
        return plot_filename
    except Exception as e:
        print(f"[Plotting] Error generating or saving plot: {e}")
        traceback.print_exc()
        return None


# --- Tool Functions ---
# (Keep tool functions: forecast_sales_to_bq_table_and_view, regression_forecast_to_bq_table_and_view as they were)
def forecast_sales_to_bq_table_and_view(
    project_id: str,
    source_dataset_name: str,
    source_table_name: str,
    forecast_table_name: str,
    forecast_view_name: str,
    date_column: str = 'order_date',
    sales_column: str = 'total_amount',
    forecast_periods: int = 30,
    country_code_for_holidays: Optional[str] = None
):
    """
    (Prophet Tool) Fetches sales data, forecasts using Prophet, stores in table/view, saves plot, returns status.
    """
    full_source_table_name = f"`{project_id}.{source_dataset_name}.{source_table_name}`"
    full_forecast_table_name = f"`{project_id}.{ANALYTICS_DATASET_ID}.{forecast_table_name}`"
    full_forecast_view_name = f"`{project_id}.{BI_DATASET_ID}.{forecast_view_name}`"
    holiday_info = f"including holidays for country code '{country_code_for_holidays}'" if country_code_for_holidays else "without specific country holidays"
    plot_filename = None

    print(f"[Prophet Tool] Starting forecast process {holiday_info}...")
    print(f"  Source: {full_source_table_name}")
    print(f"  Output Table: {full_forecast_table_name}")
    print(f"  Output View: {full_forecast_view_name}")

    try:
        client = bigquery.Client(project=project_id)
        ensure_dataset_exists(client, project_id, ANALYTICS_DATASET_ID, DEFAULT_BQ_LOCATION)
        ensure_dataset_exists(client, project_id, BI_DATASET_ID, DEFAULT_BQ_LOCATION)

        query = f"""
            SELECT DATE({date_column}) AS ds, SUM({sales_column}) AS y
            FROM {full_source_table_name}
            WHERE {date_column} IS NOT NULL AND {sales_column} IS NOT NULL
            GROUP BY ds ORDER BY ds
        """
        print(f"[Prophet Tool] Fetching and filtering historical data...")
        df = client.query(query).to_dataframe()

        if df.empty:
            return {"status": "error", "message": f"No valid historical data found. Cannot create forecast."}
        print(f"[Prophet Tool] Fetched {len(df)} rows of valid historical data.")

        print("[Prophet Tool] Initializing and fitting Prophet model...")
        model = Prophet()
        if country_code_for_holidays:
            try: model.add_country_holidays(country_name=country_code_for_holidays)
            except Exception as holiday_err: print(f"[Prophet Tool] Warning: Failed to add holidays: {holiday_err}")
        model.fit(df)
        print("[Prophet Tool] Prophet model fitted.")

        future = model.make_future_dataframe(periods=forecast_periods)
        print("[Prophet Tool] Generating forecast...")
        forecast = model.predict(future)
        forecast_df_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df_out.rename(columns={'ds': 'forecast_date', 'yhat': 'forecasted_sales', 'yhat_lower': 'forecast_lower_bound', 'yhat_upper': 'forecast_upper_bound'}, inplace=True)
        print("[Prophet Tool] Forecast generated.")

        plot_filename = _save_forecast_plot(historical_df=df, forecast_df=forecast_df_out, model_name="Prophet", view_name=forecast_view_name)

        forecast_table_ref = client.dataset(ANALYTICS_DATASET_ID).table(forecast_table_name)
        print(f"[Prophet Tool] Loading forecast data to permanent table: {full_forecast_table_name}")
        job_config_load = bigquery.LoadJobConfig(autodetect=True, write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        load_job = client.load_table_from_dataframe(forecast_df_out, forecast_table_ref, job_config=job_config_load)
        load_job.result()
        print(f"[Prophet Tool] Forecast data loaded to permanent table.")

        print(f"[Prophet Tool] Creating/Replacing view {full_forecast_view_name}...")
        view_query = f"""
            CREATE OR REPLACE VIEW {full_forecast_view_name}
            OPTIONS(description="Sales forecast generated by Prophet model {holiday_info}", labels=[("data_source","prophet_forecast")])
            AS SELECT forecast_date, forecasted_sales, forecast_lower_bound, forecast_upper_bound
            FROM {full_forecast_table_name} ORDER BY forecast_date
        """
        query_job = client.query(view_query); query_job.result()
        print(f"[Prophet Tool] View {full_forecast_view_name} created/replaced successfully.")

        return {
            "status": "success", "message": f"Prophet forecast successfully generated ({holiday_info}) and stored. Plot saved.",
            "forecast_table": full_forecast_table_name.strip('`'), "forecast_view": full_forecast_view_name.strip('`'),
            "plot_filename": plot_filename
        }
    except Exception as e:
        print(f"[Prophet Tool] ERROR during forecasting process: {traceback.format_exc()}")
        return {"status": "error", "message": f"Failed during Prophet forecasting process: {str(e)}"}

def regression_forecast_to_bq_table_and_view(
    project_id: str,
    source_dataset_name: str,
    source_table_name: str,
    forecast_table_name: str,
    forecast_view_name: str,
    date_column: str = 'order_date',
    sales_column: str = 'total_amount',
    forecast_periods: int = 30,
    model_type: str = 'RandomForest',
    feature_columns: Optional[List[str]] = None
):
    """
    (Regression Tool) Fetches data, aggregates daily sales, engineers features, trains model,
    predicts future sales, stores in table/view, saves plot, returns status.
    """
    full_source_table_name = f"`{project_id}.{source_dataset_name}.{source_table_name}`"
    full_forecast_table_name = f"`{project_id}.{ANALYTICS_DATASET_ID}.{forecast_table_name}`"
    full_forecast_view_name = f"`{project_id}.{BI_DATASET_ID}.{forecast_view_name}`"
    model_name = model_type.lower()
    plot_filename = None

    print(f"[Regression Tool] Starting forecast process using {model_type}...")
    print(f"  Source: {full_source_table_name}")
    print(f"  Features: Time features + {feature_columns or 'None'}")
    print(f"  Target: SUM({sales_column}) aggregated by day")
    print(f"  Output Table: {full_forecast_table_name}")
    print(f"  Output View: {full_forecast_view_name}")

    if model_name not in ['randomforest', 'xgboost']:
        return {"status": "error", "message": f"Invalid model_type '{model_type}'. Choose 'RandomForest' or 'XGBoost'."}

    try:
        client = bigquery.Client(project=project_id)
        ensure_dataset_exists(client, project_id, ANALYTICS_DATASET_ID, DEFAULT_BQ_LOCATION)
        ensure_dataset_exists(client, project_id, BI_DATASET_ID, DEFAULT_BQ_LOCATION)

        select_cols = [f"DATE({date_column}) AS ds", f"SUM({sales_column}) AS y"]
        group_by_cols = ["ds"]
        safe_feature_columns = []
        original_feature_names = []
        if feature_columns:
            safe_feature_columns = [f"`{col.strip('`')}`" for col in feature_columns]
            original_feature_names = [col.strip('`') for col in feature_columns]
            feature_selects = [f"MAX({col}) AS {name}_agg" for col, name in zip(safe_feature_columns, original_feature_names)]
            select_cols.extend(feature_selects)
        query = f""" SELECT {', '.join(select_cols)} FROM {full_source_table_name}
            WHERE {date_column} IS NOT NULL AND {sales_column} IS NOT NULL
            GROUP BY {', '.join(group_by_cols)} ORDER BY ds """

        print(f"[Regression Tool] Fetching, filtering, and aggregating historical data...")
        df = client.query(query).to_dataframe()

        if df.empty: return {"status": "error", "message": f"No valid historical data found. Cannot create forecast."}
        print(f"[Regression Tool] Fetched {len(df)} aggregated daily rows.")

        print("[Regression Tool] Engineering time features...")
        if feature_columns: df.rename(columns={f"{name}_agg": name for name in original_feature_names}, inplace=True)
        df_features = create_time_features(df, date_col='ds')
        categorical_cols = [col for col in original_feature_names if col in df_features.columns and (df_features[col].dtype == 'object' or df_features[col].dtype.name == 'category')] if feature_columns else []
        if categorical_cols: df_features = pd.get_dummies(df_features, columns=categorical_cols, dummy_na=False)

        target_col = 'y'
        features_for_training = [col for col in df_features.columns if col not in ['ds', target_col] and col not in original_feature_names]
        if feature_columns:
            numerical_features = [col for col in original_feature_names if col not in categorical_cols and col in df_features.columns]
            features_for_training.extend(numerical_features)
            dummy_cols = [col for col in df_features.columns if any(cat_col + '_' in col for cat_col in categorical_cols)]
            features_for_training.extend(dummy_cols)
        features_for_training = sorted(list(set(features_for_training)))
        print(f"[Regression Tool] Features used for training: {features_for_training}")

        df_features.dropna(subset=features_for_training + [target_col], inplace=True)
        if df_features.empty: return {"status": "error", "message": "No data remaining after feature engineering and NaN handling."}
        X_train = df_features[features_for_training]; y_train = df_features[target_col]

        print(f"[Regression Tool] Training {model_type} model...")
        if model_name == 'randomforest': model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_name == 'xgboost': model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        print("[Regression Tool] Model trained.")

        last_date = df['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        future_features_df = create_time_features(future_df, date_col='ds')
        if categorical_cols:
             dummy_cols_train = [col for col in X_train.columns if any(cat_col + '_' in col for cat_col in categorical_cols)]
             for col in dummy_cols_train:
                 if col not in future_features_df.columns: future_features_df[col] = 0
        if feature_columns:
            numerical_features = [col for col in original_feature_names if col not in categorical_cols]
            features_for_prediction = [f for f in features_for_training if f not in numerical_features]
            if numerical_features: print(f"[Regression Tool] Warning: Excluding numerical features {numerical_features} from future predictions.")
        else: features_for_prediction = features_for_training
        missing_cols = set(features_for_prediction) - set(future_features_df.columns)
        for c in missing_cols: future_features_df[c] = 0
        X_future = future_features_df[features_for_prediction]

        print("[Regression Tool] Generating forecast...")
        future_predictions = model.predict(X_future)
        final_forecast_df = pd.DataFrame({'forecast_date': future_dates, 'forecasted_sales': future_predictions, 'forecast_lower_bound': np.nan, 'forecast_upper_bound': np.nan})
        print("[Regression Tool] Forecast generated.")

        plot_filename = _save_forecast_plot(historical_df=df[['ds', 'y']], forecast_df=final_forecast_df, model_name=model_type, view_name=forecast_view_name)

        forecast_table_ref = client.dataset(ANALYTICS_DATASET_ID).table(forecast_table_name)
        print(f"[Regression Tool] Loading forecast data to permanent table: {full_forecast_table_name}")
        job_config_load = bigquery.LoadJobConfig(autodetect=True, write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        load_job = client.load_table_from_dataframe(final_forecast_df, forecast_table_ref, job_config=job_config_load)
        load_job.result()
        print(f"[Regression Tool] Forecast data loaded to permanent table.")

        print(f"[Regression Tool] Creating/Replacing view {full_forecast_view_name}...")
        view_query = f"""
            CREATE OR REPLACE VIEW {full_forecast_view_name}
            OPTIONS(description="Sales forecast generated by {model_type} model", labels=[("model_type","regression_{model_name}")])
            AS SELECT forecast_date, forecasted_sales, forecast_lower_bound, forecast_upper_bound
            FROM {full_forecast_table_name} ORDER BY forecast_date
        """
        query_job = client.query(view_query); query_job.result()
        print(f"[Regression Tool] View {full_forecast_view_name} created/replaced successfully.")

        return {
            "status": "success", "message": f"Regression forecast ({model_type}) successfully generated and stored. Plot saved.",
            "forecast_table": full_forecast_table_name.strip('`'), "forecast_view": full_forecast_view_name.strip('`'),
            "plot_filename": plot_filename
        }
    except Exception as e:
        print(f"[Regression Tool] ERROR during forecasting process: {traceback.format_exc()}")
        return {"status": "error", "message": f"Failed during regression forecasting process: {str(e)}"}


# --- Define the Sequential Forecasting Agent ---

root_agent = LlmAgent(
    name="SequentialForecastingAgent",
    model=GEMINI_MODEL,
    instruction=f"""
    You are an AI assistant that orchestrates sales forecasting using two methods: Prophet and Regression (RandomForest or XGBoost).
    Your task is to run these forecasts **sequentially**: first Prophet, then Regression.

    **Workflow:**

    1.  **Identify Parameters:** Extract the necessary parameters from the user request or context for *both* forecast types:
        - `project_id` (use '{project_id}' if not provided)
        - `source_dataset_name`
        - `source_table_name`
        - `prophet_forecast_table_name` (Use 'prophet_forecast_data' if not provided)
        - `prophet_forecast_view_name` (Use 'v_prophet_forecast' if not provided)
        - `regression_forecast_table_name` (Use 'regression_forecast_data' if not provided)
        - `regression_forecast_view_name` (Use 'v_regression_forecast' if not provided)
        - `date_column` (Default: 'order_date')
        - `sales_column` (Default: 'total_amount')
        - `forecast_periods` (Default: 30)
        - **Prophet Specific:** `country_code_for_holidays` (Optional, e.g., 'US')
        - **Regression Specific:** `model_type` ('RandomForest' or 'XGBoost', Default: 'RandomForest')
        - **Regression Specific:** `feature_columns` (Optional list of extra features from source table)

    2.  **Execute Prophet Forecast:**
        - Call the `forecast_sales_to_bq_table_and_view` tool with the identified parameters relevant to Prophet (source data, prophet output names, date/sales cols, periods, holidays).
        - Store the result dictionary returned by the tool. Let's call it `prophet_result`.

    3.  **Execute Regression Forecast:**
        - **After** the Prophet forecast completes, call the `regression_forecast_to_bq_table_and_view` tool with the parameters relevant to Regression (source data, regression output names, date/sales cols, periods, model_type, feature_columns). **Do NOT pass `country_code_for_holidays` to this tool.**
        - Store the result dictionary returned by the tool. Let's call it `regression_result`.

    4.  **Final Report:**
        - Once both tool calls are finished, compile a final summary report.
        - Include the status ('success' or 'error') and the message from both `prophet_result` and `regression_result`.
        - Mention the created table and view names for both forecasts if successful.
        - Mention the saved plot filenames for both forecasts if available.
        - Present this summary clearly to the user.
    """,
    description="Runs Prophet and Regression sales forecasts sequentially, storing results in BigQuery and saving plots.",
    tools=[
        forecast_sales_to_bq_table_and_view,      # Prophet tool
        regression_forecast_to_bq_table_and_view   # Regression tool
    ],
    output_key="sequential_forecast_summary" # Store the final compiled report here
)

