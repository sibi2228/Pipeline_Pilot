"""
ADK Agent for performing sales forecasting using Prophet and storing results in BigQuery.
Forecast data is stored in a table in 'analytics' dataset.
A view is created in the 'BI' dataset.
Uses a standard function tool (non-LRF). Includes NULL filtering.
"""

import traceback
from typing import Dict, List, Any, Optional
import datetime
import uuid
import time

import numpy as np
import pandas as pd
from google.adk.agents.llm_agent import LlmAgent
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Assuming Prophet is installed: pip install prophet
try:
    from prophet import Prophet
except ImportError:
    print("Prophet library not found. Please install it: pip install prophet")
    # Define a dummy class if Prophet is not installed
    class Prophet:
        def __init__(self, *args, **kwargs): pass
        def fit(self, df): pass
        def make_future_dataframe(self, periods): return pd.DataFrame({'ds': pd.to_datetime(['2024-01-01'])})
        def predict(self, future): return pd.DataFrame({'ds': future['ds'], 'yhat': [0]*len(future)})

# --- Constants ---
GEMINI_MODEL: str = "gemini-1.5-flash" # Use a model known for good function calling
DEFAULT_BQ_LOCATION: str = "US"
ANALYTICS_DATASET_ID: str = "analytics" # New dataset for forecast table
BI_DATASET_ID: str = "BI"             # New dataset for forecast view

# --- Helper Function to Ensure Dataset Exists ---
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
        client.create_dataset(dataset, exists_ok=True) # exists_ok=True handles race conditions
        print(f"[Dataset Check] Dataset '{project_id}.{dataset_id}' created.")
    except Exception as e:
        print(f"[Dataset Check] Error checking/creating dataset '{project_id}.{dataset_id}': {e}")
        raise # Re-raise the exception to stop the process if dataset creation fails critically


# --- Standard Tool Function for Forecasting ---

def forecast_sales_to_bq_table_and_view(
    project_id: str,
    source_dataset_name: str, # Dataset containing historical data
    source_table_name: str,   # Table containing historical data
    forecast_table_name: str, # Name for the new table in 'analytics' dataset
    forecast_view_name: str,  # Name for the new view in 'BI' dataset
    date_column: str = 'order_date',
    sales_column: str = 'total_amount',
    forecast_periods: int = 30
): # REMOVED return type hint
    """
    Fetches sales data (filtering NULLs), forecasts using Prophet,
    stores forecast in a permanent table in the 'analytics' dataset,
    and creates a view in the 'BI' dataset. Returns a status dictionary.
    """
    full_source_table_name = f"`{project_id}.{source_dataset_name}.{source_table_name}`"
    full_forecast_table_name = f"`{project_id}.{ANALYTICS_DATASET_ID}.{forecast_table_name}`"
    full_forecast_view_name = f"`{project_id}.{BI_DATASET_ID}.{forecast_view_name}`"

    print(f"[Forecast Tool] Starting forecast process...")
    print(f"  Source: {full_source_table_name}")
    print(f"  Output Table: {full_forecast_table_name}")
    print(f"  Output View: {full_forecast_view_name}")

    try:
        # Initialize a BigQuery client
        client = bigquery.Client(project=project_id)
        print(f"[Forecast Tool] BigQuery client initialized.")

        # Ensure output datasets exist
        ensure_dataset_exists(client, project_id, ANALYTICS_DATASET_ID, DEFAULT_BQ_LOCATION)
        ensure_dataset_exists(client, project_id, BI_DATASET_ID, DEFAULT_BQ_LOCATION)

        # Construct the SQL query to fetch, FILTER, and aggregate sales data
        query = f"""
            SELECT
                DATE({date_column}) AS ds,
                SUM({sales_column}) AS y
            FROM
                {full_source_table_name}
            WHERE
                {date_column} IS NOT NULL
                AND {sales_column} IS NOT NULL
            GROUP BY
                ds
            ORDER BY
                ds
        """

        print(f"[Forecast Tool] Fetching and filtering historical data...")
        df = client.query(query).to_dataframe()

        if df.empty:
            print("[Forecast Tool] No valid historical data found after filtering NULLs. Aborting.")
            return {"status": "error", "message": f"No valid (non-NULL in {date_column} or {sales_column}) historical data found in the source table. Cannot create forecast."}
        print(f"[Forecast Tool] Fetched {len(df)} rows of valid historical data.")

        # Initialize and fit the Prophet model
        print("[Forecast Tool] Initializing and fitting Prophet model...")
        model = Prophet()
        model.fit(df)
        print("[Forecast Tool] Prophet model fitted.")

        # Create a future dataframe for forecasting
        print(f"[Forecast Tool] Creating future dataframe for {forecast_periods} periods...")
        future = model.make_future_dataframe(periods=forecast_periods)

        # Make the forecast
        print("[Forecast Tool] Generating forecast...")
        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df.rename(columns={'ds': 'forecast_date', 'yhat': 'forecasted_sales',
                                    'yhat_lower': 'forecast_lower_bound', 'yhat_upper': 'forecast_upper_bound'}, inplace=True)
        print("[Forecast Tool] Forecast generated.")

        # Load forecast data into the permanent table in 'analytics' dataset
        forecast_table_ref = client.dataset(ANALYTICS_DATASET_ID).table(forecast_table_name)
        print(f"[Forecast Tool] Preparing to load forecast data to permanent table: {full_forecast_table_name}")

        # Overwrite the table each time
        job_config_load = bigquery.LoadJobConfig(autodetect=True, write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        load_job = client.load_table_from_dataframe(forecast_df, forecast_table_ref, job_config=job_config_load)
        print(f"[Forecast Tool] Load job to '{ANALYTICS_DATASET_ID}.{forecast_table_name}' started: {load_job.job_id}")
        load_job.result()  # Wait for the load job to complete
        print(f"[Forecast Tool] Forecast data loaded to permanent table {full_forecast_table_name}.")

        # Construct the SQL query to create the view in 'BI' dataset
        print(f"[Forecast Tool] Creating/Replacing view {full_forecast_view_name}...")
        # View selects from the permanent table in 'analytics'
        view_query = f"""
            CREATE OR REPLACE VIEW {full_forecast_view_name}
            OPTIONS(
                description="Sales forecast generated by Prophet model",
                labels=[("data_source","prophet_forecast")]
                )
            AS
            SELECT forecast_date, forecasted_sales, forecast_lower_bound, forecast_upper_bound
            FROM {full_forecast_table_name} -- Select from the permanent table
            ORDER BY forecast_date
        """

        # Execute the query to create the view
        query_job = client.query(view_query)
        print(f"[Forecast Tool] Create view job started: {query_job.job_id}")
        query_job.result()  # Wait for the query job to complete
        print(f"[Forecast Tool] View {full_forecast_view_name} created/replaced successfully.")

        # Return success dictionary
        return {
            "status": "success",
            "message": f"Forecast successfully generated and stored in table {full_forecast_table_name} and view {full_forecast_view_name}.",
            "forecast_table": full_forecast_table_name.strip('`'), # Return clean names
            "forecast_view": full_forecast_view_name.strip('`')
        }

    except Exception as e:
        print(f"[Forecast Tool] ERROR during forecasting process: {traceback.format_exc()}")
        # Return error dictionary
        return {"status": "error", "message": f"Failed during forecasting process: {str(e)}"}

    # No finally block needed for temp table cleanup


# --- Simple Forecasting Agent Definition ---
root_agent = LlmAgent(
    name="SimpleSalesForecastingAgent",
    model=GEMINI_MODEL,
    instruction=f"""
    You are an agent specialized in generating sales forecasts using historical data stored in BigQuery.
    Your goal is to create a forecast table in the '{ANALYTICS_DATASET_ID}' dataset and a corresponding view in the '{BI_DATASET_ID}' dataset.

    When the user asks for a sales forecast, follow these steps:

    1.  **Identify Parameters:** Determine the required information from the user or context:
        - Google Cloud `project_id`.
        - The `source_dataset_name` containing the historical data table.
        - The `source_table_name` with the historical sales data.
        - The desired `forecast_table_name` for the new table in the '{ANALYTICS_DATASET_ID}' dataset.
        - The desired `forecast_view_name` for the new view in the '{BI_DATASET_ID}' dataset.
        - Optionally, the user might specify `date_column` (default 'order_date'), `sales_column` (default 'total_amount'), or `forecast_periods` (default 30 days). If not specified, use the defaults. Ask the user if any mandatory parameters (`project_id`, `source_dataset_name`, `source_table_name`, `forecast_table_name`, `forecast_view_name`) are missing.

    2.  **Execute Forecast:** Call the `forecast_sales_to_bq_table_and_view` tool directly. This tool automatically filters NULLs, creates the datasets if needed, loads the forecast to a table in '{ANALYTICS_DATASET_ID}', and creates a view in '{BI_DATASET_ID}'.
        - Pass all the identified parameters: `project_id`, `source_dataset_name`, `source_table_name`, `forecast_table_name`, `forecast_view_name`, `date_column`, `sales_column`, `forecast_periods`.

    3.  **Report Result:** The `forecast_sales_to_bq_table_and_view` tool will perform the entire operation and return a final status dictionary (containing 'status' and 'message'). Present this final status dictionary clearly to the user as the outcome of the forecasting operation. If successful, mention the name of the table created in '{ANALYTICS_DATASET_ID}' and the view created in '{BI_DATASET_ID}'.
    """,
    description=(
        f"Generates a sales forecast using Prophet and BigQuery data (filtering NULLs), storing the result in a table "
        f"in the '{ANALYTICS_DATASET_ID}' dataset and creating a view in the '{BI_DATASET_ID}' dataset."
    ),
    tools=[
        forecast_sales_to_bq_table_and_view # Use the updated function directly as a tool
    ],
    output_key="forecast_final_status",
)