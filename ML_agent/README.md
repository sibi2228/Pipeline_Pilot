# ML Sales Forecasting Agent (Prophet & Regression)

## Description

This repository contains a Python-based agent designed to perform and orchestrate time-series sales forecasting using two distinct methodologies:  Prophet and Regression (RandomForest or XGBoost). The agent runs these forecasts sequentially, processes data from Google BigQuery, stores the forecast results back into BigQuery tables and views, and saves visualization plots locally.

This agent leverages the Google Agent Development Kit (ADK) for its structure and execution flow.

## Agent: `SequentialForecastingAgent`

### Description

The `SequentialForecastingAgent` acts as an orchestrator for a multi-step machine learning workflow. Built using the Google Agent Development Kit (`google-adk`), it manages the sequential execution of two forecasting tools: one based on Prophet and another based on regression models. Its primary role is to interpret user requests, invoke the appropriate tools with the correct parameters in the specified order, and compile a final summary report.

### Use Case & Functionality

The agent executes the following workflow, driven by its internal instruction and the available tools:

1.  **Identify Parameters:** Extracts necessary parameters from the user request or context. These include source data details (project, dataset, table, date/sales columns), output destinations (table/view names for both Prophet and Regression results), forecast horizon (`forecast_periods`), and model-specific configurations (Prophet's `country_code_for_holidays`, Regression's `model_type` and optional `feature_columns`). Defaults are used if parameters are not provided.
2.  **Execute Prophet Forecast:**
    *   Calls the `forecast_sales_to_bq_table_and_view` tool first.
    *   This tool fetches data, aggregates daily sales, fits a Prophet model (optionally incorporating country holidays), generates forecasts, saves results to a specified table in the `analytics` dataset and a view in the `BI` dataset within BigQuery, and saves a plot locally.
    *   Stores the status dictionary returned by the tool.
3.  **Execute Regression Forecast:**
    *   *After* the Prophet forecast completes, calls the `regression_forecast_to_bq_table_and_view` tool.
    *   This tool fetches data, aggregates daily sales, engineers time-based features (and optionally uses additional provided features), trains a specified regression model (RandomForest or XGBoost), generates forecasts, saves results to a specified table (`analytics` dataset) and view (`BI` dataset) in BigQuery, and saves a plot locally.
    *   Stores the status dictionary returned by this tool.
4.  **Compile Final Report:**
    *   Once both tools have finished, the agent compiles a summary report containing the status, messages, output table/view names, and plot filenames from both the Prophet and Regression tool executions.
    *   This summary is returned as the final output of the agent.

## Tools Overview

The agent utilizes the following specialized tools:

1.  **`forecast_sales_to_bq_table_and_view`:**
    *   **Purpose:** Performs time-series forecasting using the Prophet library.
    *   **Input:** Source BigQuery table details, output table/view names, date/sales columns, forecast periods, optional country code for holidays.
    *   **Output:** Writes forecast data (date, forecast, lower/upper bounds) to a BigQuery table (`analytics` dataset) and creates/replaces a view (`BI` dataset). Saves a plot locally. Returns a status dictionary.

2.  **`regression_forecast_to_bq_table_and_view`:**
    *   **Purpose:** Performs time-series forecasting using regression models (RandomForest or XGBoost).
    *   **Input:** Source BigQuery table details, output table/view names, date/sales columns, forecast periods, model type, optional additional feature columns.
    *   **Output:** Writes forecast data (date, forecast) to a BigQuery table (`analytics` dataset) and creates/replaces a view (`BI` dataset). Saves a plot locally. Returns a status dictionary.


1.  **Install required libraries:**
    ```bash
    pip install google-adk google-cloud-bigquery pandas numpy prophet scikit-learn xgboost matplotlib
    ```
    *   `google-adk`: The Google Agent Development Kit framework.
    *   `google-cloud-bigquery`: For interacting with BigQuery.
    *   `pandas`: For data manipulation.
    *   `numpy`: Numerical library (dependency for pandas, scikit-learn).
    *   `prophet`: Facebook's time-series forecasting library. *(Note: Prophet installation can sometimes require specific build tools or dependencies. Refer to the official Prophet installation guide if you encounter issues.)*
    *   `scikit-learn`: For the RandomForestRegressor model and metrics.
    *   `xgboost`: For the XGBoost regression model.
    *   `matplotlib`: For generating forecast plots. *(Note: `plt.switch_backend('Agg')` is used, suitable for non-interactive environments.)*

    *(The code includes `try...except ImportError` blocks for ML libraries, allowing it to run with dummy classes if they are not installed, but forecasting functionality will be disabled.)*

## Configuration

This agent interacts with Google Cloud Platform (GCP) services and the local filesystem.

1.  **GCP Authentication:** The environment where you run the agent must be authenticated to GCP. The recommended method is using Application Default Credentials (ADC):
    ```bash
    gcloud auth application-default login
    ```
    Alternatively, set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account key file.

2.  **GCP Project ID:** The agent needs the GCP `project_id`. While a default (`us-con-gcp-sbx-0000489-032025`) is hardcoded in the script, it's best practice to provide it as input during execution. **Crucially, the GCP project associated with your authenticated credentials MUST contain:**
    *   The source dataset and table specified in the inputs.
    *   Permissions to:
        *   Read data from the source table.
        *   Create datasets (`analytics`, `BI`) if they don't exist (or ensure they exist beforehand).
        *   Create/overwrite tables within the `analytics` dataset.
        *   Create/replace views within the `BI` dataset.

3.  **BigQuery Datasets:** The agent is configured to write tables to the `analytics` dataset and views to the `BI` dataset. Ensure these datasets exist in your target GCP project and location (`US` by default), or that the authenticated user/service account has permissions to create them.

4.  **Local Plot Directory:** The agent saves forecast plots to a local directory named `forecast_plots` (relative to where the script is run). Ensure the execution environment has write permissions to create this directory and save files within it.

5.  **`.env` File (Optional):** You can manage environment variables like `GOOGLE_APPLICATION_CREDENTIALS` using a `.env` file and a library like `python-dotenv` in your execution script (not part of the agent code itself).

## Google Agent Development Kit (ADK)

This agent is built using the Google Agent Development Kit (ADK). The ADK provides the `LlmAgent` class and the framework for defining agent instructions, tools, and managing execution flow. For more detailed information on the ADK, including running agents and advanced features, please refer to the official **Google Agent Development Kit Documentation** *(<- Consider adding the actual link here if available)*.

## Usage

The `agent.py` file defines the `SequentialForecastingAgent` instance as `root_agent`. To use it:

1.  Ensure your environment is configured and authenticated correctly (see Configuration section).
2.  Import the `root_agent` into your main execution script.
3.  Use the ADK's runtime mechanisms to invoke the agent, providing the required parameters as a dictionary or structured input.


## Running through ADK
1. use adk web to run this agent and select ingestion agent from the dropdown in adk UI
2. please refer to prompts.md for example prompts.

**Example (Conceptual - actual execution depends on your ADK setup):**

```python
# Example execution script (e.g., run_forecasting.py)
from google.adk.runtime import AgentInteraction # Or appropriate ADK execution method
from agent import root_agent # Import the agent defined in agent.py
# from dotenv import load_dotenv # If using .env

# load_dotenv() # Load environment variables from .env if used

# --- Define Inputs for the Agent ---
# Adjust these values based on your specific BQ setup and requirements
agent_inputs = {
    "project_id": "your-gcp-project-id", # Replace with your project ID
    "source_dataset_name": "your_source_dataset", # e.g., "dw"
    "source_table_name": "your_source_table",     # e.g., "fact_sales"
    "prophet_forecast_table_name": "prophet_sales_forecast", # Optional, defaults provided
    "prophet_forecast_view_name": "v_prophet_sales_forecast", # Optional, defaults provided
    "regression_forecast_table_name": "regression_sales_forecast", # Optional, defaults provided
    "regression_forecast_view_name": "v_regression_sales_forecast", # Optional, defaults provided
    "date_column": "order_date", # Optional, default 'order_date'
    "sales_column": "total_amount", # Optional, default 'total_amount'
    "forecast_periods": 90, # Optional, default 30
    "country_code_for_holidays": "US", # Optional, for Prophet only (e.g., 'US', 'GB', 'DE')
    "model_type": "XGBoost", # Optional, 'RandomForest' or 'XGBoost', default 'RandomForest'
    "feature_columns": ["day_of_week", "is_promo_period"] # Optional, list of extra features for Regression
}



