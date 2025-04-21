# Pipeline Pilot: AI-Powered ELT & Analytics Agent System

## Description

Pipeline Pilot is an AI agent-driven system designed to automate the entire ELT (Extract, Load, Transform) process, extending into Machine Learning analytics and visualization. Developed using the Google Agent Development Kit (ADK), this project demonstrates how multiple specialized agents can collaborate to handle a complex data pipeline workflow within the Google Cloud Platform ecosystem.

The pipeline covers:
1.  **Ingestion:** Fetching raw data (CSVs) from Google Cloud Storage (GCS).
2.  **Loading & Validation:** Loading data into a 'raw' BigQuery dataset with basic validation and cleaning.
3.  **Transformation:** Transforming raw data by merging and enriching it with lookup tables, storing the result in a 'data warehouse' (dw) BigQuery dataset.
4.  **ML Analytics:** Performing time-series sales forecasting using both Prophet and Regression models on the transformed data, storing results in 'analytics' and 'BI' BigQuery datasets/views.
5.  **Visualization & Notification:** Generating forecast plots and emailing them as attachments.

## Acknowledgement

Developed by **Sibi Sukanesh** and **Regi Clinton** for a **Deloitte** hackathon project.

## Features

*   **Automated Ingestion:** Seamlessly ingests CSV data from GCS to BigQuery.
*   **Data Transformation:** Merges and enriches data using lookup tables within BigQuery.
*   **Dual Forecasting Models:** Implements both Prophet and Regression (RandomForest/XGBoost) for time-series forecasting.
*   **BigQuery Integration:** Leverages BigQuery for storing raw, transformed, and forecast data, including tables and views.
*   **Automated Visualization:** Generates forecast plots using Matplotlib.
*   **Email Notifications:** Sends forecast charts via email using SMTP (Gmail configured).
*   **Agent-Based Orchestration:** Uses Google ADK to manage the workflow with specialized agents.
*   **Modular Design:** Each step (Ingestion, Transform, ML, Email) is handled by a dedicated agent/tool.

## Architecture & Workflow

The system employs a sequence of agents, each responsible for a specific part of the pipeline:

1.  **`Ingestion_Agent`**: Monitors a GCS location, validates incoming CSV files (checks for nulls, duplicates), cleans them, and loads the data into a `raw` dataset in BigQuery.
2.  **`Transform_Agent` (`LookupJoinTempTableAgent`)**: Reads tables from the `raw` dataset, merges them, joins with lookup tables (`country_lookup`, `product_catalouge`) from the `dw` dataset using temporary tables, and loads the final transformed data into a target table within the `dw` dataset.
3.  **`ML_Agent` (`SequentialForecastingAgent`)**: Reads data from the `dw` dataset, performs time-series forecasting sequentially using:
    *   **Prophet:** Stores results in `analytics` table and `BI` view, saves plot locally.
    *   **Regression (RF/XGBoost):** Stores results in `analytics` table and `BI` view, saves plot locally.
4.  **`Forecast_Email_Agent` (`forecast_chart_email_agent`)**: Reads forecast data from a specified table in the `analytics` dataset, generates a visualization plot (`forecast_chart.png`), and emails this plot as an attachment using SMTP.

## Agents Overview

*   ### `Ingestion_Agent`
    *   **Description:** Automates ingestion of CSV files from GCS to a BigQuery `raw` dataset.
    *   **Functionality:** Lists files, reads CSVs, validates (nulls, duplicates), cleans (drops invalid rows), derives schema/table names, loads to BigQuery (`WRITE_TRUNCATE`).
*   ### `Transform_Agent` (`LookupJoinTempTableAgent`)
    *   **Description:** Transforms data from the `raw` dataset to the `dw` dataset using lookups.
    *   **Functionality:** Merges tables from `raw` into a temp table, joins with `country_lookup` and `product_catalouge` from `dw` into a second temp table, loads the result to the final `dw` table (`WRITE_TRUNCATE`), cleans up temp tables.
*   ### `ML_Agent` (`SequentialForecastingAgent`)
    *   **Description:** Orchestrates sequential execution of Prophet and Regression forecasting models.
    *   **Functionality:** Fetches data, trains Prophet (with optional holidays), trains Regression (RF/XGBoost with time features), generates forecasts, saves results to `analytics` tables and `BI` views in BigQuery, saves plots locally.
*   ### `Forecast_Email_Agent` (`forecast_chart_email_agent`)
    *   **Description:** Generates a forecast chart from BigQuery data and emails it.
    *   **Functionality:** Queries forecast table, creates a plot using Matplotlib, sends an email via SMTP with the plot as an attachment. **Requires App Password configuration.**

## Prerequisites

*   **Python:** Version 3.8 or higher recommended.
*   **Google Cloud SDK (`gcloud`):** Required for authenticating with GCP. Installation Guide
*   **Google Cloud Platform (GCP) Project:** Access to a GCP project with the following APIs enabled:
    *   Google Cloud Storage API
    *   BigQuery API
*   **GCP Permissions:** Your user account or service account needs appropriate permissions in the GCP project to:
    *   Read/Write to GCS buckets.
    *   Create/Read/Write/Delete BigQuery datasets and tables (specifically for `raw`, `dw`, `analytics`, `BI` datasets).
    *   Run BigQuery jobs.
*   **Gmail Account (for Email Agent):** A Gmail account to send emails from. **Requires 2-Step Verification enabled and an App Password.**

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url> # Replace with your repo URL
    cd pipeline_pilot # Or your repository directory name
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Setup Google Cloud SDK (`gcloud`):**
    *   If not already installed, follow the Installation Guide.
    *   Initialize `gcloud` and log in:
        ```bash
        gcloud init
        ```
        *(Follow prompts to choose your account and set the default GCP project for this project)*
    *   Set up Application Default Credentials (ADC) for authentication:
        ```bash
        gcloud auth application-default login
        ```
        *(This allows the Python client libraries to automatically find your credentials)*

4.  **Setup Google Agent Development Kit (ADK):**
    *   The ADK will be installed via `pip` in the next step.
    *   The ADK provides the framework for defining and running the agents. You will primarily interact with it using the `adk web` command.
    *   Refer to the official Google ADK Documentation for more details (if available).

## Installation

Install all necessary Python libraries using the provided `requirements.txt` file (or create one with the contents below):

*   **Create `requirements.txt` (if it doesn't exist):**
    ```plaintext
    # requirements.txt
    google-adk
    google-cloud-storage
    google-cloud-bigquery
    pandas
    numpy
    prophet
    scikit-learn
    xgboost
    matplotlib
    pyarrow
    # Add python-dotenv if you plan to use .env files
    # python-dotenv
    ```

*   **Install Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `prophet` installation might require additional system dependencies. Consult the Prophet documentation if you encounter issues.)*

## Configuration

1.  **GCP Project ID:** Ensure the `gcloud` CLI is configured with the correct default project (done during `gcloud init`). Some agent scripts might have hardcoded project IDs (like in `ML_agent`) - **it's best practice to modify these or pass the correct project ID during agent execution.**
2.  **BigQuery Datasets:** The agents assume the existence of datasets named `raw`, `dw`, `analytics`, and `BI` in your default GCP project and location (often `US`). Ensure these exist or that the authenticated user has permissions to create them. Modify agent code if you use different dataset names.
3.  **Email Agent Configuration (CRITICAL):**
    *   Open the `Forecast_email_Agent/agent.py` file (or wherever the `send_email` tool is defined).
    *   **Change `from_address`:** Replace the hardcoded sender email (`romeojuli1997@gmail.com`) with your actual Gmail address.
    *   **Generate & Set App Password:**
        1.  Enable 2-Step Verification on the sender Google Account.
        2.  Go to Google Account -> Security -> 2-Step Verification -> App passwords.
        3.  Generate a new App Password (select App: Mail, Device: Other).
        4.  **Copy the 16-character password.**
        5.  **Replace the hardcoded `password` variable in the `send_email` function with this App Password.**
    *   **Security:** Avoid committing App Passwords directly. Use environment variables or a secure secrets manager for production.
4.  **`.env` File (Optional but Recommended):**
    *   For managing sensitive information like the App Password or `GOOGLE_APPLICATION_CREDENTIALS` path (if using service accounts), create a `.env` file in the root directory:
        ```plaintext
        # .env example
        # GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
        GMAIL_APP_PASSWORD="your_16_character_app_password"
        SENDER_EMAIL="your_sender_email@gmail.com"
        ```
    *   You would need to install `python-dotenv` (`pip install python-dotenv`) and modify the agent code (specifically the `send_email` tool) to load these variables using `load_dotenv()` and `os.getenv()`.

## Usage (ADK Web UI)

The primary way to interact with the agents in this project is via the ADK Web UI:

1.  **Start the ADK Web Server:**
    *   Make sure your virtual environment is active and you are in the project's root directory.
    *   Run the command:
        ```bash
        adk web
        ```
2.  **Access the UI:**
    *   Open your browser and navigate to the URL provided (usually `http://127.0.0.1:8080`).
3.  **Select an Agent:**
    *   Use the dropdown menu in the ADK Web UI to select the specific agent you want to run for a task:
        *   `Ingestion_Agent`: For ingesting data from GCS.
        *   `LookupJoinTempTableAgent`: For transforming raw data.
        *   `SequentialForecastingAgent`: For running ML forecasts.
        *   `forecast_chart_email_agent`: For generating and emailing a forecast chart.
4.  **Provide Input:**
    *   In the input text area, provide a natural language prompt describing the task and necessary parameters for the selected agent. Examples:
        *   **For Ingestion:** "Ingest CSV files from GCS bucket `my-raw-data-bucket` folder `incoming_sales/` into BigQuery."
        *   **For Transformation:** "Transform data in project `my-gcp-project` by merging tables in `raw` dataset and joining with lookups in `dw` dataset, storing the result in table `dw.fact_sales_transformed`."
        *   **For ML:** "Run Prophet and XGBoost forecasts using data from `my-gcp-project.dw.fact_sales_transformed`, date column `sale_date`, sales column `amount`. Forecast 60 periods. Use US holidays for Prophet. Store results with base names `sales_forecast_q1`."
        *   **For Email:** "Create a forecast chart from `my-gcp-project.analytics.prophet_sales_forecast_q1` and email it to `manager@example.com` with subject 'Q1 Sales Forecast Chart'."
5.  **Run and Monitor:**
    *   Submit the request. The ADK Web UI will show the agent's execution trace, including tool calls and final output/status. Check GCP console (BigQuery, GCS) and email inbox as needed to verify results.

---
