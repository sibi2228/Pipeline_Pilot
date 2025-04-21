# GCS CSV to BigQuery Ingestion Agent

## Description

This repository contains a Python-based agent designed to automate the ingestion of CSV data from Google Cloud Storage (GCS) into Google BigQuery. It leverages the Google Agent Development Kit (ADK) and provides a robust tool for handling the ETL (Extract, Transform, Load) process.

## Agent: `Ingestion_Agent`

### Description

The `Ingestion_Agent` acts as an orchestrator for a data pipeline. It is built using the Google Agent Development Kit (`google-adk`) and utilizes a specialized internal tool (`process_gcs_to_bq`) to perform the heavy lifting of moving data from GCS to BigQuery. The agent's role is to receive the necessary parameters (GCS location) and invoke the tool, then report the outcome.

### Use Case & Functionality

The core functionality is encapsulated within the `process_gcs_to_bq` tool, which the `Ingestion_Agent` calls. This tool performs the following steps for all `.csv` files found within the specified GCS bucket and prefix:

1.  **List Files:** Identifies all `.csv` files in the target GCS path (`gs://<bucket_name>/<prefix>`).
2.  **Read Data:** Downloads and parses each CSV file into a pandas DataFrame. Handles potential parsing errors and empty files gracefully.
3.  **Validate Data:** Checks each DataFrame for:
    *   Presence of null values.
    *   Presence of duplicate rows.
4.  **Clean Data:** If validation issues are found:
    *   Rows containing *any* null values are dropped.
    *   Duplicate rows are dropped (after nulls are handled).
    *   Handles cases where cleaning results in an empty DataFrame.
5.  **Derive Table Name:** Automatically generates a valid BigQuery table name based on the original CSV filename (e.g., `my_data_file.csv` might become `my_data_file`). It sanitizes the name to comply with BigQuery naming rules.
6.  **Load to BigQuery:** Loads the cleaned DataFrame into the specified target BigQuery dataset (defaults to `raw`).
    *   Uses BigQuery's `autodetect=True` for schema inference.
    *   Uses `write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE`, meaning the target table will be overwritten if it already exists.
7.  **Report:** Returns a comprehensive dictionary detailing the overall success/warning/error status, a summary message, and the specific outcome (read, validate, clean, load status) for each processed file.

The `Ingestion_Agent` receives this report and presents it as the final output.

#
3.  **Install required libraries:**
    ```bash
    pip install google-adk google-cloud-storage google-cloud-bigquery pandas pyarrow
    ```
    *   `google-adk`: The Google Agent Development Kit framework.
    *   `google-cloud-storage`: For interacting with GCS.
    *   `google-cloud-bigquery`: For interacting with BigQuery.
    *   `pandas`: For reading CSVs and data manipulation.
    *   `pyarrow`: Required by the BigQuery client library for efficient DataFrame transfers.

## Configuration

This agent interacts with Google Cloud Platform (GCP) services. Proper authentication and configuration are crucial.

1.  **GCP Authentication:** The environment where you run the agent must be authenticated to GCP. The recommended method is using Application Default Credentials (ADC). Run the following command and follow the prompts:
    ```bash
    gcloud auth application-default login
    ```
    Alternatively, you can download a service account key file and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json" # Linux/macOS
    # set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\keyfile.json" # Windows (cmd)
    # $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\keyfile.json" # Windows (PowerShell)
    ```

2.  **GCP Project ID:** The agent's tool (`process_gcs_to_bq`) automatically detects the GCP Project ID from the authenticated environment (via the BigQuery client). **Ensure the credentials configured in the step above belong to the specific GCP project** that contains your source GCS bucket and target BigQuery dataset (`raw` by default).

3.  **`.env` File (Optional but Recommended):** If you prefer managing environment variables like the service account path, you can use a `.env` file in the project's root directory. Create a file named `.env` and add variables like:
    ```plaintext
    # .env
    GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
    # Add any other relevant environment variables if needed
    ```
    You would then need to use a library like `python-dotenv` in your execution script (not shown in `agent.py` itself) to load these variables before running the agent. **Make sure the path and any other settings in your `.env` file correspond to the correct GCP project you intend to use.**

## Google Agent Development Kit (ADK)

This agent is built using the Google Agent Development Kit (ADK). The ADK provides tools and structures for creating agents like this one. For more detailed information on the ADK, including how to run agents, advanced configuration, and best practices, please refer to the official **Google Agent Development Kit Documentation** *(<- Consider adding the actual link here if available)*.

## Usage

The provided `agent.py` file defines the `Ingestion_Agent` and its associated tool. To use it, you would typically:

1.  Import the `root_agent` (which is the `Ingestion_Agent` instance) into another Python script.
2.  Use the ADK's execution mechanisms to run the agent, providing the necessary inputs (like `bucket_name` and optional `prefix`) through the ADK's context or invocation methods.


## Running through ADK
1. use adk web to run this agent and select ingestion agent from the dropdown in adk UI
2. please refer to prompts.md for example prompts.



**Example (Conceptual - actual execution depends on your ADK setup):**

```python
# Example execution script (e.g., run_pipeline.py)
from google.adk.runtime import AgentInteraction # Or appropriate ADK execution method
from agent import root_agent # Import the agent defined in agent.py
# from dotenv import load_dotenv # If using .env

# load_dotenv() # Load environment variables from .env if used

# Define inputs for the agent
inputs = {
    "bucket_name": "your-gcs-bucket-name",
    "prefix": "path/to/your/csv/files/" # Optional, use "" for root
    # target_dataset_id defaults to "raw" based on agent instruction
}

# Execute the agent using ADK's runtime
interaction = AgentInteraction(agent=root_agent, request=str(inputs)) # Adapt based on ADK usage
result = interaction.run() # Or interaction.invoke(), etc.

# Print the summary report generated by the agent
print(result.response)

