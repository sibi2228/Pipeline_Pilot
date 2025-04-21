# BigQuery Raw-to-DW Transform Agent

## Description

This repository contains a Python-based agent designed to perform transformation tasks within Google BigQuery. Specifically, it orchestrates a multi-step process to read data from a 'raw' dataset, merge it, enrich it by joining with lookup tables from a 'dw' (data warehouse) dataset using intermediate temporary tables, and finally load the transformed data into a final table within the 'dw' dataset.

This agent leverages the Google Agent Development Kit (ADK) and utilizes several specialized tools to manage the workflow efficiently.

## Agent: `LookupJoinTempTableAgent`

### Description

The `LookupJoinTempTableAgent` acts as a workflow manager for transforming raw data into an enriched format suitable for a data warehouse. It uses a sequence of tools to handle merging, joining, and loading data between BigQuery datasets, employing temporary tables to manage intermediate states and ensure atomicity of steps.

### Use Case & Functionality

The agent executes the following sequence, driven by its internal instruction and the available tools:

1.  **Receive Inputs:** Requires the Google Cloud `project_id` and the name of the final `destination_table_id` where the transformed data will be stored in the `dw` dataset.
2.  **Read & Merge Raw Data:**
    *   Calls the `read_merge_write_temp` tool.
    *   Reads *all* tables found within the specified `raw` dataset (default: `raw`).
    *   Concatenates the data from these tables into a single pandas DataFrame.
    *   Writes this merged DataFrame to a *first* temporary BigQuery table (e.g., `temp_merged_data_<uuid>`) within the `raw` dataset.
3.  **Join with Lookups:**
    *   Calls the `join_lookups_write_temp` tool, using the ID of the first temporary table.
    *   Reads data from the first temporary table.
    *   Reads lookup data from predefined tables in the `dw` dataset:
        *   `country_lookup` (default name)
        *   `product_catalouge` (default name)
    *   Performs **left joins**:
        *   Merges country data based on the `country` column in the source and the `country_code` column in the lookup table.
        *   Merges product data based on the `product` column (assuming it exists in both source and lookup).
    *   Handles cases where lookup tables might be empty or required join columns are missing in the source data.
    *   Writes the resulting joined DataFrame to a *second* temporary BigQuery table (e.g., `temp_lookup_data_<uuid>`) within the `raw` dataset.
    *   Cleans up (deletes) the *first* temporary table upon successful completion or failure of this step.
4.  **Load to Data Warehouse:**
    *   Calls the `load_temp_table_to_dw` tool, using the ID of the second temporary table.
    *   Reads the fully joined data from the second temporary table.
    *   Loads this data into the final `destination_table_id` (provided by the user) within the `dw` dataset (default: `dw`).
    *   The default write mode is `WRITE_TRUNCATE` (overwriting the destination table), but this can potentially be configured.
    *   Cleans up (deletes) the *second* temporary table upon successful completion or failure of this step.
5.  **Return Status:** Returns a dictionary containing the status (`success`, `error`, `warning`), message, and details (like rows loaded, job ID) of the final load operation.

## Installation

To run this agent and its underlying tools, you need to install the necessary Python libraries.


1.  **Install required libraries:**
    ```bash
    pip install google-adk google-cloud-bigquery pandas numpy pyarrow
    ```
    *   `google-adk`: The Google Agent Development Kit framework.
    *   `google-cloud-bigquery`: For interacting with BigQuery datasets and tables.
    *   `pandas`: For data manipulation (merging, joining).
    *   `numpy`: Often a dependency for pandas.
    *   `pyarrow`: Required by the BigQuery client library for efficient DataFrame transfers.
    *   (`uuid`, `traceback`, `datetime`, `typing` are built-in Python libraries).

## Configuration

This agent interacts heavily with Google Cloud Platform (GCP) services, specifically BigQuery. Proper authentication and configuration are essential.

1.  **GCP Authentication:** The environment where you run the agent must be authenticated to GCP. The recommended method is using Application Default Credentials (ADC). Run the following command in your terminal and follow the prompts:
    ```bash
    gcloud auth application-default login
    ```
    Alternatively, you can download a service account key file and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to its path:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json" # Linux/macOS
    # set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\keyfile.json" # Windows (cmd)
    # $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\keyfile.json" # Windows (PowerShell)
    ```

2.  **GCP Project ID:** The agent requires the `project_id` as an input parameter during execution. **Crucially, the GCP project associated with your authenticated credentials (from step 1) MUST contain:**
    *   The `raw` dataset (default name: `raw`) with the source tables.
    *   The `dw` dataset (default name: `dw`) where the final table will be created/overwritten.
    *   The lookup tables (`country_lookup`, `product_catalouge` by default) within the `dw` dataset.
    *   Permissions to list tables, read tables, create tables (temporary), load data, and delete tables within the specified datasets.

3.  **`.env` File (Optional but Recommended):** If you manage environment variables like the service account path or potentially override default dataset names (though the agent currently uses hardcoded defaults), you can use a `.env` file in the project's root directory.
    ```plaintext
    # .env example
    GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    # RAW_DATASET_ID="your_raw_dataset" # Example if you modify agent to use env vars
    # DW_DATASET_ID="your_dw_dataset"   # Example if you modify agent to use env vars
    ```
    Remember to load these variables (e.g., using `python-dotenv`) in your execution script if you use this method. **Ensure any settings correspond to the correct GCP project.**

## Google Agent Development Kit (ADK)

This agent is built using the Google Agent Development Kit (ADK). The ADK provides the framework (`LlmAgent`) and tools for defining agent instructions, tools, and execution flow. For more detailed information on the ADK, including how to run agents, advanced configuration, and best practices, please refer to the official **Google Agent Development Kit Documentation** *(<- Consider adding the actual link here if available)*.

## Usage

The `agent.py` file defines the `LookupJoinTempTableAgent` instance as `root_agent`. To use it:

1.  Ensure your environment is configured and authenticated correctly (see Configuration section).
2.  Import the `root_agent` into your main execution script.
3.  Use the ADK's runtime mechanisms to invoke the agent, providing the required `project_id` and `destination_table_id` inputs.


## Running through ADK
1. use adk web to run this agent and select ingestion agent from the dropdown in adk UI
2. please refer to prompts.md for example prompts.


**Example (Conceptual - actual execution depends on your ADK setup):**

```python
# Example execution script (e.g., run_transform.py)
from google.adk.runtime import AgentInteraction # Or appropriate ADK execution method
from agent import root_agent # Import the agent defined in agent.py
# from dotenv import load_dotenv # If using .env

# load_dotenv() # Load environment variables from .env if used

# --- Define Inputs for the Agent ---
# Replace with your actual GCP project ID and desired final table name
gcp_project_id = "your-gcp-project-id"
final_dw_table_name = "fact_sales_enriched" # Example final table name

agent_inputs = {
    "project_id": gcp_project_id,
    "destination_table_id": final_dw_table_name
}

# --- Execute the Agent using ADK's Runtime ---
# The exact invocation might differ based on your ADK setup
# Using str(agent_inputs) assumes the agent expects a string representation
# or that the ADK handles dictionary inputs appropriately. Adjust as needed.
interaction = AgentInteraction(agent=root_agent, request=str(agent_inputs))
result = interaction.run() # Or interaction.invoke(), etc.

# --- Print the Final Result ---
# The agent is designed to return the dictionary from the last step (load_temp_table_to_dw)
print("Agent Execution Result:")
print(result.response)

