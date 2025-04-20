# c:\Users\Rahul\Desktop\Hackathon\multi_tool_agent\agent.py

from google.adk.agents.llm_agent import LlmAgent
# Removed SequentialAgent and Agent imports as they are no longer needed here
import pandas as pd
from google.cloud import storage, bigquery
from io import StringIO
import pyarrow # Keep if push_to_bigquery might handle Arrow types implicitly
import re # Import regex for table ID sanitization

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash" # Use a model known for good function calling/orchestration
RAW_DATASET_ID = "raw" # Target dataset for loaded raw data
DW_DATASET_ID = "dw" # Keep if other potential tools/flows might use it

# --- Tool Functions (Keep them defined at the module level) ---

def read_csv_from_gcs(bucket_name: str, prefix: str = "") -> dict:
    """
    Reads all CSV files from a specified GCS bucket and optional prefix.
    Returns a dictionary where keys are filenames and values are file data as dictionaries (orient='list').
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(client.list_blobs(bucket_name, prefix=prefix)) # Get blobs matching prefix

        all_data = {}
        csv_files_found = [] # Keep track of CSV files processed

        for blob in blobs:
            # Process only .csv files, potentially filtering out "directory" placeholders if necessary
            if blob.name.endswith(".csv"):
                 # Basic check to avoid processing empty "folder" blobs if they exist
                 if blob.size == 0 and blob.name.endswith('/'):
                     print(f"Skipping likely directory placeholder: {blob.name}")
                     continue

                 # Check if it's directly under the prefix or in a subdirectory
                 # Let's read all CSVs under the prefix for simplicity now.
                 # If only direct children are needed, add:
                 # relative_path = blob.name[len(prefix):] if prefix else blob.name
                 # if '/' not in relative_path.strip('/'):
                 print(f"Reading file: {blob.name}")
                 content = blob.download_as_text()
                 # Handle potential empty CSV files gracefully
                 try:
                     df = pd.read_csv(StringIO(content))
                     all_data[blob.name] = df.to_dict(orient='list') # Store data as dict
                     csv_files_found.append(blob.name)
                 except pd.errors.EmptyDataError:
                     print(f"Warning: Skipping empty CSV file: {blob.name}")
                     all_data[blob.name] = {} # Represent empty file explicitly if needed
                     csv_files_found.append(blob.name) # Still acknowledge the file was found
                 except Exception as pd_err:
                     print(f"Error parsing CSV file {blob.name}: {pd_err}. Skipping.")
                     # Optionally return partial results or a more specific error

        if not csv_files_found:
            # Distinguish between no blobs found and no *CSV* blobs found
            if not blobs:
                 return {"status": "warning", "message": f"No objects found in gs://{bucket_name}/{prefix}"}
            else:
                 return {"status": "warning", "message": f"No CSV files found matching gs://{bucket_name}/{prefix}*.csv"}

        # Return dict {filename: data_dict}
        return {"status": "success", "data": all_data, "files_processed": csv_files_found}
    except Exception as e:
        print(f"ERROR: Error reading from GCS: {str(e)}") # Log error server-side
        # Provide a more informative error message back to the agent
        return {"status": "error", "message": f"Failed to read from GCS bucket '{bucket_name}' (prefix: '{prefix}'). Error: {str(e)}"}


def perform_data_quality_checks(data: dict, filename: str) -> dict:
    """
    Performs basic data quality checks on a DataFrame represented as a dictionary (orient='list').
    Includes the filename in the response for clarity. Handles empty data dicts.
    """
    try:
        # Handle case where the data dict might be empty (e.g., empty CSV)
        if not data:
            return {"status": "success", "filename": filename, "message": "Skipped DQ check for empty file."}

        # Convert dictionary back to DataFrame for checks
        df = pd.DataFrame(data)

        results = []
        # Check for null values
        if df.isnull().values.any():
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            results.append(f"Null values found in columns: {null_cols.to_dict()}")

        # Check for duplicate rows
        if df.duplicated().any():
            num_duplicates = df.duplicated().sum()
            results.append(f"{num_duplicates} duplicate rows found.")

        if not results:
            return {"status": "success", "filename": filename, "message": "Data quality checks passed."}
        else:
            # Return warning if checks identify issues but the process didn't crash
            return {"status": "warning", "filename": filename, "message": "Data quality issues found: " + " | ".join(results)}

    except ValueError as ve:
         # Catch errors during DataFrame creation (e.g., lists of different lengths)
         error_msg = f"Error performing data quality check for '{filename}': Invalid data structure. {str(ve)}"
         print(f"ERROR: {error_msg}")
         return {"status": "error", "filename": filename, "message": error_msg}
    except Exception as e:
        error_msg = f"Error during data quality check for '{filename}': {str(e)}"
        print(f"ERROR: {error_msg}")
        return {"status": "error", "filename": filename, "message": error_msg}


def push_to_bigquery(dataset_id: str, table_id: str, data: dict, filename: str) -> dict:
    """
    Pushes data for a single file to a specified BigQuery table.
    Creates or replaces the table. Handles empty data dicts.
    Args:
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID (will be created/replaced). Must be valid BQ name.
        data: The data for the file as a dictionary (orient='list').
        filename: The original filename (for logging/reference).
    Returns:
        A dictionary indicating the status and message.
    """
    # Validate table_id format (start with letter/underscore, contain letters, numbers, underscores)
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_id):
         error_msg = f"Invalid BigQuery table ID format: '{table_id}'. Must contain only letters, numbers, and underscores, and start with a letter or underscore."
         print(f"ERROR: {error_msg}")
         return {"status": "error", "filename": filename, "message": error_msg}

    try:
        client = bigquery.Client() # Assumes project is configured via ADC or env var
        project_id = client.project
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        table_ref = client.dataset(dataset_id).table(table_id)

        print(f"Attempting to load data for '{filename}' into table: {full_table_id}")

        # Handle case where the data dict might be empty (e.g., empty CSV, skip load)
        if not data:
            return {"status": "success", "filename": filename, "table_id": full_table_id, "message": "Input data is empty, load skipped."}

        # Convert dictionary (orient='list') back to DataFrame
        df = pd.DataFrame(data)
        # Double check DataFrame isn't empty after conversion (might happen with dict of empty lists)
        if df.empty:
             return {"status": "success", "filename": filename, "table_id": full_table_id, "message": "Input data resulted in an empty DataFrame, load skipped."}


        # Configure the load job
        job_config = bigquery.LoadJobConfig(
            autodetect=True,  # Automatically detect schema
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE # Replace table if exists
        )

        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for the job to complete

        return {
            "status": "success",
            "filename": filename,
            "table_id": full_table_id,
            "message": f"Data successfully loaded to {full_table_id}"
        }
    except ValueError as ve:
         # Catch errors during DataFrame creation (e.g., lists of different lengths)
         error_message = f"Error preparing data for BigQuery load for '{filename}' (Table: '{table_id}'): Invalid data structure. {str(ve)}"
         print(f"ERROR: {error_message}") # Log the error server-side
         return {"status": "error", "filename": filename, "table_id": full_table_id, "message": error_message}
    except Exception as e:
        # Attempt to provide more context in the error
        error_message = f"Error pushing data for '{filename}' to BigQuery table '{full_table_id}': {str(e)}"
        print(f"ERROR: {error_message}") # Log the error server-side
        return {
            "status": "error",
            "filename": filename,
            "table_id": full_table_id, # Report intended table
            "message": error_message
        }


# --- Root Agent (Single LLM Agent Orchestrator) ---
# This agent uses the functions directly as tools and follows instructions
# to orchestrate the pipeline steps.
root_agent = LlmAgent(
    name="GcsToBqPipelineAgent",
    model=GEMINI_MODEL,
    instruction=f"""
    You are an agent designed to orchestrate a data pipeline with the following steps:
    1. Read all CSV files from a specified Google Cloud Storage (GCS) location.
    2. Perform data quality checks on each file read.
    3. Load the data from files that pass quality checks into separate BigQuery tables.

    Follow these steps precisely:

    **Step 1: Read from GCS**
    - You will receive a `bucket_name` and an optional `prefix` from the user or context.
    - Call the `read_csv_from_gcs` tool with the provided `bucket_name` and `prefix`.
    - The tool returns a dictionary. Store this result. Let's call it `gcs_result`.
    - Check `gcs_result['status']`. If it's not 'success', report the `gcs_result['message']` and stop the pipeline.
    - If successful, the result contains `gcs_result['data']`, a dictionary where keys are filenames and values are the data dictionaries for each file. It also contains `gcs_result['files_processed']`, a list of filenames.

    **Step 2: Perform Data Quality Checks**
    - Initialize an empty dictionary to store validation results, e.g., `validation_summary = {{}}`.
    - Iterate through each `filename` in `gcs_result['files_processed']`.
    - For each `filename`, retrieve its corresponding data dictionary from `gcs_result['data'][filename]`.
    - Call the `perform_data_quality_checks` tool, passing the retrieved `data` dictionary and the current `filename` string.
    - Store the dictionary returned by the tool in your `validation_summary` using the `filename` as the key. For example: `validation_summary[filename] = dq_check_result`.

    **Step 3: Load to BigQuery (Conditional)**
    - Initialize an empty dictionary to store loading results, e.g., `load_summary = {{}}`.
    - Iterate through each `filename` for which a validation result exists in `validation_summary`.
    - Check the `status` within `validation_summary[filename]`.
    - **If `validation_summary[filename]['status']` is 'success':**
        a. Retrieve the original data dictionary for this file from `gcs_result['data'][filename]`.
        b. Derive a BigQuery `table_id` from the `filename`:
            - Remove any leading directory structure (e.g., 'path/to/your_file.csv' -> 'your_file.csv'). Get the base name.
            - Remove the '.csv' extension from the base name.
            - Replace any remaining non-alphanumeric characters (like hyphens, spaces) with underscores.
            - Ensure the resulting `table_id` is not empty and conforms to BigQuery naming rules (starts with letter/underscore, contains only letters, numbers, underscores). If derivation fails, record an error for this file in `load_summary` and skip to the next file.
        c. Call the `push_to_bigquery` tool with:
            - `dataset_id`: '{RAW_DATASET_ID}' or '{DW_DATASET_ID}' based on users input
            - `table_id`: the derived table_id from step 3b.
            - `data`: the data dictionary retrieved in step 3a.
            - `filename`: the original filename (key from the loop).
        d. Store the dictionary returned by the `push_to_bigquery` tool in your `load_summary` using the `filename` as the key.
    - **If `validation_summary[filename]['status']` is *not* 'success' (e.g., 'warning' or 'error'):**
        - Record an entry in `load_summary` for this `filename` indicating it was skipped due to the validation status. For example: `load_summary[filename] = {{'status': 'skipped', 'reason': validation_summary[filename]['message']}}`.

    **Step 4: Final Report**
    - Compile a final summary report containing:
        - The overall status ('Pipeline completed successfully', 'Pipeline completed with warnings/errors', 'Pipeline failed').
        - GCS Read Summary: Report the message from `gcs_result`. If successful, mention the number of files processed (`len(gcs_result['files_processed'])`).
        - Validation Summary: Report the status for each file based on `validation_summary`.
        - Load Summary: Report the status for each file based on `load_summary` (success, skipped, error), including target table IDs for successful loads.
    - Present this final summary report clearly.
    """,
    description="Orchestrates reading GCS CSVs, validating data, and loading valid files to BigQuery using direct tool calls.",
    tools=[
        read_csv_from_gcs,
        perform_data_quality_checks,
        push_to_bigquery
    ],
    output_key="pipeline_summary" # Store the final compiled report here
)

