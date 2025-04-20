from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents import Agent # Keep Agent if other tools might be used elsewhere or directly
import pandas as pd
from google.cloud import storage, bigquery
from io import StringIO
import pyarrow # Keep if push_to_bigquery or transform_and_load_bq are used by other potential agents

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash" # Define the model centrally
RAW_DATASET_ID = "raw" # Keep if needed for other tools
DW_DATASET_ID = "dw"   # Keep if needed for other tools

# --- Tool Functions (Keep them defined at the module level) ---

def read_csv_from_gcs(bucket_name: str, prefix: str = "") -> dict:
    """
    Reads all CSV files from a specified GCS bucket and optional prefix.
    Returns a dictionary where keys are filenames and values are file data as dictionaries.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(client.list_blobs(bucket_name, prefix=prefix)) # Get blobs matching prefix

        all_data = {}
        csv_found = False
        for blob in blobs:
            # Ensure we only process files directly under the prefix (if specified)
            # and that they are CSV files. Handle potential subdirectories if needed.
            if blob.name.endswith(".csv") and (not prefix or blob.name.startswith(prefix)):
                 # Avoid double-checking prefix if already filtered by list_blobs
                 # Check if it's directly under the prefix or a subdirectory file
                 relative_path = blob.name[len(prefix):] if prefix else blob.name
                 if '/' not in relative_path.strip('/'): # Only process files, not in subdirs
                    print(f"Reading file: {blob.name}") # Added for logging
                    content = blob.download_as_text()
                    df = pd.read_csv(StringIO(content))
                    all_data[blob.name] = df.to_dict(orient='list') # Store data as dict
                    csv_found = True

        if not csv_found:
            # Distinguish between no blobs found and no *CSV* blobs found
            if not blobs:
                 return {"status": "warning", "message": f"No objects found in gs://{bucket_name}/{prefix}"}
            else:
                 return {"status": "warning", "message": f"No CSV files found directly under gs://{bucket_name}/{prefix}"}


        return {"status": "success", "data": all_data} # Return dict {filename: data_dict}
    except Exception as e:
        return {"status": "error", "message": f"Error reading from GCS: {str(e)}"}

def perform_data_quality_checks(data: dict, filename: str) -> dict:
    """
    Performs basic data quality checks on a DataFrame represented as a dictionary.
    Includes the filename in the response for clarity.
    """
    try:
        # Convert dictionary back to DataFrame for checks
        # Ensure orient='list' was used when creating the dict
        df = pd.DataFrame(data)

        results = []
        # Check for null values
        if df.isnull().values.any():
            null_cols = df.isnull().sum()
            null_cols = null_cols[null_cols > 0]
            results.append(f"Null values found in columns: {null_cols.to_dict()}")

        # Check for duplicate rows
        if df.duplicated().any():
            num_duplicates = df.duplicated().sum()
            results.append(f"{num_duplicates} duplicate rows found.")

        if not results:
            return {"status": "success", "filename": filename, "message": "Data quality checks passed."}
        else:
            # Return warning instead of error if checks fail but process didn't crash
            return {"status": "warning", "filename": filename, "message": "Data quality issues found: " + " | ".join(results)}

    except Exception as e:
        return {"status": "error", "filename": filename, "message": f"Error during data quality check: {str(e)}"}

# --- Other Tool Functions (Keep if needed for other potential agents/workflows) ---

def push_to_bigquery(dataset_id: str, table_id: str, data: dict, filename: str) -> dict:
    """
    Pushes data for a single file to a specified BigQuery table.
    Creates or replaces the table.
    Args:
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID (will be created/replaced).
        data: The data for the file as a dictionary (orient='list').
        filename: The original filename (for logging/reference).
    Returns:
        A dictionary indicating the status and message.
    """
    try:
        # Ensure table_id is valid for BigQuery (alphanumeric + underscores)
        # Basic sanitization - replace common invalid chars. More robust needed for production.
        sanitized_table_id = ''.join(c if c.isalnum() else '_' for c in table_id)
        if not sanitized_table_id:
             return {"status": "error", "filename": filename, "message": f"Could not generate a valid table ID from '{table_id}'"}


        client = bigquery.Client() # Assumes project is configured via ADC or env var
        project_id = client.project
        full_table_id = f"{project_id}.{dataset_id}.{sanitized_table_id}"
        table_ref = client.dataset(dataset_id).table(sanitized_table_id)

        print(f"Attempting to load data for '{filename}' into table: {full_table_id}") # Logging

        # Convert dictionary (orient='list') back to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            return {"status": "warning", "filename": filename, "table_id": full_table_id, "message": "Input data is empty, skipping load."}

        # Configure the load job
        job_config = bigquery.LoadJobConfig(
            autodetect=True,  # Automatically detect schema
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE # Replace table if exists
            # Use WRITE_APPEND if you want to append to existing tables
            # Use WRITE_EMPTY to only write if the table is new/empty
        )

        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for the job to complete

        return {
            "status": "success",
            "filename": filename,
            "table_id": full_table_id,
            "message": f"Data successfully loaded to {full_table_id}"
        }
    except Exception as e:
        # Attempt to provide more context in the error
        error_message = f"Error pushing data for '{filename}' to BigQuery table '{sanitized_table_id}': {str(e)}"
        print(f"ERROR: {error_message}") # Log the error server-side
        return {
            "status": "error",
            "filename": filename,
            "table_id": f"{dataset_id}.{sanitized_table_id}", # Report intended table
            "message": error_message
        }

# --- Sub-Agent Definitions ---

# 1. Agent to read CSVs from GCS
gcs_reader_agent = LlmAgent(
    name="GcsCsvReaderAgent",
    model=GEMINI_MODEL,
    instruction="""You are an agent that reads CSV files from Google Cloud Storage.
    1. Use the 'read_csv_from_gcs' tool.
    2. You will be given a 'bucket_name' and an optional 'prefix'.
    3. Call the tool with these parameters.
    4. The tool will return a dictionary where keys are filenames and values are the data from each CSV.
    5. Output the result from the tool directly.
    """,
    description="Reads multiple CSV files from a GCS bucket/prefix.",
    tools=[read_csv_from_gcs],
    output_key="gcs_read_output" # Store the dict {filename: data} here
)

# 2. Agent to validate the data read from GCS
data_validator_agent = LlmAgent(
    name="DataValidatorAgent",
    model=GEMINI_MODEL,
    instruction="""You are an agent that validates data quality for multiple files.
    1. Access the data stored in the session state under the key 'gcs_read_output'. This will be the result from the GCS reader agent. It contains a 'status' and potentially a 'data' dictionary like {'filename1': {col1: [...], col2: [...]}, 'filename2': {...}}.
    2. Check if the 'status' of 'gcs_read_output' is 'success' and if the 'data' dictionary exists and is not empty. If not, report the status and message from 'gcs_read_output' and stop.
    3. If successful, iterate through each key-value pair in the 'data' dictionary (where the key is the 'filename' and the value is the 'data' dictionary for that file).
    4. For each file, use the 'perform_data_quality_checks' tool, passing the 'data' dictionary and the 'filename' string as arguments.
    5. Collect the results (status and message) from the tool for *each* file.
    6. Format the final output as a summary dictionary containing the validation status for all processed files. For example: {'validation_summary': {'file1.csv': {'status': 'success', 'message': '...'}, 'file2.csv': {'status': 'warning', 'message': '...'}}}
    """,
    description="Validates data quality for each CSV file read from GCS.",
    tools=[perform_data_quality_checks],
    output_key="validation_results" # Store the final validation summary
)

# --- Sub-Agent Definitions ---

# ... (gcs_reader_agent and data_validator_agent remain the same) ...

# 3. Agent to load validated data into BigQuery tables
bq_loader_agent = LlmAgent(
    name="BigQueryLoaderAgent",
    model=GEMINI_MODEL,
    instruction=f"""You are an agent that loads validated data into separate BigQuery tables.
    1. Access the original data read from GCS, stored in the session state under the key 'gcs_read_output'. This contains ['data'][filename] -> dict_data.
    2. Access the validation results stored in the session state under the key 'validation_results'. This contains ['validation_summary'][filename] -> {{'status': '...', 'message': '...'}}.
    3. Check if 'validation_results' and 'validation_results['validation_summary']' exist. If not, report an error and stop.
    4. Iterate through each 'filename' in the 'validation_results['validation_summary']' dictionary.
    5. If the 'status' for a given 'filename' is 'success':
        a. Retrieve the corresponding data dictionary for that file from 'gcs_read_output['data'][filename]'.
        b. Derive a suitable BigQuery 'table_id' from the 'filename'. Remove any directory prefix and the '.csv' extension. Replace any remaining non-alphanumeric characters with underscores. For example, 'data/raw/my_sales_data-v1.csv' should become 'my_sales_data_v1'. Ensure the result is not empty.
        c. Call the 'push_to_bigquery' tool with:
            - dataset_id: '{RAW_DATASET_ID}' # Target dataset for raw data
            -dataset_id: '{DW_DATASET_ID}' # Target dataset for lookpup data
            - table_id: the derived table_id from step 5b.
            - data: the data dictionary retrieved in step 5a.
            - filename: the original filename (key from the loop).
    6. Collect the results (status, message, table_id) from the tool for *each* file processed.
    7. Format the final output as a summary dictionary containing the loading status for all attempted files. For example: {{'load_summary': {{'file1.csv': {{'status': 'success', 'table_id': '...', 'message': '...'}}, 'file2.csv': {{'status': 'error', 'message': '...'}} }}}}
    """,
    description="Loads successfully validated data files into individual BigQuery tables.",
    tools=[push_to_bigquery],
    output_key="bq_load_results" # Store the final loading summary
)


# --- Root Agent (Sequential Orchestrator) ---
# This agent runs the reader, then the validator, then the loader.
root_agent = SequentialAgent(
    name="GcsIngestionValidationLoadPipeline",
    sub_agents=[
        gcs_reader_agent,
        data_validator_agent,
        bq_loader_agent # Added the new agent here
    ],
    description="Pipeline to read CSVs from GCS, perform data quality validation, and load valid files into BigQuery."
    # The final output of the SequentialAgent will be the output of the *last*
    # sub-agent in the sequence (i.e., 'bq_load_results').
)