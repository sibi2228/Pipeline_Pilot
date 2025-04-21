# c:\Users\Rahul\Desktop\Hackathon\multi_tool_agent\agent.py

from google.adk.agents.llm_agent import LlmAgent
import pandas as pd
from google.cloud import storage, bigquery
from io import StringIO
import pyarrow # Keep if BQ client uses it implicitly
import re # Import regex for table ID sanitization
import os # Needed for basename extraction

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash"
RAW_DATASET_ID = "raw" # Default target dataset for loaded raw data
# DW_DATASET_ID = "dw" # Keep if needed for other potential future tools

# --- Consolidated Tool Function ---

def process_gcs_to_bq(bucket_name: str, target_dataset_id: str, prefix: str = "") -> dict:
    """
    Reads all CSV files from a GCS path, validates (nulls, duplicates),
    cleans (drops null rows, drops duplicate rows), derives BigQuery table names,
    and loads the data into the specified BigQuery dataset.

    Args:
        bucket_name: The GCS bucket name.
        target_dataset_id: The BigQuery dataset ID to load data into.
        prefix: Optional GCS prefix (folder path) to filter files.

    Returns:
        A dictionary containing:
        - 'overall_status': 'success', 'warning', or 'error'.
        - 'overall_message': A summary message of the entire process.
        - 'file_details': A dictionary where keys are filenames and values are
                          dictionaries detailing the processing steps and status
                          for each file (read, validate, clean, load).
    """
    all_file_details = {}
    processed_files_count = 0
    successful_loads = 0
    skipped_files = 0
    files_with_errors = 0
    overall_status = "success" # Assume success initially
    error_messages = []
    warning_messages = []

    try:
        gcs_client = storage.Client()
        bq_client = bigquery.Client() # Initialize BQ client once
        project_id = bq_client.project # Get project ID for table refs

        bucket = gcs_client.bucket(bucket_name)
        blobs = list(gcs_client.list_blobs(bucket_name, prefix=prefix))

        if not blobs:
            return {
                "overall_status": "warning",
                "overall_message": f"No objects found in gs://{bucket_name}/{prefix}",
                "file_details": {}
            }

        csv_files_found = False
        for blob in blobs:
            filename = blob.name
            if not filename.endswith(".csv"):
                continue # Skip non-csv files

            if blob.size == 0 and filename.endswith('/'):
                print(f"Skipping likely directory placeholder: {filename}")
                continue

            csv_files_found = True
            processed_files_count += 1
            file_status = {
                "filename": filename,
                "read_status": "pending",
                "validation_status": "pending",
                "cleaning_status": "pending",
                "load_status": "pending",
                "target_table": None,
                "message": ""
            }
            print(f"--- Processing file: {filename} ---")

            df = None # Initialize df to None for this file's scope

            # --- 1. Read from GCS ---
            try:
                content = blob.download_as_text()
                file_status["read_status"] = "success"
                print(f"  Read success.")
            except Exception as read_err:
                error_msg = f"Error reading file from GCS: {str(read_err)}"
                print(f"  ERROR: {error_msg}")
                file_status["read_status"] = "error"
                file_status["message"] = error_msg
                files_with_errors += 1
                overall_status = "error"
                error_messages.append(f"{filename}: {error_msg}")
                all_file_details[filename] = file_status
                continue # Skip to next file

            # --- 2. Parse CSV & Initial Validation ---
            try:
                # Use StringIO for pandas
                string_io_content = StringIO(content)
                # Attempt parsing, warn on bad lines but continue if possible
                df = pd.read_csv(string_io_content, on_bad_lines='warn')

                if df.empty:
                    warn_msg = "File resulted in an empty DataFrame after read (check for header-only or parsing issues)."
                    print(f"  WARNING: {warn_msg}")
                    file_status["read_status"] = "skipped_empty"
                    file_status["validation_status"] = "skipped"
                    file_status["cleaning_status"] = "skipped"
                    file_status["load_status"] = "skipped"
                    file_status["message"] = warn_msg
                    skipped_files += 1
                    overall_status = "warning" if overall_status == "success" else overall_status
                    warning_messages.append(f"{filename}: {warn_msg}")
                    all_file_details[filename] = file_status
                    continue # Skip to next file

                # Basic structural validation (check if DataFrame creation worked implicitly)
                file_status["validation_status"] = "initial_success" # Placeholder
                print(f"  Parse success. Initial rows: {len(df)}")

            except pd.errors.EmptyDataError:
                warn_msg = "Skipped processing for empty file (no data)."
                print(f"  WARNING: {warn_msg}")
                file_status["read_status"] = "skipped_empty"
                file_status["validation_status"] = "skipped"
                file_status["cleaning_status"] = "skipped"
                file_status["load_status"] = "skipped"
                file_status["message"] = warn_msg
                skipped_files += 1
                overall_status = "warning" if overall_status == "success" else overall_status
                warning_messages.append(f"{filename}: {warn_msg}")
                all_file_details[filename] = file_status
                continue # Skip to next file

            except Exception as pd_err:
                # Catch other pandas parsing errors (includes potential structure issues)
                error_msg = f"Error parsing CSV: {pd_err}. Check file structure/encoding."
                print(f"  ERROR: {error_msg}")
                file_status["read_status"] = "error" # Mark read as error if parsing fails
                file_status["validation_status"] = "error"
                file_status["message"] = error_msg
                files_with_errors += 1
                overall_status = "error"
                error_messages.append(f"{filename}: {error_msg}")
                all_file_details[filename] = file_status
                continue # Skip to next file

            # --- 3. Data Quality Validation (Nulls, Duplicates) ---
            dq_issues = []
            has_nulls = df.isnull().values.any()
            has_duplicates = df.duplicated().any()

            if has_nulls:
                null_counts = df.isnull().sum()
                null_cols = null_counts[null_counts > 0]
                dq_issues.append(f"Nulls found: {null_cols.to_dict()}")
            if has_duplicates:
                num_duplicates = df.duplicated().sum()
                dq_issues.append(f"{num_duplicates} duplicates found.")

            if not dq_issues:
                file_status["validation_status"] = "success"
                file_status["cleaning_status"] = "not_needed"
                file_status["message"] = "Validation passed. No cleaning needed."
                print(f"  Validation success. No cleaning needed.")
            else:
                file_status["validation_status"] = "warning"
                file_status["message"] = "Validation warnings: " + " | ".join(dq_issues)
                print(f"  Validation warning: {' | '.join(dq_issues)}")
                # Proceed to cleaning

                # --- 4. Cleaning (if validation had warnings) ---
                try:
                    cleaning_actions = []
                    rows_before_clean = len(df)

                    if has_nulls:
                        df.dropna(inplace=True)
                        rows_after_na = len(df)
                        if rows_before_clean > rows_after_na:
                            cleaning_actions.append(f"Dropped {rows_before_clean - rows_after_na} rows with nulls.")
                            rows_before_clean = rows_after_na # Update count for duplicate check

                    if has_duplicates:
                        # Drop duplicates only after handling nulls
                        df.drop_duplicates(inplace=True)
                        rows_after_dup = len(df)
                        if rows_before_clean > rows_after_dup:
                             cleaning_actions.append(f"Dropped {rows_before_clean - rows_after_dup} duplicate rows.")

                    if df.empty and rows_before_clean > 0: # Check if cleaning emptied the DataFrame
                         warn_msg = f"Cleaning resulted in an empty DataFrame (original rows: {rows_before_clean})."
                         print(f"  WARNING: {warn_msg}")
                         file_status["cleaning_status"] = "success_emptied"
                         file_status["load_status"] = "skipped_empty_after_clean"
                         file_status["message"] += " | " + warn_msg
                         skipped_files += 1
                         overall_status = "warning" if overall_status == "success" else overall_status
                         warning_messages.append(f"{filename}: {warn_msg}")
                         all_file_details[filename] = file_status
                         continue # Skip loading this file

                    clean_msg = "Cleaning applied: " + " | ".join(cleaning_actions) + f" Final rows: {len(df)}."
                    file_status["cleaning_status"] = "success"
                    file_status["message"] += " | " + clean_msg # Append cleaning info
                    print(f"  {clean_msg}")

                except Exception as clean_err:
                    error_msg = f"Error during data cleaning: {str(clean_err)}"
                    print(f"  ERROR: {error_msg}")
                    file_status["cleaning_status"] = "error"
                    file_status["load_status"] = "skipped_cleaning_error"
                    file_status["message"] += " | " + error_msg
                    files_with_errors += 1
                    overall_status = "error"
                    error_messages.append(f"{filename}: {error_msg}")
                    all_file_details[filename] = file_status
                    continue # Skip loading

            # --- 5. Derive Table ID ---
            try:
                base_name = os.path.basename(filename)
                table_id_base = os.path.splitext(base_name)[0]
                # Replace non-alphanumeric with underscore, ensure starts with letter/underscore
                table_id = re.sub(r'[^a-zA-Z0-9_]', '_', table_id_base)
                if not re.match(r"^[a-zA-Z_]", table_id):
                    table_id = "_" + table_id # Prepend underscore if starts with number
                if not table_id: # Handle cases where filename becomes empty after sanitization
                    raise ValueError("Derived table ID is empty after sanitization.")
                # Final check for BQ validity (redundant with regex but safe)
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_id):
                     raise ValueError(f"Derived table ID '{table_id}' is invalid for BigQuery.")

                full_table_id = f"{project_id}.{target_dataset_id}.{table_id}"
                file_status["target_table"] = full_table_id
                print(f"  Derived Target Table: {full_table_id}")

            except Exception as derive_err:
                error_msg = f"Error deriving BigQuery table ID: {str(derive_err)}"
                print(f"  ERROR: {error_msg}")
                file_status["load_status"] = "skipped_table_id_error"
                file_status["message"] += " | " + error_msg
                files_with_errors += 1
                overall_status = "error"
                error_messages.append(f"{filename}: {error_msg}")
                all_file_details[filename] = file_status
                continue # Skip loading

            # --- 6. Load to BigQuery ---
            try:
                if df.empty: # Double check df isn't empty before load attempt
                    warn_msg = "Skipping load because DataFrame is empty (check original file or cleaning results)."
                    print(f"  WARNING: {warn_msg}")
                    file_status["load_status"] = "skipped_empty"
                    file_status["message"] += " | " + warn_msg
                    skipped_files += 1
                    overall_status = "warning" if overall_status == "success" else overall_status
                    warning_messages.append(f"{filename}: {warn_msg}")
                    all_file_details[filename] = file_status
                    continue # Skip loading this file

                table_ref = bq_client.dataset(target_dataset_id).table(table_id)
                job_config = bigquery.LoadJobConfig(
                    autodetect=True,
                    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
                )
                job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
                job.result()  # Wait for completion

                load_msg = f"Successfully loaded to {full_table_id}"
                file_status["load_status"] = "success"
                file_status["message"] += " | " + load_msg
                print(f"  {load_msg}")
                successful_loads += 1

            except Exception as bq_err:
                error_msg = f"Error loading data to BigQuery table {full_table_id}: {str(bq_err)}"
                print(f"  ERROR: {error_msg}")
                file_status["load_status"] = "error"
                file_status["message"] += " | " + error_msg
                files_with_errors += 1
                overall_status = "error"
                error_messages.append(f"{filename}: {error_msg}")
                # Store details even if load failed
                all_file_details[filename] = file_status
                continue # Continue to next file

            # Store final details for successfully processed file (up to load attempt)
            all_file_details[filename] = file_status


        if not csv_files_found:
             return {
                 "overall_status": "warning",
                 "overall_message": f"No CSV files found matching gs://{bucket_name}/{prefix}*.csv",
                 "file_details": {}
             }

        # --- Compile Final Summary ---
        summary_parts = [f"Processed {processed_files_count} CSV file(s)."]
        if successful_loads > 0:
            summary_parts.append(f"{successful_loads} loaded successfully.")
        if skipped_files > 0:
            summary_parts.append(f"{skipped_files} skipped (empty, cleaned empty, or parse issues).")
        if files_with_errors > 0:
            summary_parts.append(f"{files_with_errors} encountered errors (read, clean, derive ID, or load).")
            overall_status = "error" # Ensure overall status is error if any file had errors

        if overall_status == "error":
             summary_parts.append("Errors occurred during the process. See file details and error list.")
             summary_parts.extend(error_messages) # Add specific errors to message
        elif overall_status == "warning":
             summary_parts.append("Warnings or skips occurred. See file details and warning list.")
             summary_parts.extend(warning_messages) # Add specific warnings

        final_message = " | ".join(summary_parts)

        return {
            "overall_status": overall_status,
            "overall_message": final_message,
            "file_details": all_file_details
        }

    except Exception as e:
        # Catch broad errors during client initialization or listing blobs etc.
        print(f"CRITICAL ERROR: Unhandled exception during GCS to BQ processing: {str(e)}")
        return {
            "overall_status": "error",
            "overall_message": f"Critical failure during pipeline execution: {str(e)}",
            "file_details": all_file_details # Return any partial results
        }


# --- Root Agent (Single LLM Agent Orchestrator) ---
root_agent = LlmAgent(
    name="Ingestion_Agent",
    model=GEMINI_MODEL,
    instruction=f"""
    You are an agent designed to execute a data pipeline that processes CSV files from Google Cloud Storage (GCS) and loads them into BigQuery.

    Your task is to:
    1. Receive the `bucket_name` and optionally a `prefix` for the GCS location from the user or context.
    2. Determine the `target_dataset_id` for BigQuery. Use '{RAW_DATASET_ID}' unless specifically instructed otherwise.
    3. Call the `process_gcs_to_bq` tool with the `bucket_name`, `target_dataset_id`, and `prefix`. This single tool handles reading, validating, cleaning, and loading all relevant CSV files.
    4. Receive the results dictionary from the `process_gcs_to_bq` tool.
    5. Present a final report to the user based *only* on the information returned by the tool. The report should include:
        - The `overall_status` ('success', 'warning', 'error').
        - The `overall_message` summarizing the process.
        - A summary of the processing details for each file found in the `file_details` part of the result. For each file, mention its final outcome (e.g., loaded successfully to [table_name], skipped due to [reason], error during [step]).

    Do not try to perform individual steps like reading, cleaning, or loading yourself. Rely solely on the `process_gcs_to_bq` tool to perform the entire operation.
    """,
    description="Orchestrates a GCS CSV to BigQuery pipeline using a single comprehensive tool.",
    tools=[
        process_gcs_to_bq, # The only tool needed now
    ],
    output_key="pipeline_summary" # Store the final formatted report here
)
