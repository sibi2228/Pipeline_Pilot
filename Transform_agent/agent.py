"""
Agent to read 'raw' tables, merge to temp1, join lookups to temp2,
then load temp2 to the final 'dw' table.
"""

import traceback
from typing import Dict, List, Any, Optional
import datetime
import uuid # For temporary table names

import numpy as np
import pandas as pd
from google.adk.agents.llm_agent import LlmAgent
from google.cloud import bigquery
from google.cloud.exceptions import NotFound # To handle temp table deletion errors

# --- Constants ---
GEMINI_MODEL: str = "gemini-2.0-flash" # Or your preferred model
RAW_DATASET_ID: str = "raw"
DW_DATASET_ID: str = "dw"
DEFAULT_BQ_LOCATION: str = "US"
COUNTRY_LOOKUP_TABLE: str = "country_lookup" # Standardized name
PRODUCT_LOOKUP_TABLE: str = "product_catalouge" # Standardized name

# --- Tool Function 1: Read, Merge, and Write to First Temporary Table ---
def read_merge_write_temp(project_id: str, raw_dataset_id: str = RAW_DATASET_ID):
    """
    Reads all tables from the raw dataset, merges them, prints head,
    and writes the merged DataFrame to a new temporary table (temp1).
    Returns the ID of the temporary table created (temp1_id).
    """
    print(f"\n--- Starting: read_merge_write_temp (Dataset: {raw_dataset_id}) ---")
    temp_table_id_1 = f"temp_merged_data_{uuid.uuid4().hex[:12]}"
    full_temp_table_ref_str = f"{project_id}.{raw_dataset_id}.{temp_table_id_1}"
    print(f"Target temporary table (1): {full_temp_table_ref_str}")

    try:
        client = bigquery.Client(project=project_id)
        dataset_ref = client.dataset(raw_dataset_id)
        tables = list(client.list_tables(dataset_ref))
        if not tables:
            return {"status": "error", "message": f"No tables found in BigQuery dataset: {raw_dataset_id} to merge."}

        all_data_frames = []
        for table in tables:
            # Avoid reading previous temp tables if they exist
            if table.table_id.startswith("temp_"):
                print(f"  Skipping potential temporary table: {table.table_id}")
                continue
            table_ref = dataset_ref.table(table.table_id)
            full_table_id = f"`{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`"
            query = f"SELECT * FROM {full_table_id}"
            try:
                df = client.query(query).to_dataframe()
                if not df.empty:
                    all_data_frames.append(df)
                    print(f"  Read {len(df):,} rows from table: {table.table_id}")
                else:
                    print(f"  Table {table.table_id} is empty, skipping.")
            except Exception as table_e:
                print(f"  Warning: Failed to read or process table {table.table_id}: {str(table_e)}")

        if not all_data_frames:
            return {"status": "error", "message": "No data read from any source table in the dataset."}

        merged_df = pd.concat(all_data_frames, ignore_index=True)
        print(f"Successfully merged data. Total rows: {len(merged_df):,}")

        if not merged_df.empty:
            print("\n--- First 10 rows of merged raw data (before writing to temp 1) ---")
            try: print(merged_df.head(10).to_string())
            except Exception as print_e: print(f"  (Could not print DataFrame head: {print_e})")
            print("--- End of first 10 rows ---\n")
        else:
            return {"status": "error", "message": "Merged data resulted in an empty DataFrame."}

        temp_table_ref = client.dataset(raw_dataset_id).table(temp_table_id_1)
        job_config = bigquery.LoadJobConfig(autodetect=True, write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        print(f"Attempting to write {len(merged_df):,} rows to temporary table {full_temp_table_ref_str}...")
        load_job = client.load_table_from_dataframe(merged_df, temp_table_ref, job_config=job_config)
        load_job.result()
        print(f"  Temp table (1) load job {load_job.job_id} completed.")

        load_job = client.get_job(load_job.job_id, location=DEFAULT_BQ_LOCATION)
        if load_job.errors: raise Exception(f"Failed to load temporary table {temp_table_id_1}: {load_job.errors}")

        print(f"--- Finished: read_merge_write_temp ---")
        return {"status": "success", "temp_table_id_1": temp_table_id_1} # Return ID of first temp table

    except Exception as e:
        print(f"ERROR in read_merge_write_temp: {traceback.format_exc()}")
        return {"status": "error", "message": f"Failed during read/merge/write to temp table 1: {str(e)}"}


# --- Tool Function 2: Join Lookups and Write to Second Temporary Table ---
def join_lookups_write_temp(
    project_id: str,
    source_temp_table_id: str, # ID of temp table 1 from previous step
    raw_dataset_id: str = RAW_DATASET_ID,
    dw_dataset_id: str = DW_DATASET_ID,
    country_lookup_table: str = COUNTRY_LOOKUP_TABLE,
    product_lookup_table: str = PRODUCT_LOOKUP_TABLE,
    location: str = DEFAULT_BQ_LOCATION,
    cleanup_source_temp: bool = True # Option to delete temp table 1
):
    """
    Reads data from the first temporary table, joins it with country and product lookups,
    and writes the result to a second temporary table (temp2).
    Returns the ID of the second temporary table (temp2_id).
    Optionally cleans up the first temporary table.
    """
    print(f"\n--- Starting: join_lookups_write_temp ---")
    print(f"  Source Temp Table (1): {project_id}.{raw_dataset_id}.{source_temp_table_id}")
    temp_table_id_2 = f"temp_lookup_data_{uuid.uuid4().hex[:12]}" # ID for the second temp table
    full_temp_table_2_ref_str = f"{project_id}.{raw_dataset_id}.{temp_table_id_2}"
    print(f"  Target Temp Table (2): {full_temp_table_2_ref_str}")

    client = bigquery.Client(project=project_id, location=location)
    source_temp_table_ref = client.dataset(raw_dataset_id).table(source_temp_table_id)
    temp_table_2_ref = client.dataset(raw_dataset_id).table(temp_table_id_2)

    try:
        # --- Read data from Source Temporary Table (temp1) ---
        print(f"Reading data from source temporary table {source_temp_table_id}...")
        query_temp1 = f"SELECT * FROM `{project_id}.{raw_dataset_id}.{source_temp_table_id}`"
        df_source = client.query(query_temp1).to_dataframe()
        if df_source.empty:
            return {"status": "warning", "message": f"Source temporary table {source_temp_table_id} was empty. Cannot perform joins."}
        print(f"Read {len(df_source):,} rows from source temporary table.")

        # --- Read Lookup Tables ---
        df_country = pd.DataFrame()
        try:
            print(f"Reading country lookup: {dw_dataset_id}.{country_lookup_table}")
            query_country = f"SELECT * FROM `{project_id}.{dw_dataset_id}.{country_lookup_table}`"
            df_country = client.query(query_country).to_dataframe()
            if df_country.empty: print("  Warning: Country lookup table is empty.")
            else: print(f"  Read {len(df_country)} rows from country lookup.")
        except Exception as lookup_e:
            print(f"  Warning: Failed to read country lookup table: {lookup_e}")

        df_product = pd.DataFrame()
        try:
            print(f"Reading product lookup: {dw_dataset_id}.{product_lookup_table}")
            query_product = f"SELECT * FROM `{project_id}.{dw_dataset_id}.{product_lookup_table}`"
            df_product = client.query(query_product).to_dataframe()
            if df_product.empty: print("  Warning: Product lookup table is empty.")
            else: print(f"  Read {len(df_product)} rows from product lookup.")
        except Exception as lookup_e:
            print(f"  Warning: Failed to read product lookup table: {lookup_e}")

        # --- Perform Joins ---
        print("Performing joins...")
        joined_df = df_source.copy() # Start with data from temp table 1

        # --- Country Join Logic (CORRECTED) ---
        country_lookup_not_empty = not df_country.empty
        country_col_in_source = 'country' in joined_df.columns
        country_code_col_in_lookup = 'country_code' in df_country.columns # Check for the correct key
        print(f"  Country Join Check: LookupNotEmpty={country_lookup_not_empty}, ColInSource={country_col_in_source}, CodeColInLookup={country_code_col_in_lookup}")
        if not country_col_in_source:
            print(f"    Source Columns (Pre-Join): {list(joined_df.columns)}")
        if not country_code_col_in_lookup and country_lookup_not_empty:
             print(f"    Country Lookup Columns: {list(df_country.columns)}")

        if country_lookup_not_empty and country_col_in_source and country_code_col_in_lookup:
            print("  Joining with country data (on country/country_code)...")
            # Use left_on and right_on for different column names
            joined_df = pd.merge(
                joined_df,
                df_country,
                left_on='country',          # Column in the left DataFrame (df_source/joined_df)
                right_on='country_code',    # Column in the right DataFrame (df_country)
                how='left',
                suffixes=('', '_country_lookup') # Suffixes for any other overlapping columns (besides keys)
            )
            # Optional: Drop the redundant key column from the right table after join
            # if 'country_code' in joined_df.columns:
            #    joined_df = joined_df.drop(columns=['country_code'])
            print(f"    Shape after country join: {joined_df.shape}")
        elif country_lookup_not_empty:
            print("  Warning: Skipping country join - Check conditions above (need 'country' in source, 'country_code' in lookup).")
        # --- End Corrected Country Join Logic ---

        # --- Product Join Logic (Assuming 'product' is the key in both) ---
        product_lookup_not_empty = not df_product.empty
        product_col_in_source = 'product' in joined_df.columns
        product_col_in_lookup = 'product' in df_product.columns
        print(f"  Product Join Check: LookupNotEmpty={product_lookup_not_empty}, ColInSource={product_col_in_source}, ColInLookup={product_col_in_lookup}")
        if not product_col_in_source:
             print(f"    Source Columns (After Country Join Attempt): {list(joined_df.columns)}")
        if not product_col_in_lookup and product_lookup_not_empty:
             print(f"    Product Lookup Columns: {list(df_product.columns)}")

        if product_lookup_not_empty and product_col_in_source and product_col_in_lookup:
            print("  Joining with product data...")
            # Assuming 'product' is the correct key for this join
            joined_df = pd.merge(joined_df, df_product, on='product', how='left', suffixes=('', '_product_lookup'))
            print(f"    Shape after product join: {joined_df.shape}")
        elif product_lookup_not_empty:
            print("  Warning: Skipping product join - Check conditions above (need 'product' in source and lookup).")
        # --- End Product Join Logic ---

        print(f"Joins complete. Final resulting shape: {joined_df.shape}")

        # --- Write Joined DataFrame to Second Temporary Table (temp2) ---
        job_config = bigquery.LoadJobConfig(autodetect=True, write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        print(f"Attempting to write {len(joined_df):,} rows to temporary table {full_temp_table_2_ref_str}...")
        load_job = client.load_table_from_dataframe(joined_df, temp_table_2_ref, job_config=job_config)
        load_job.result()
        print(f"  Temp table (2) load job {load_job.job_id} completed.")

        load_job = client.get_job(load_job.job_id, location=DEFAULT_BQ_LOCATION)
        if load_job.errors: raise Exception(f"Failed to load temporary table {temp_table_id_2}: {load_job.errors}")

        print(f"--- Finished: join_lookups_write_temp ---")
        # Return the ID of the second temp table
        return {"status": "success", "temp_table_id_2": temp_table_id_2}

    except Exception as e:
        print(f"ERROR in join_lookups_write_temp: {traceback.format_exc()}")
        return {"status": "error", "message": f"Failed during join/write to temp table 2: {str(e)}"}

    finally:
        # --- Cleanup Source Temporary Table (temp1) ---
        if cleanup_source_temp and 'source_temp_table_ref' in locals(): # Check if ref was created
            print(f"Attempting to clean up source temporary table (1): {project_id}.{raw_dataset_id}.{source_temp_table_id}")
            try:
                client.delete_table(source_temp_table_ref, not_found_ok=True)
                print(f"  Successfully deleted source temporary table {source_temp_table_id}.")
            except Exception as delete_e:
                print(f"  Warning: Failed to delete source temporary table {source_temp_table_id}: {delete_e}")

# --- Tool Function 3: Load Data from Second Temporary Table to DW ---
def load_temp_table_to_dw(
    project_id: str,
    source_temp_table_id: str, # ID of temp table 2 from previous step
    destination_table_id: str, # Final destination table name
    raw_dataset_id: str = RAW_DATASET_ID, # Dataset where temp table 2 lives
    destination_dataset_id: str = DW_DATASET_ID,
    location: str = DEFAULT_BQ_LOCATION,
    write_mode: str = "WRITE_TRUNCATE", # Default write mode for final table
    cleanup_temp_table: bool = True # Option to delete temp table 2
):
    """
    Reads data from the specified (second) temporary table in the raw dataset and
    writes it to the final destination table in the DW dataset.
    Optionally cleans up the second temporary table.
    """
    print(f"\n--- Starting: load_temp_table_to_dw ---")
    print(f"  Source Temp Table (2): {project_id}.{raw_dataset_id}.{source_temp_table_id}")
    print(f"  Destination Table: {project_id}.{destination_dataset_id}.{destination_table_id}")

    client = bigquery.Client(project=project_id, location=location)
    full_source_temp_table_ref_str = f"{project_id}.{raw_dataset_id}.{source_temp_table_id}"
    source_temp_table_ref = client.dataset(raw_dataset_id).table(source_temp_table_id) # Ref to temp table 2
    final_table_ref = client.dataset(destination_dataset_id).table(destination_table_id)
    final_job_id = None # Initialize job ID

    try:
        # --- Read data from Second Temporary Table (temp2) ---
        print(f"Reading data from temporary table {full_source_temp_table_ref_str}...")
        query = f"SELECT * FROM `{full_source_temp_table_ref_str}`"
        df_from_temp = client.query(query).to_dataframe()

        if df_from_temp.empty:
            print("Warning: Temporary table (2) contained no data.")
            return {"status": "warning", "message": "Source temporary table (with lookups) was empty, nothing loaded to destination."}
        print(f"Read {len(df_from_temp):,} rows from temporary table (2).")

        # --- Write data to Final Destination Table ---
        valid_write_modes = ["WRITE_TRUNCATE", "WRITE_APPEND", "WRITE_EMPTY"]
        if write_mode not in valid_write_modes:
             raise ValueError(f"Invalid write_mode '{write_mode}'. Must be one of {valid_write_modes}")
        write_disposition = getattr(bigquery.WriteDisposition, write_mode)

        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            write_disposition=write_disposition,
            # Add partitioning/clustering here if desired for the final table
            # time_partitioning=bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.MONTH, field="order_date"), # Example
            # clustering_fields=["country", "product"], # Example
        )
        print(f"Configuring final load job to {destination_dataset_id}.{destination_table_id} with mode {write_mode}...")

        print(f"Attempting final BigQuery load for {len(df_from_temp):,} rows...")
        load_job = client.load_table_from_dataframe(df_from_temp, final_table_ref, job_config=job_config)
        final_job_id = load_job.job_id
        load_job.result()
        print(f"  Final load job {final_job_id} completed.")

        # --- Check Job Results ---
        load_job = client.get_job(final_job_id, location=location)
        if load_job.errors:
            return {"status": "error", "message": f"Final BigQuery load job {final_job_id} failed: {load_job.errors}", "job_id": final_job_id}

        output_rows = load_job.output_rows
        print(f"  Final BigQuery load successful. Rows loaded: {output_rows:,}")

        success_dict = {
            "status": "success",
            "message": f"Data successfully loaded from temp table to {destination_dataset_id}.{destination_table_id}. Rows loaded: {output_rows}",
            "output_rows": output_rows,
            "job_id": final_job_id
        }
        print(f"--- Load successful: load_temp_table_to_dw ---")
        return success_dict

    except Exception as e:
        print(f"ERROR in load_temp_table_to_dw: {traceback.format_exc()}")
        return {"status": "error", "message": f"Failed during load from temp 2 to DW: {str(e)}", "job_id": final_job_id}

    finally:
        # --- Cleanup Second Temporary Table (temp2) ---
        if cleanup_temp_table and source_temp_table_id:
            print(f"Attempting to clean up temporary table (2): {full_source_temp_table_ref_str}")
            try:
                client.delete_table(source_temp_table_ref, not_found_ok=True)
                print(f"  Successfully deleted temporary table {source_temp_table_id}.")
            except Exception as delete_e:
                print(f"  Warning: Failed to delete temporary table {source_temp_table_id}: {delete_e}")


# --- Agent Definition ---
root_agent = LlmAgent(
    name="LookupJoinTempTableAgent",
    model=GEMINI_MODEL,
    instruction=f"""
    You are an agent designed to merge data from Google Cloud BigQuery, join lookups using temporary tables.
    Your goal is:
    1. Read all tables from the '{RAW_DATASET_ID}' dataset.
    2. Merge them and write to a first temporary table ('temp1') in '{RAW_DATASET_ID}'.
    3. Read 'temp1', join it with '{COUNTRY_LOOKUP_TABLE}' and '{PRODUCT_LOOKUP_TABLE}' from '{DW_DATASET_ID}'.
    4. Write the joined result to a second temporary table ('temp2') in '{RAW_DATASET_ID}'.
    5. Load the data from 'temp2' into a final table in the '{DW_DATASET_ID}' dataset.
    6. Clean up temporary tables.

    Follow these steps precisely:

    1.  **Receive Inputs:** The user will provide the Google Cloud `project_id` and the final `destination_table_id` for the merged data in the '{DW_DATASET_ID}' dataset.

    2.  **Read, Merge, and Write to Temp Table 1:** Call the `read_merge_write_temp` tool.
        - Pass the user's `project_id`.
        - Pass `raw_dataset_id='{RAW_DATASET_ID}'`.
        - Store the entire dictionary result. Let's call it `temp_result_1`.
        - If `temp_result_1['status']` is 'error', stop immediately and return the 'message'.

    3.  **Prepare for Join Step:**
        - Extract the first temporary table ID: `temp_table_name_1 = temp_result_1['temp_table_id_1']`.

    4.  **Join Lookups and Write to Temp Table 2:** Call the `join_lookups_write_temp` tool.
        - Pass `project_id`: The user's project ID.
        - Pass `source_temp_table_id`: The `temp_table_name_1` extracted in step 3.
        - Pass `raw_dataset_id`: Use the constant `{RAW_DATASET_ID}`.
        - Pass `dw_dataset_id`: Use the constant `{DW_DATASET_ID}`.
        - Pass `country_lookup_table`: Use the constant '{COUNTRY_LOOKUP_TABLE}'.
        - Pass `product_lookup_table`: Use the constant '{PRODUCT_LOOKUP_TABLE}'.
        - Optional: Pass `cleanup_source_temp=True` (default) to delete the first temp table.
        - Store the result. Let's call it `temp_result_2`.
        - If `temp_result_2['status']` is 'error', stop immediately and return the 'message'.
        - If `temp_result_2['status']` is 'warning', note the message but proceed if possible (e.g., if lookups were empty but join still produced a table).

    5.  **Prepare for Final Load:**
        - Extract the second temporary table ID: `temp_table_name_2 = temp_result_2['temp_table_id_2']`.

    6.  **Load Temp Table 2 Data to Final Table:** Call the `load_temp_table_to_dw` tool.
        - Pass `project_id`: The user-provided project ID.
        - Pass `source_temp_table_id`: The `temp_table_name_2` extracted in step 5.
        - Pass `destination_table_id`: The user-provided final destination table ID.
        - Pass `raw_dataset_id`: Use the constant `{RAW_DATASET_ID}` (where temp table 2 lives).
        - Pass `destination_dataset_id`: Use the constant `{DW_DATASET_ID}`.
        - Optional: Specify `write_mode` (defaults to 'WRITE_TRUNCATE').
        - Optional: Pass `cleanup_temp_table=True` (default) to delete the second temp table.
        - Store the result. Let's call it `final_result`.

    7.  **Return Final Result:** Return the entire `final_result` dictionary received from the `load_temp_table_to_dw` tool call in step 6.
    """,
    description=(
        "Reads 'raw' tables, merges to temp1, joins lookups from 'dw' to temp2, "
        f"loads temp2 to a final table in '{DW_DATASET_ID}', cleaning up temp tables."
    ),
    tools=[
        read_merge_write_temp,
        join_lookups_write_temp, # Added new tool
        load_temp_table_to_dw,
    ],
    output_key="final_load_status",
)
