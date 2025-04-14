from google.adk.agents import Agent
import pandas as pd
from google.cloud import storage, bigquery
from io import StringIO
import pyarrow

# Define constants for project and dataset IDs
RAW_DATASET_ID = "raw"
DW_DATASET_ID = "dw"

def read_csv_from_gcs(bucket_name: str, file_name: str) -> dict:
    """Reads a CSV file from GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        content = blob.download_as_text()

        # Read CSV content into a pandas DataFrame
        df = pd.read_csv(StringIO(content))
        return {"status": "success", "data": df.to_dict()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def perform_data_quality_checks(data: dict) -> dict:
    """Performs basic data quality checks."""
    try:
        df = pd.DataFrame(data)

        # Check for null values and duplicates
        if df.isnull().values.any():
            return {"status": "error", "message": "Data contains null values."}
        if df.duplicated().any():
            return {"status": "error", "message": "Data contains duplicate rows."}

        return {"status": "success", "data": df.to_dict()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def push_to_bigquery(dataset_id: str, table_id: str, data: dict) -> dict:
    """Pushes validated data to BigQuery."""
    try:
        client = bigquery.Client()
        table_ref = client.dataset(dataset_id).table(table_id)

        # Convert dictionary to pandas DataFrame
        df = pd.DataFrame(data)

        # Use pyarrow for efficient data transfer
        job_config = bigquery.LoadJobConfig(autodetect=True)
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for job completion

        return {"status": "success", "message": f"Data successfully loaded to {dataset_id}.{table_id}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def transform_and_load_bq(source_table: str, target_table: str, transform_instructions: str) -> dict:
    """
    Reads a BigQuery table, performs transformations based on instructions,
    and loads the result into a new BigQuery table in the DW dataset.
    """
    try:
        client = bigquery.Client()
        source_full_table_id = f"{RAW_DATASET_ID}.{source_table}"
        target_full_table_id = f"{DW_DATASET_ID}.{target_table}"

        # 1. Read data from the source BigQuery table
        query = f"SELECT * FROM `{source_full_table_id}`"
        df = client.query(query).to_dataframe()

        if df.empty:
            return {"status": "warning", "message": f"Source table {source_full_table_id} is empty."}

        # 2. Apply transformations based on instructions (Basic example - needs LLM integration)
        transformed_df = df.copy()
        instructions_lower = transform_instructions.lower()

        if "filter" in instructions_lower:
            try:
                filter_condition = transform_instructions.split("filter by ")[-1].strip()
                transformed_df = transformed_df.query(filter_condition)
            except Exception as e:
                return {"status": "error", "message": f"Error applying filter: {e}"}

        if "create column" in instructions_lower:
            try:
                parts = transform_instructions.split("create column ")[-1].strip().split(" as ")
                if len(parts) == 2:
                    new_column_name = parts[0].strip()
                    expression = parts[1].strip()
                    transformed_df[new_column_name] = transformed_df.eval(expression)
                else:
                    return {"status": "error", "message": "Invalid 'create column' instruction format."}
            except Exception as e:
                return {"status": "error", "message": f"Error creating column: {e}"}

        # 3. Load the transformed data into a new BigQuery table in the DW dataset
        table_ref = client.dataset(DW_DATASET_ID).table(target_table)
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        
        job = client.load_table_from_dataframe(transformed_df, table_ref, job_config=job_config)
        job.result()  # Wait for job completion

        return {"status": "success", "message": f"Transformed data loaded to: {target_full_table_id}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

root_agent = Agent(
    name="data_ingestion_agent",
    model="gemini-2.0-flash",
    description="Agent for ingesting and validating CSV data from GCS.",
    instruction="Perform data ingestion, validation, transformation, and push to BigQuery.",
    tools=[read_csv_from_gcs, perform_data_quality_checks, push_to_bigquery, transform_and_load_bq]
)
