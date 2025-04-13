from google.adk.agents import Agent
import pandas as pd
from google.cloud import storage, bigquery
from io import StringIO
import pyarrow

def read_csv_from_gcs(bucket_name: str, file_name: str) -> dict:
    """Reads a CSV file from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    content = blob.download_as_text()
    
    df = pd.read_csv(StringIO(content))
    return {"status": "success", "data": df.to_dict()}

def perform_data_quality_checks(data: dict) -> dict:
    """Performs basic data quality checks."""
    df = pd.DataFrame(data)
    
    # Example checks: Null values and duplicate rows
    if df.isnull().values.any():
        return {"status": "error", "message": "Data contains null values."}
    if df.duplicated().any():
        return {"status": "error", "message": "Data contains duplicate rows."}
    
    return {"status": "success", "data": df.to_dict()}

def push_to_bigquery(dataset_id: str, table_id: str, data: dict) -> dict:
    """Pushes validated data to BigQuery."""
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    
    df = pd.DataFrame(data)
# Use pyarrow for efficient data transfer
    job_config = bigquery.LoadJobConfig(autodetect=True)
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    
    print(f"Data successfully loaded to {dataset_id}.{table_id}")
    
    return {"status": "success", "message": f"Data pushed to {dataset_id}.{table_id}"}


root_agent = Agent(
    name="data_ingestion_agent",
    model="gemini-2.0-flash",
    description="Agent for ingesting and validating CSV data from GCS.",
    instruction="Perform data ingestion, validation, and push to BigQuery.",
    tools=[read_csv_from_gcs, perform_data_quality_checks, push_to_bigquery]
)