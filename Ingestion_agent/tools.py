import pandas as pd
from google.cloud import storage, bigquery

def read_csv_from_gcs(bucket_name: str, file_name: str) -> dict:
    """Reads a CSV file from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    content = blob.download_as_text()
    
    df = pd.read_csv(pd.compat.StringIO(content))
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
    
    job = client.load_table_from_dataframe(pd.DataFrame(data), table_ref)
    job.result()  # Wait for job completion
    
    return {"status": "success", "message": f"Data pushed to {dataset_id}.{table_id}"}
