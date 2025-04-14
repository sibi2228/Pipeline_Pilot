
def transform_and_load_bq(source_table: str, target_table: str, transform_instructions: str) -> dict:
    """
    Reads a BigQuery table, performs transformations based on instructions,
    and loads the result into a new BigQuery table in the DW dataset.
    """
    try:
        client = bigquery.Client(project="your-project-id")  # Explicitly set project ID
        source_full_table_id = f"your-project-id.{RAW_DATASET_ID}.{source_table}"
        target_full_table_id = f"your-project-id.{DW_DATASET_ID}.{target_table}"

        # Read data from source table
        query = f"SELECT * FROM `{source_full_table_id}`"
        df = client.query(query).to_dataframe()

        if df.empty:
            return {"status": "warning", "message": f"Source table {source_full_table_id} is empty."}

        # Apply transformations (example logic)
        transformed_df = df.copy()
        instructions_lower = transform_instructions.lower()

        if "filter" in instructions_lower:
            filter_condition = transform_instructions.split("filter by ")[-1].strip()
            transformed_df = transformed_df.query(filter_condition)

        if "create column" in instructions_lower:
            parts = transform_instructions.split("create column ")[-1].strip().split(" as ")
            if len(parts) == 2:
                new_column_name = parts[0].strip()
                expression = parts[1].strip()
                transformed_df[new_column_name] = transformed_df.eval(expression)

        # Load transformed data into target table
        table_ref = client.dataset(DW_DATASET_ID).table(target_table)
        job_config = bigquery.LoadJobConfig()
        
        job = client.load_table_from_dataframe(transformed_df, table_ref, job_config=job_config)
        job.result()  # Wait for job completion

        return {"status": "success", "message": f"Transformed data loaded to: {target_full_table_id}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}