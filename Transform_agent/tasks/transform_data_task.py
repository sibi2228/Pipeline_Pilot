# tasks/transform_data_task.py
from google.adk import Task
from google.cloud import bigquery
from typing import List, Dict, Optional
import os

PROJECT_ID = os.environ.get("PROJECT_ID")
RAW_DATASET_ID = os.environ.get("RAW_DATASET_ID")
DW_DATASET_ID = os.environ.get("DW_DATASET_ID")

class TransformDataTask(Task):
    """A Task to transform data in BigQuery."""

    def __init__(self, source_table: str, target_table: str, transform_instructions: str):
        super().__init__()
        self.source_table = source_table
        self.target_table = target_table
        self.transform_instructions = transform_instructions
        self.client = bigquery.Client(project=PROJECT_ID)

    def generate_sql_query(self, source_table_id: str, target_table_id: str, instructions: str, source_table_schema: List[Dict]) -> Optional[str]:
        """
        Generates the BigQuery SQL query based on the transformation instructions.

        This is a placeholder and needs significant improvement (as discussed before).
        Consider using an LLM here for more complex transformations.
        """
        sql_select_parts = ["*"]
        sql_where_clause = ""

        instructions_lower = instructions.lower()

        if "filter" in instructions_lower:
            filter_condition = instructions.split("filter by ")[-1].strip()
            sql_where_clause = f"WHERE {filter_condition}"

        if "aggregate" in instructions_lower:
            aggregate_column = instructions.split("aggregate by ")[-1].split()[0]
            sql_select_parts = [f"{aggregate_column}, COUNT(*) as count_rows"]
            sql_query = f"""
                CREATE OR REPLACE TABLE `{target_table_id}` AS
                SELECT {', '.join(sql_select_parts)}
                FROM `{source_table_id}`
                {sql_where_clause}
                GROUP BY {aggregate_column}
            """
            return sql_query

        if "create column" in instructions_lower:
            column_def = instructions.split("create column ")[-1].strip()
            new_column_name = column_def.split(" as ")[0].strip()
            expression = column_def.split(" as ")[-1].strip()
            sql_select_parts.append(f"{expression} as {new_column_name}")

        sql_query = f"""
            CREATE OR REPLACE TABLE `{target_table_id}` AS
            SELECT {', '.join(sql_select_parts)}
            FROM `{source_table_id}`
            {sql_where_clause}
        """
        return sql_query

    def get_table_schema(self, table_id: str) -> List[Dict]:
        """Gets the schema of a BigQuery table."""
        try:
            table = self.client.get_table(table_id)
            schema = [{"name": field.name, "type": field.field_type} for field in table.schema]
            return schema
        except Exception as e:
            print(f"Error getting table schema: {e}")
            return []

    def execute(self) -> Dict:
        """Executes the data transformation."""
        try:
            source_full_table_id = f"{PROJECT_ID}.{RAW_DATASET_ID}.{self.source_table}"
            target_full_table_id = f"{PROJECT_ID}.{DW_DATASET_ID}.{self.target_table}"

            source_table_schema = self.get_table_schema(source_full_table_id)
            if not source_table_schema:
                return {"status": "error", "message": f"Could not retrieve schema for {source_full_table_id}"}

            sql_query = self.generate_sql_query(
                source_full_table_id, target_full_table_id, self.transform_instructions, source_table_schema
            )
            if not sql_query:
                return {"status": "error", "message": "Failed to generate SQL query."}

            job = self.client.query(sql_query)
            job.result()  # Wait for the query to complete

            return {
                "status": "success",
                "message": f"Data transformed and table created: {target_full_table_id}",
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}