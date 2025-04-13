from google.adk.agents import Agent
from pathlib import Path
from typing import Optional
import pandas as pd
from google.cloud import storage, bigquery
from io import StringIO
import pyarrow

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def process_design_document(document_path: Optional[str] = None) -> dict:
    """Process design document using RAG with improved path handling"""
    try:
        # Use pathlib for OS-agnostic path handling
        default_path = Path.home() / "Desktop" / "Hackathon" / "rag_agent" / "design_document.pdf"
        doc_path = Path(document_path) if document_path else default_path

        # Verify file existence
        if not doc_path.is_file():
            return {"status": "error", "message": f"File not found: {doc_path}"}

        # Read document content
        with doc_path.open('r', encoding='utf-8') as file:
            content = file.read()

        # Initialize FAISS vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local("vector_store/", embeddings)

        # Process document
        qa_chain = RetrievalQA.from_chain_type(
            llm="openai-gpt",
            retriever=vector_store.as_retriever()
        )
        instructions = qa_chain.run(content)

        return {"status": "success", "instructions": instructions}

    except Exception as e:
        return {"status": "error", "message": str(e)}

def generate_etl_pipeline(instructions: str) -> dict:
    """Generates ETL pipeline steps with enhanced validation"""
    pipeline_steps = []
    
    # Use case-insensitive checks
    instruction_lower = instructions.lower()
    
    if "data ingestion" in instruction_lower:
        pipeline_steps.append("Ingest data from Cloud Storage into BigQuery")
    
    if "data transformation" in instruction_lower:
        pipeline_steps.append("Apply transformations using BigQuery SQL")
    
    if "load data" in instruction_lower:
        pipeline_steps.append("Load transformed data into final BigQuery tables")

    return {
        "status": "success" if pipeline_steps else "error",
        "pipeline_steps": pipeline_steps,
        "message": "No valid instructions found" if not pipeline_steps else ""
    }

root_agent = Agent(
    name="etl_pipeline_agent",
    model="gemini-2.0-flash",
    description="Agent for designing ETL pipelines based on design documents.",
    instruction="Read the provided design document and generate ETL pipeline steps.",
    tools=[process_design_document, generate_etl_pipeline]
)
