import os
import io
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.incidents.evaluator import IncidentEvaluator
from src.rag.assistant import RootCauseAssistant
from src.api.schemas import (
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    InvestigationRequest,
    InvestigationResponse,
    ReferenceProfileResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Data Failure Early-Warning & Root-Cause Intelligence API",
    description="Production-grade API combining deterministic data quality gates with evidence-grounded GenAI investigation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Singletons
evaluator = IncidentEvaluator()
assistant = RootCauseAssistant()

SIMULATED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "simulated_batches")
MANIFEST_PATH = os.path.join(SIMULATED_DIR, "manifest.json")
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "reference", "baseline_profile.json")


@app.get("/health", tags=["System"])
def health_check():
    """Health status and vector index readiness."""
    chunk_count = len(assistant.vector_store.chunks) if assistant.vector_store else 0
    return {
        "status": "healthy",
        "service": "ML Data Failure Intelligence Platform",
        "version": "1.0.0",
        "rag_engine": "SentenceTransformers + FAISS",
        "indexed_runbook_chunks": chunk_count,
        "baseline_ready": os.path.exists(PROFILE_PATH)
    }


@app.get("/profile/reference", response_model=ReferenceProfileResponse, tags=["Profiling"])
def get_reference_profile():
    """Retrieves baseline statistical profile and reference quantile boundaries."""
    if not os.path.exists(PROFILE_PATH):
        raise HTTPException(status_code=404, detail="Baseline profile not found. Run profiler first.")

    with open(PROFILE_PATH, "r") as f:
        profile = json.load(f)

    return {
        "metadata": profile.get("metadata", {}),
        "schema_definitions": profile.get("schema", {}),
        "null_rates": profile.get("null_rates", {}),
        "numeric_summaries": profile.get("numeric_profiles", {}),
        "country_vocabulary_size": len(profile.get("categorical_vocabularies", {}).get("Country", {}).get("unique_values", []))
    }


@app.get("/batches", tags=["Batches"])
def list_available_batches():
    """Lists available simulated batches with clean/corrupted ground-truth labels."""
    if not os.path.exists(MANIFEST_PATH):
        raise HTTPException(status_code=404, detail="Manifest not found.")

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    return {
        "total_batches": len(manifest),
        "batches": [
            {
                "batch_id": v["batch_id"],
                "file_name": v["file_name"],
                "is_clean": v["is_clean"],
                "injected_fault_types": v["injected_fault_types"],
                "expected_action": v["expected_action"],
                "description": v["description"]
            }
            for v in manifest.values()
        ]
    }


@app.post("/batches/analyze", response_model=BatchAnalysisResponse, tags=["Inspection & Gate"])
def analyze_batch(payload: BatchAnalysisRequest):
    """
    Evaluates a batch against deterministic rules and emits an ALLOW, WARN, or BLOCK release decision.
    Accepts batch_id (from simulated suite) or raw batch_data rows.
    """
    df: Optional[pd.DataFrame] = None

    if payload.batch_data:
        df = pd.DataFrame(payload.batch_data)
    elif payload.batch_id:
        batch_path = os.path.join(SIMULATED_DIR, f"{payload.batch_id}.csv")
        if not os.path.exists(batch_path):
            raise HTTPException(status_code=404, detail=f"Batch '{payload.batch_id}' not found in simulated directory.")
        df = pd.read_csv(batch_path)
    else:
        raise HTTPException(status_code=400, detail="Must provide either batch_id or batch_data.")

    result = evaluator.analyze_batch(df, batch_id=payload.batch_id or "custom_batch")
    return result


@app.post("/batches/upload", response_model=BatchAnalysisResponse, tags=["Inspection & Gate"])
async def upload_and_analyze_batch(file: UploadFile = File(...)):
    """Uploads a custom CSV batch and runs deterministic inspection."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        batch_id = file.filename.replace(".csv", "")
        result = evaluator.analyze_batch(df, batch_id=batch_id)
        return result
    except Exception as e:
        logger.error(f"Error parsing uploaded CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")


@app.post("/investigate", response_model=InvestigationResponse, tags=["GenAI Root-Cause Copilot"])
def investigate_incident(payload: InvestigationRequest):
    """
    RAG Investigation Copilot: Retrieves relevant runbook evidence and synthesizes
    evidence-grounded hypotheses with exact citations and actionable diagnostic steps.
    """
    incident_dict = payload.incident

    # If batch_id was provided instead of full incident dict, analyze the batch first
    if not incident_dict and payload.batch_id:
        batch_path = os.path.join(SIMULATED_DIR, f"{payload.batch_id}.csv")
        if not os.path.exists(batch_path):
            raise HTTPException(status_code=404, detail=f"Batch '{payload.batch_id}' not found.")
        df = pd.read_csv(batch_path)
        incident_dict = evaluator.analyze_batch(df, batch_id=payload.batch_id)

    if not incident_dict:
        raise HTTPException(status_code=400, detail="Must provide either 'incident' report or 'batch_id'.")

    report = assistant.investigate(incident_dict, top_k=payload.top_k)
    return report


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
