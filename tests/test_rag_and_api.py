"""
Automated Pytest Suite for FastAPI Endpoints and RAG Citation Behavior.
Tests REST API contract, deterministic release gate routes, and evidence-grounded RAG retrieval accuracy.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.rag.assistant import RootCauseAssistant
from src.rag.vector_store import RunbookVectorStore

client = TestClient(app)


# ==============================================================================
# 1. System & Profile Endpoints
# ==============================================================================
def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["indexed_runbook_chunks"] >= 20
    assert data["baseline_ready"] is True


def test_api_reference_profile():
    response = client.get("/profile/reference")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "numeric_summaries" in data
    assert "UnitPrice" in data["numeric_summaries"]
    assert "Quantity" in data["numeric_summaries"]
    assert data["country_vocabulary_size"] > 10


def test_api_list_batches():
    response = client.get("/batches")
    assert response.status_code == 200
    data = response.json()
    assert data["total_batches"] == 40
    assert len(data["batches"]) == 40
    # Check both clean and corrupted exist
    clean_count = sum(1 for b in data["batches"] if b["is_clean"])
    assert clean_count == 12


# ==============================================================================
# 2. Batch Analysis & Release Gate Endpoints
# ==============================================================================
def test_api_analyze_clean_batch():
    response = client.post("/batches/analyze", json={"batch_id": "batch_001"})
    assert response.status_code == 200
    data = response.json()
    assert data["batch_id"] == "batch_001"
    assert data["gate_decision"] == "ALLOW"
    assert data["is_incident"] is False
    assert data["anomaly_count"] == 0


def test_api_analyze_corrupted_missing_prices():
    response = client.post("/batches/analyze", json={"batch_id": "batch_015"})
    assert response.status_code == 200
    data = response.json()
    assert data["batch_id"] == "batch_015"
    assert data["gate_decision"] == "BLOCK"
    assert data["is_incident"] is True
    assert "MISSING_VALUES" in data["check_types_triggered"]
    assert "UnitPrice" in data["affected_features"]


def test_api_analyze_corrupted_extreme_outliers():
    response = client.post("/batches/analyze", json={"batch_id": "batch_020"})
    assert response.status_code == 200
    data = response.json()
    assert data["gate_decision"] == "BLOCK"
    assert "EXTREME_OUTLIERS" in data["check_types_triggered"]
    assert "UnitPrice" in data["affected_features"]


# ==============================================================================
# 3. Evidence-Grounded RAG Investigation Copilot Endpoints
# ==============================================================================
def test_api_investigate_incident_citations():
    response = client.post("/investigate", json={"batch_id": "batch_015", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["batch_id"] == "batch_015"
    assert data["gate_decision"] == "BLOCK"
    assert len(data["evidence_citations"]) >= 1
    # Must cite RB-001 for missing UnitPrice
    cited_ids = [c["runbook_id"] for c in data["evidence_citations"]]
    assert "RB-001" in cited_ids
    assert len(data["hypotheses"]) >= 1
    assert len(data["recommended_investigation"]) >= 1
    assert "confidence_caveat" in data


def test_api_investigate_clean_batch():
    response = client.post("/investigate", json={"batch_id": "batch_001"})
    assert response.status_code == 200
    data = response.json()
    assert data["incident_status"] == "CLEAN_BATCH"
    assert data["gate_decision"] == "ALLOW"
    assert len(data["evidence_citations"]) == 0
    assert data["confidence_score"] > 0.90


# ==============================================================================
# 4. RAG Citation Correctness Across All 4 Fault Types
# ==============================================================================
@pytest.mark.parametrize("query,expected_runbook_id", [
    ("missing null UnitPrice catalog synchronization failure", "RB-001"),
    ("extreme outlier 100x scale shift unit conversion error", "RB-002"),
    ("unregistered country category geographic schema violation", "RB-003"),
    ("quantity distribution drift high PSI bulk wholesale orders", "RB-004"),
    ("release gate SLA policy allowable error margin", "RB-005"),
])
def test_rag_citation_retrieval_relevance(query, expected_runbook_id):
    store = RunbookVectorStore()
    hits = store.search(query, top_k=2)
    assert len(hits) > 0
    top_hit_id = hits[0]["runbook_id"]
    assert top_hit_id == expected_runbook_id, f"Query '{query}' expected {expected_runbook_id}, but got {top_hit_id}"
