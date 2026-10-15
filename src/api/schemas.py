from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BatchAnalysisRequest(BaseModel):
    batch_id: Optional[str] = Field(default="batch_custom", description="Identifier of the batch")
    batch_data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Raw rows of the batch in JSON format")


class AnomalyItem(BaseModel):
    check_type: str
    feature: str
    severity: str
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


class BatchAnalysisResponse(BaseModel):
    batch_id: str
    row_count: int
    gate_decision: str = Field(description="Deterministic safe release decision: ALLOW, WARN, or BLOCK")
    is_incident: bool
    highest_severity: str
    impact_score: int
    anomaly_count: int
    check_types_triggered: List[str]
    affected_features: List[str]
    anomalies: List[Dict[str, Any]]
    latency_ms: float


class EvidenceCitation(BaseModel):
    citation: str
    runbook_id: str
    title: str
    section: str
    similarity_score: float
    source_file: str
    excerpt: str


class InvestigationResponse(BaseModel):
    batch_id: str
    incident_status: str
    gate_decision: str
    impact_score: int
    summary: str
    hypotheses: List[str]
    evidence_citations: List[EvidenceCitation]
    recommended_investigation: List[str]
    confidence_score: float
    confidence_caveat: str


class InvestigationRequest(BaseModel):
    incident: Optional[Dict[str, Any]] = Field(default=None, description="Incident report from batch analysis")
    batch_id: Optional[str] = Field(default=None, description="Pre-evaluated batch ID (e.g. batch_015)")
    top_k: int = Field(default=3, description="Number of runbook passages to retrieve")


class ReferenceProfileResponse(BaseModel):
    metadata: Dict[str, Any]
    schema_definitions: Dict[str, Any]
    null_rates: Dict[str, float]
    numeric_summaries: Dict[str, Any]
    country_vocabulary_size: int
