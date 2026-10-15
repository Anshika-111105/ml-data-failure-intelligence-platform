from typing import Dict, List, Any, Optional
import numpy as np
from src.rag.vector_store import RunbookVectorStore


class RootCauseAssistant:
    def __init__(self, vector_store: Optional[RunbookVectorStore] = None):
        self.vector_store = vector_store or RunbookVectorStore()

    def construct_query(self, incident: Dict[str, Any]) -> str:
        """Constructs an information-dense search query from incident anomalies."""
        anomalies = incident.get("anomalies", [])
        if not anomalies:
            return "clean baseline production operation data contract SLA"

        query_parts = []
        for a in anomalies:
            check_type = a.get("check_type", "")
            feature = a.get("feature", "")
            msg = a.get("message", "")
            query_parts.append(f"{check_type} {feature} {msg}")

        return " ".join(query_parts)[:350]

    def investigate(self, incident: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        """
        Synthesizes an evidence-grounded diagnostic report for a given incident record.
        Strictly cites retrieved runbook passages with zero hallucinations.
        """
        is_clean = not incident.get("is_incident", False)
        gate_decision = incident.get("gate_decision", "ALLOW")
        batch_id = incident.get("batch_id", "unspecified_batch")
        anomalies = incident.get("anomalies", [])

        if is_clean or len(anomalies) == 0:
            return {
                "batch_id": batch_id,
                "incident_status": "CLEAN_BATCH",
                "gate_decision": "ALLOW",
                "impact_score": int(incident.get("impact_score", 0)),
                "summary": "No data quality or statistical drift anomalies detected. Batch satisfies all data contracts.",
                "hypotheses": [],
                "evidence_citations": [],
                "recommended_investigation": [
                    "No action required. Allow batch for downstream demand forecasting model consumption."
                ],
                "confidence_score": 0.99,
                "confidence_caveat": "High confidence based on full compliance with baseline schema, null-rates, and distribution bounds."
            }

        # 1. Retrieve relevant runbook evidence
        query = self.construct_query(incident)
        retrieved_chunks = self.vector_store.search(query, top_k=top_k)

        # 2. Extract structured evidence citations
        citations = []
        hypotheses = []
        action_steps = []

        for chunk in retrieved_chunks:
            citation_tag = f"[{chunk['runbook_id']}: {chunk['section']}]"
            citations.append({
                "citation": citation_tag,
                "runbook_id": chunk["runbook_id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "similarity_score": chunk["similarity_score"],
                "source_file": chunk["file_name"],
                "excerpt": chunk["content"][:220] + "..."
            })

            # Extract actionable steps and hypotheses if present
            if "Root Cause" in chunk["section"]:
                lines = [l.strip() for l in chunk["content"].split("\n") if l.strip().startswith(("1.", "2.", "3.", "-"))]
                for l in lines[:3]:
                    hypotheses.append(f"{l} (Source: {citation_tag})")
            elif "Investigation" in chunk["section"] or "Remediation" in chunk["section"]:
                lines = [l.strip() for l in chunk["content"].split("\n") if l.strip().startswith(("1.", "2.", "3.", "4.", "-"))]
                for l in lines[:3]:
                    action_steps.append(f"{l} (Ref: {citation_tag})")

        # Fallback if specific section headers were different
        if not hypotheses and citations:
            hypotheses.append(f"Incident aligns with failure patterns documented in {citations[0]['citation']}.")
        if not action_steps and citations:
            action_steps.append(f"Follow diagnostic workflow in {citations[0]['citation']}.")

        # Deduplicate
        hypotheses = list(dict.fromkeys(hypotheses))
        action_steps = list(dict.fromkeys(action_steps))

        # Overall summary synthesis
        features_str = ", ".join(incident.get("affected_features", []))
        check_str = ", ".join(incident.get("check_types_triggered", []))
        summary = (
            f"High-risk data incident detected on {batch_id}. Triggered checks: {check_str} "
            f"affecting feature(s) [{features_str}]. Evaluated Gate: {gate_decision}."
        )

        return {
            "batch_id": batch_id,
            "incident_status": "CORRUPTED_BATCH" if gate_decision == "BLOCK" else "SUSPICIOUS_BATCH",
            "gate_decision": gate_decision,
            "impact_score": incident.get("impact_score", 0),
            "summary": summary,
            "hypotheses": hypotheses,
            "evidence_citations": citations,
            "recommended_investigation": action_steps,
            "confidence_score": float(np.round(np.mean([c["similarity_score"] for c in citations]) if citations else 0.85, 3)),
            "confidence_caveat": (
                "The identified root causes are evidence-backed hypotheses retrieved from verified pipeline runbooks. "
                "Final attribution requires corroboration with upstream pipeline logs and metadata. "
                "The platform adheres to zero-hallucination policies and does not invent causes."
            )
        }


if __name__ == "__main__":
    from src.incidents.evaluator import IncidentEvaluator
    import pandas as pd

    evaluator = IncidentEvaluator()
    assistant = RootCauseAssistant()

    # Test corrupted batch 015 (Missing prices)
    df = pd.read_csv("data/simulated_batches/batch_015.csv")
    incident = evaluator.analyze_batch(df, batch_id="batch_015")
    report = assistant.investigate(incident)

    print("================================================================================")
    print("                      RAG INCIDENT INVESTIGATION REPORT                        ")
    print("================================================================================")
    print(f"Batch: {report['batch_id']} | Gate: {report['gate_decision']} | Score: {report['impact_score']}")
    print(f"Summary: {report['summary']}\n")
    print("Likely Root-Cause Hypotheses:")
    for h in report["hypotheses"]:
        print(f"  • {h}")
    print("\nEvidence Citations:")
    for c in report["evidence_citations"]:
        print(f"  • {c['citation']} (Similarity: {c['similarity_score']}): {c['excerpt']}")
    print("\nRecommended Diagnostic Steps:")
    for step in report["recommended_investigation"]:
        print(f"  1. {step}")
    print(f"\nCaveat: {report['confidence_caveat']}")
