import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.incidents.evaluator import IncidentEvaluator

SIMULATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "simulated_batches")
MANIFEST_PATH = os.path.join(SIMULATED_DIR, "manifest.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_results.json")


def run_benchmark() -> Dict[str, Any]:
    print("================================================================================")
    print("      ML DATA FAILURE INTELLIGENCE PLATFORM — BENCHMARK EVALUATION GATE       ")
    print("================================================================================\n")

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}. Run fault injector first.")

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    evaluator = IncidentEvaluator()

    total_batches = len(manifest)
    clean_batches = 0
    corrupted_batches = 0

    true_positives = 0   # Corrupted batch correctly flagged
    false_positives = 0  # Clean batch incorrectly flagged
    true_negatives = 0   # Clean batch correctly allowed
    false_negatives = 0  # Corrupted batch missed (allowed)

    fault_type_stats = {
        "MISSING_VALUES": {"total": 0, "detected": 0},
        "EXTREME_OUTLIERS": {"total": 0, "detected": 0},
        "SCHEMA_CATEGORICAL_VIOLATION": {"total": 0, "detected": 0},
        "DISTRIBUTION_DRIFT": {"total": 0, "detected": 0}
    }

    latencies_ms = []
    batch_results = []

    print(f"Executing deterministic inspection across {total_batches} simulated batches...")
    for batch_id, meta in manifest.items():
        batch_file = os.path.join(SIMULATED_DIR, meta["file_name"])
        df = pd.read_csv(batch_file)

        t0 = time.perf_counter()
        result = evaluator.analyze_batch(df, batch_id=batch_id)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(latency)

        is_clean = meta["is_clean"]
        is_flagged = result["is_incident"]
        triggered_checks = result["check_types_triggered"]

        if is_clean:
            clean_batches += 1
            if is_flagged:
                false_positives += 1
            else:
                true_negatives += 1
        else:
            corrupted_batches += 1
            if is_flagged:
                true_positives += 1
            else:
                false_negatives += 1

            # Track per-fault-type recall
            for ft in meta["injected_fault_types"]:
                if ft in fault_type_stats:
                    fault_type_stats[ft]["total"] += 1
                    # Check if the specific check or general incident detected it
                    if ft in triggered_checks:
                        fault_type_stats[ft]["detected"] += 1
                    elif ft == "SCHEMA_CATEGORICAL_VIOLATION" and "SCHEMA_CATEGORICAL_VIOLATION" in triggered_checks:
                        fault_type_stats[ft]["detected"] += 1
                    elif is_flagged:
                        # Counted as detected at incident level
                        fault_type_stats[ft]["detected"] += 1

        batch_results.append({
            "batch_id": batch_id,
            "is_clean": is_clean,
            "injected_fault_types": meta["injected_fault_types"],
            "gate_decision": result["gate_decision"],
            "impact_score": result["impact_score"],
            "latency_ms": round(latency, 2),
            "triggered_checks": triggered_checks
        })

    # Metric Calculations
    recall = true_positives / corrupted_batches if corrupted_batches > 0 else 1.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
    false_alarm_rate = false_positives / clean_batches if clean_batches > 0 else 0.0
    p50_latency = float(np.percentile(latencies_ms, 50))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))

    # Acceptance Gate Targets
    targets = {
        "fault_detection_recall": {"target": ">= 90.0%", "actual": f"{recall*100:.1f}%", "passed": recall >= 0.90},
        "fault_detection_precision": {"target": ">= 85.0%", "actual": f"{precision*100:.1f}%", "passed": precision >= 0.85},
        "clean_batch_false_alarm_rate": {"target": "<= 10.0%", "actual": f"{false_alarm_rate*100:.1f}%", "passed": false_alarm_rate <= 0.10},
        "p95_latency_sec": {"target": "<= 5.0 s", "actual": f"{p95_latency/1000.0:.3f} s", "passed": (p95_latency/1000.0) <= 5.0}
    }

    all_passed = all(t["passed"] for t in targets.values())

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_batches_evaluated": total_batches,
        "clean_batches": clean_batches,
        "corrupted_batches": corrupted_batches,
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        },
        "metrics": {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "false_alarm_rate": round(false_alarm_rate, 4),
            "p50_latency_ms": round(p50_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "p99_latency_ms": round(p99_latency, 2)
        },
        "fault_type_breakdown": {
            k: {
                "total": v["total"],
                "detected": v["detected"],
                "recall": round(v["detected"] / v["total"], 4) if v["total"] > 0 else 1.0
            } for k, v in fault_type_stats.items()
        },
        "acceptance_gate": targets,
        "gate_status": "PASSED" if all_passed else "FAILED",
        "detailed_results": batch_results
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary_report, f, indent=2)

    # Print Formatted CLI Table
    print("\n--------------------------------------------------------------------------------")
    print(f"{'KPI / Acceptance Metric':<32} | {'Target':<12} | {'Actual Measured':<15} | {'Status'}")
    print("--------------------------------------------------------------------------------")
    for kpi, data in targets.items():
        status_str = "PASSED [OK]" if data["passed"] else "FAILED [X]"
        print(f"{kpi:<32} | {data['target']:<12} | {data['actual']:<15} | {status_str}")
    print("--------------------------------------------------------------------------------")
    print(f"\nPer-Fault-Type Detection Recall:")
    for ft, st in summary_report["fault_type_breakdown"].items():
        print(f"  - {ft:<30}: {st['detected']}/{st['total']} ({st['recall']*100:.1f}%)")
    print(f"\nLatency Profiling:")
    print(f"  - P50: {p50_latency:.2f} ms | P95: {p95_latency:.2f} ms | P99: {p99_latency:.2f} ms")
    print(f"\nRelease Gate Overall Status: {'APPROVED' if all_passed else 'REJECTED'}")
    print(f"Results saved to: {OUTPUT_PATH}\n")

    return summary_report


if __name__ == "__main__":
    run_benchmark()
