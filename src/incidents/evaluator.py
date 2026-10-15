import os
import json
import yaml
import time
from typing import Dict, List, Any, Optional
import pandas as pd

from src.validation.schema_checker import SchemaChecker
from src.validation.missingness_checker import MissingnessChecker
from src.validation.outlier_checker import OutlierChecker
from src.drift.distribution_drift import DriftDetector

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "detection_rules.yaml")
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "reference", "baseline_profile.json")
REF_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "reference", "reference_data.csv")


class IncidentEvaluator:
    def __init__(self, config_path: str = CONFIG_PATH, profile_path: str = PROFILE_PATH, ref_data_path: str = REF_DATA_PATH):
        # 1. Load Configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 2. Load Baseline Profile
        with open(profile_path, "r") as f:
            self.baseline_profile = json.load(f)

        # 3. Load Sample Reference Data for KS test if available
        self.ref_sample = None
        if os.path.exists(ref_data_path):
            try:
                self.ref_sample = pd.read_csv(ref_data_path, nrows=5000)
            except Exception:
                self.ref_sample = None

        # 4. Instantiate Checkers
        self.schema_checker = SchemaChecker(self.baseline_profile, self.config)
        self.missingness_checker = MissingnessChecker(self.baseline_profile, self.config)
        self.outlier_checker = OutlierChecker(self.baseline_profile, self.config)
        self.drift_detector = DriftDetector(self.baseline_profile, self.config, self.ref_sample)

        # 5. Gating Weights
        self.weights = self.config.get("gating_policy", {}).get("severity_weights", {
            "CRITICAL": 100,
            "HIGH": 50,
            "MEDIUM": 20,
            "LOW": 5
        })
        self.thresholds = self.config.get("gating_policy", {}).get("thresholds", {
            "block_score": 50,
            "warn_score": 20
        })

    def analyze_batch(self, batch_df: pd.DataFrame, batch_id: str = "batch_unnamed") -> Dict[str, Any]:
        """Runs all multi-signal validation and drift checks and makes release gate decision."""
        start_time = time.perf_counter()
        anomalies: List[Dict[str, Any]] = []

        # Run checks in pipeline order
        anomalies.extend(self.schema_checker.evaluate(batch_df))
        anomalies.extend(self.missingness_checker.evaluate(batch_df))
        anomalies.extend(self.outlier_checker.evaluate(batch_df))
        anomalies.extend(self.drift_detector.evaluate(batch_df))

        # Calculate incident impact score
        total_score = 0
        has_critical = False
        highest_severity = "CLEAN"
        severity_order = {"CLEAN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

        for anomaly in anomalies:
            sev = anomaly.get("severity", "LOW")
            total_score += self.weights.get(sev, 10)
            if sev == "CRITICAL":
                has_critical = True
            if severity_order.get(sev, 0) > severity_order.get(highest_severity, 0):
                highest_severity = sev

        # Release Gating Logic
        if has_critical or total_score >= self.thresholds.get("block_score", 50):
            action = "BLOCK"
        elif total_score >= self.thresholds.get("warn_score", 20):
            action = "WARN"
        else:
            action = "ALLOW"

        latency_sec = time.perf_counter() - start_time
        affected_features = sorted(list(set(a.get("feature") for a in anomalies if a.get("feature"))))
        check_types = sorted(list(set(a.get("check_type") for a in anomalies if a.get("check_type"))))

        return {
            "batch_id": batch_id,
            "row_count": len(batch_df),
            "gate_decision": action,
            "is_incident": (action != "ALLOW"),
            "highest_severity": highest_severity,
            "impact_score": total_score,
            "anomaly_count": len(anomalies),
            "check_types_triggered": check_types,
            "affected_features": affected_features,
            "anomalies": anomalies,
            "latency_ms": round(latency_sec * 1000, 2)
        }


if __name__ == "__main__":
    evaluator = IncidentEvaluator()
    sim_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "simulated_batches")
    test_batch = pd.read_csv(os.path.join(sim_dir, "batch_001.csv"))
    res = evaluator.analyze_batch(test_batch, batch_id="batch_001")
    print(f"Batch 001 result: Gate = {res['gate_decision']}, Anomalies = {res['anomaly_count']}, Latency = {res['latency_ms']} ms")
