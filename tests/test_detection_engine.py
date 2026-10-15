import os
import json
import yaml
import pytest
import numpy as np
import pandas as pd

from src.validation.schema_checker import SchemaChecker
from src.validation.missingness_checker import MissingnessChecker
from src.validation.outlier_checker import OutlierChecker
from src.drift.distribution_drift import DriftDetector, calculate_psi
from src.incidents.evaluator import IncidentEvaluator

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PROFILE_PATH = os.path.join(PROJECT_ROOT, "data", "reference", "baseline_profile.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "detection_rules.yaml")
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "evaluation", "benchmark_results.json")


@pytest.fixture(scope="session")
def baseline_profile():
    assert os.path.exists(PROFILE_PATH), f"Baseline profile missing at {PROFILE_PATH}"
    with open(PROFILE_PATH, "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def detection_config():
    assert os.path.exists(CONFIG_PATH), f"Detection config missing at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def clean_sample_df():
    """Loads a verified clean baseline batch from simulated batches."""
    batch_file = os.path.join(PROJECT_ROOT, "data", "simulated_batches", "batch_001.csv")
    if os.path.exists(batch_file):
        return pd.read_csv(batch_file)
    ref_file = os.path.join(PROJECT_ROOT, "data", "reference", "reference_data.csv")
    return pd.read_csv(ref_file, nrows=350)



# 1. Baseline Profile Verification
def test_baseline_profile_structure(baseline_profile):
    assert "metadata" in baseline_profile
    assert "numeric_profiles" in baseline_profile
    assert "categorical_vocabularies" in baseline_profile
    assert baseline_profile["metadata"]["total_records"] > 0
    assert "UnitPrice" in baseline_profile["numeric_profiles"]
    assert "Quantity" in baseline_profile["numeric_profiles"]
    assert len(baseline_profile["categorical_vocabularies"]["Country"]["unique_values"]) > 10


# 2. Schema Checker Tests
def test_schema_checker_clean(baseline_profile, detection_config, clean_sample_df):
    checker = SchemaChecker(baseline_profile, detection_config)
    anomalies = checker.evaluate(clean_sample_df)
    assert len(anomalies) == 0


def test_schema_checker_missing_column(baseline_profile, detection_config, clean_sample_df):
    checker = SchemaChecker(baseline_profile, detection_config)
    corrupted = clean_sample_df.drop(columns=["UnitPrice"])
    anomalies = checker.evaluate(corrupted)
    assert len(anomalies) >= 1
    assert any(a["check_type"] == "SCHEMA_MISSING_COLUMNS" for a in anomalies)


def test_schema_checker_unregistered_category(baseline_profile, detection_config, clean_sample_df):
    checker = SchemaChecker(baseline_profile, detection_config)
    corrupted = clean_sample_df.copy()
    corrupted["Country"] = "TERRA_INCOGNITA_99"
    anomalies = checker.evaluate(corrupted)
    assert len(anomalies) >= 1
    assert any(a["check_type"] == "SCHEMA_CATEGORICAL_VIOLATION" for a in anomalies)


# 3. Missingness Checker Tests
def test_missingness_clean(baseline_profile, detection_config, clean_sample_df):
    checker = MissingnessChecker(baseline_profile, detection_config)
    anomalies = checker.evaluate(clean_sample_df)
    assert len(anomalies) == 0


def test_missingness_critical_price_spike(baseline_profile, detection_config, clean_sample_df):
    checker = MissingnessChecker(baseline_profile, detection_config)
    corrupted = clean_sample_df.copy()
    corrupted.loc[:40, "UnitPrice"] = np.nan  # 20% null
    anomalies = checker.evaluate(corrupted)
    assert len(anomalies) >= 1
    crit_anomaly = [a for a in anomalies if a["feature"] == "UnitPrice"][0]
    assert crit_anomaly["severity"] == "CRITICAL"
    assert crit_anomaly["check_type"] == "MISSING_VALUES"


# 4. Outlier Checker Tests
def test_outlier_checker_clean(baseline_profile, detection_config, clean_sample_df):
    checker = OutlierChecker(baseline_profile, detection_config)
    anomalies = checker.evaluate(clean_sample_df)
    assert len(anomalies) == 0


def test_outlier_checker_extreme_price_spike(baseline_profile, detection_config, clean_sample_df):
    checker = OutlierChecker(baseline_profile, detection_config)
    corrupted = clean_sample_df.copy()
    corrupted.loc[:20, "UnitPrice"] = corrupted.loc[:20, "UnitPrice"] * 100.0  # 100x scale shift
    anomalies = checker.evaluate(corrupted)
    assert len(anomalies) >= 1
    assert any(a["check_type"] == "EXTREME_OUTLIERS" for a in anomalies)


# 5. Drift Detector Tests (PSI & KS)
def test_drift_psi_identical_distribution():
    np.random.seed(42)
    sample_a = np.random.normal(50, 10, 1000)
    sample_b = np.random.normal(50, 10, 1000)
    edges = np.linspace(20, 80, 11).tolist()
    edges[0] = -1e9
    edges[-1] = 1e9
    counts, _ = np.histogram(sample_a, bins=edges)
    props = counts / len(sample_a)

    psi = calculate_psi(sample_b, edges, props)
    assert psi < 0.10, f"Expected stable PSI < 0.10, got {psi}"


def test_drift_psi_shifted_distribution():
    np.random.seed(42)
    sample_ref = np.random.normal(20, 5, 1000)
    sample_shifted = np.random.normal(60, 15, 1000)
    edges = np.linspace(0, 40, 11).tolist()
    edges[0] = -1e9
    edges[-1] = 1e9
    counts, _ = np.histogram(sample_ref, bins=edges)
    props = counts / len(sample_ref)

    psi = calculate_psi(sample_shifted, edges, props)
    assert psi > 0.25, f"Expected severe drift PSI > 0.25, got {psi}"


# 6. Incident Evaluator & Release Gate Tests
def test_evaluator_clean_batch_allow(clean_sample_df):
    evaluator = IncidentEvaluator()
    res = evaluator.analyze_batch(clean_sample_df, batch_id="test_clean")
    assert res["gate_decision"] == "ALLOW"
    assert res["is_incident"] is False
    assert res["anomaly_count"] == 0


def test_evaluator_corrupted_batch_block(clean_sample_df):
    evaluator = IncidentEvaluator()
    corrupted = clean_sample_df.copy()
    corrupted.loc[:50, "UnitPrice"] = np.nan
    res = evaluator.analyze_batch(corrupted, batch_id="test_corrupted")
    assert res["gate_decision"] == "BLOCK"
    assert res["is_incident"] is True
    assert "UnitPrice" in res["affected_features"]


# 7. Benchmark Release Gate KPI Verification
def test_benchmark_kpi_targets_met():
    assert os.path.exists(BENCHMARK_PATH), "Benchmark results file missing. Run benchmark first."
    with open(BENCHMARK_PATH, "r") as f:
        results = json.load(f)

    metrics = results["metrics"]
    assert metrics["recall"] >= 0.90, f"Recall {metrics['recall']} < 0.90"
    assert metrics["precision"] >= 0.85, f"Precision {metrics['precision']} < 0.85"
    assert metrics["false_alarm_rate"] <= 0.10, f"False Alarm Rate {metrics['false_alarm_rate']} > 0.10"
    assert (metrics["p95_latency_ms"] / 1000.0) <= 5.0, f"P95 Latency > 5.0s"
    assert results["gate_status"] == "PASSED"
