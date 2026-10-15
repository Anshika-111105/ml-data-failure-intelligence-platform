from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(actual: np.ndarray, expected_edges: list, expected_props: list, epsilon: float = 1e-4) -> float:
    """
    Computes Population Stability Index (PSI) using reference bin edges and expected proportions.
    PSI = sum((Actual - Expected) * ln((Actual + eps) / (Expected + eps)))
    """
    if len(actual) == 0 or len(expected_edges) < 2 or len(expected_props) == 0:
        return 0.0

    counts, _ = np.histogram(actual, bins=expected_edges)
    actual_props = counts / len(actual)

    # Pad or slice to match lengths
    actual_props = np.array(actual_props, dtype=float)
    expected_props = np.array(expected_props, dtype=float)

    # Add epsilon smoothing
    actual_smooth = np.clip(actual_props, epsilon, 1.0)
    expected_smooth = np.clip(expected_props, epsilon, 1.0)

    # Renormalize
    actual_smooth /= actual_smooth.sum()
    expected_smooth /= expected_smooth.sum()

    psi_value = np.sum((actual_smooth - expected_smooth) * np.log(actual_smooth / expected_smooth))
    return float(np.round(psi_value, 4))


class DriftDetector:
    def __init__(self, baseline_profile: Dict[str, Any], config: Dict[str, Any], reference_df: pd.DataFrame = None):
        self.baseline_profile = baseline_profile
        self.config = config.get("distribution_drift", {})
        self.numeric_profiles = baseline_profile.get("numeric_profiles", {})
        self.reference_df = reference_df
        self.features_to_check = self.config.get("numerical_features", ["Quantity", "UnitPrice"])

    def evaluate(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Evaluates batch for statistical distribution drift."""
        anomalies = []
        psi_cfg = self.config.get("psi", {})
        ks_cfg = self.config.get("ks_test", {})

        moderate_threshold = psi_cfg.get("moderate_drift_threshold", 0.15)
        severe_threshold = psi_cfg.get("severe_drift_threshold", 0.25)
        ks_alpha = ks_cfg.get("alpha", 0.005)

        for col in self.features_to_check:
            if col not in batch_df.columns or col not in self.numeric_profiles:
                continue

            ref_stat = self.numeric_profiles[col]
            batch_series = pd.to_numeric(batch_df[col], errors="coerce").dropna().values
            if len(batch_series) < 10:
                continue

            # 1. Compute PSI
            bin_edges = ref_stat.get("bin_edges", [])
            bin_props = ref_stat.get("bin_proportions", [])
            psi_score = calculate_psi(batch_series, bin_edges, bin_props)

            # 2. Compute KS-test if reference samples are available
            ks_p_value = 1.0
            ks_stat = 0.0
            if self.reference_df is not None and col in self.reference_df.columns:
                ref_sample = self.reference_df[col].dropna().values
                if len(ref_sample) > 500:
                    # Subsample reference for fast, stable computation
                    ref_sample = np.random.choice(ref_sample, size=500, replace=False)
                res = stats.ks_2samp(batch_series, ref_sample)
                ks_stat = float(np.round(res.statistic, 4))
                ks_p_value = float(res.pvalue)

            # Assess Drift Severity
            if psi_score >= severe_threshold or (psi_score >= moderate_threshold and ks_p_value < ks_alpha):
                severity = "HIGH" if psi_score >= severe_threshold else "MEDIUM"
                anomalies.append({
                    "check_type": "DISTRIBUTION_DRIFT",
                    "feature": col,
                    "severity": severity,
                    "message": (
                        f"Significant distribution shift in '{col}': PSI={psi_score:.4f} "
                        f"(threshold: {severe_threshold:.2f}), KS p-value={ks_p_value:.2e}."
                    ),
                    "metric_value": psi_score,
                    "psi": psi_score,
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_p_value,
                    "threshold": severe_threshold
                })
            elif psi_score >= moderate_threshold:
                anomalies.append({
                    "check_type": "DISTRIBUTION_DRIFT",
                    "feature": col,
                    "severity": "LOW",
                    "message": f"Moderate distribution shift in '{col}': PSI={psi_score:.4f} (threshold: {moderate_threshold:.2f}).",
                    "metric_value": psi_score,
                    "psi": psi_score,
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_p_value,
                    "threshold": moderate_threshold
                })

        return anomalies
