from typing import Dict, List, Any
import pandas as pd


class MissingnessChecker:
    def __init__(self, baseline_profile: Dict[str, Any], config: Dict[str, Any]):
        self.baseline_profile = baseline_profile
        self.config = config.get("missingness", {})
        self.baseline_null_rates = baseline_profile.get("null_rates", {})

    def evaluate(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Evaluates batch null-rates against baseline and triggers warnings/blocks."""
        anomalies = []
        batch_len = len(batch_df)
        if batch_len == 0:
            return anomalies

        # Evaluate critical columns
        critical_cols = self.config.get("critical_columns", {})
        for col, rules in critical_cols.items():
            if col not in batch_df.columns:
                continue

            current_null_rate = float(batch_df[col].isna().mean())
            baseline_null_rate = float(self.baseline_null_rates.get(col, 0.0))
            delta = current_null_rate - baseline_null_rate

            warn_delta = rules.get("warn_delta", 0.05)
            block_delta = rules.get("block_delta", 0.10)
            base_severity = rules.get("severity", "CRITICAL")

            if delta >= block_delta:
                anomalies.append({
                    "check_type": "MISSING_VALUES",
                    "feature": col,
                    "severity": base_severity,
                    "message": (
                        f"Critical missingness spike in '{col}': {current_null_rate:.1%} "
                        f"(baseline: {baseline_null_rate:.1%}, delta: +{delta:.1%}, block threshold: +{block_delta:.1%})"
                    ),
                    "metric_value": round(current_null_rate, 4),
                    "baseline_value": round(baseline_null_rate, 4),
                    "delta": round(delta, 4),
                    "threshold": block_delta
                })
            elif delta >= warn_delta:
                anomalies.append({
                    "check_type": "MISSING_VALUES",
                    "feature": col,
                    "severity": "MEDIUM",
                    "message": (
                        f"Moderate missingness increase in '{col}': {current_null_rate:.1%} "
                        f"(baseline: {baseline_null_rate:.1%}, delta: +{delta:.1%}, warn threshold: +{warn_delta:.1%})"
                    ),
                    "metric_value": round(current_null_rate, 4),
                    "baseline_value": round(baseline_null_rate, 4),
                    "delta": round(delta, 4),
                    "threshold": warn_delta
                })

        # Evaluate secondary columns
        secondary_cols = self.config.get("secondary_columns", {})
        for col, rules in secondary_cols.items():
            if col not in batch_df.columns:
                continue

            current_null_rate = float(batch_df[col].isna().mean())
            baseline_null_rate = float(self.baseline_null_rates.get(col, 0.0))
            delta = current_null_rate - baseline_null_rate

            warn_delta = rules.get("warn_delta", 0.15)
            if delta >= warn_delta:
                anomalies.append({
                    "check_type": "MISSING_VALUES_SECONDARY",
                    "feature": col,
                    "severity": rules.get("severity", "LOW"),
                    "message": f"Elevated missingness in secondary column '{col}': {current_null_rate:.1%} (delta: +{delta:.1%})",
                    "metric_value": round(current_null_rate, 4),
                    "baseline_value": round(baseline_null_rate, 4),
                    "delta": round(delta, 4),
                    "threshold": warn_delta
                })

        return anomalies
