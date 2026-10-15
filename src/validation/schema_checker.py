from typing import Dict, List, Any
import pandas as pd


class SchemaChecker:
    def __init__(self, baseline_profile: Dict[str, Any], config: Dict[str, Any]):
        self.baseline_profile = baseline_profile
        self.config = config.get("schema", {})
        self.required_columns = self.config.get("required_columns", [])
        self.known_countries = set(
            baseline_profile.get("categorical_vocabularies", {}).get("Country", {}).get("unique_values", [])
        )

    def evaluate(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Evaluates batch against schema and categorical vocabulary rules."""
        anomalies = []

        # 1. Missing required columns
        missing_cols = [col for col in self.required_columns if col not in batch_df.columns]
        if missing_cols:
            anomalies.append({
                "check_type": "SCHEMA_MISSING_COLUMNS",
                "feature": "SCHEMA",
                "severity": "CRITICAL",
                "message": f"Missing mandatory columns: {missing_cols}",
                "metric_value": len(missing_cols),
                "threshold": 0
            })

        # 2. Categorical vocabulary validation (e.g., Country)
        if "Country" in batch_df.columns and self.known_countries:
            country_cfg = self.config.get("categorical_checks", {}).get("Country", {})
            allow_unknown_ratio = country_cfg.get("allow_unknown_ratio", 0.03)
            severity = country_cfg.get("severity", "MEDIUM")

            series = batch_df["Country"].dropna().astype(str)
            if len(series) > 0:
                unknown_mask = ~series.isin(self.known_countries)
                unknown_count = int(unknown_mask.sum())
                unknown_ratio = float(unknown_count / len(series))
                unknown_labels = series[unknown_mask].unique().tolist()[:5]

                if unknown_ratio > allow_unknown_ratio:
                    anomalies.append({
                        "check_type": "SCHEMA_CATEGORICAL_VIOLATION",
                        "feature": "Country",
                        "severity": severity,
                        "message": (
                            f"Unregistered Country categories detected: {unknown_ratio:.1%} of rows "
                            f"(threshold: {allow_unknown_ratio:.1%}). Sample unknown labels: {unknown_labels}"
                        ),
                        "metric_value": round(unknown_ratio, 4),
                        "threshold": allow_unknown_ratio,
                        "sample_labels": unknown_labels
                    })

        return anomalies
