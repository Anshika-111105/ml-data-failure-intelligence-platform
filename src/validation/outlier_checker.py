from typing import Dict, List, Any
import pandas as pd


class OutlierChecker:
    def __init__(self, baseline_profile: Dict[str, Any], config: Dict[str, Any]):
        self.baseline_profile = baseline_profile
        self.config = config.get("outliers", {})
        self.numeric_profiles = baseline_profile.get("numeric_profiles", {})

    def evaluate(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Evaluates batch against non-parametric reference outlier fences."""
        anomalies = []

        for col, rules in self.config.items():
            if col not in batch_df.columns or col not in self.numeric_profiles:
                continue

            ref_stat = self.numeric_profiles[col]
            series = pd.to_numeric(batch_df[col], errors="coerce").dropna()
            if len(series) == 0:
                continue

            method = rules.get("method", "iqr")
            max_extreme_ratio = rules.get("max_extreme_ratio", 0.03)
            absolute_hard_max = rules.get("absolute_hard_max", 1000.0)
            base_severity = rules.get("severity", "HIGH")

            if method == "iqr":
                upper_bound = ref_stat.get("iqr_upper_outer", ref_stat.get("p75") + 3.0 * ref_stat.get("iqr", 1.0))
                lower_bound = ref_stat.get("iqr_lower_outer", max(0.0, ref_stat.get("p25") - 3.0 * ref_stat.get("iqr", 1.0)))
            else:
                upper_bound = ref_stat.get("mad_upper_bound", ref_stat.get("median") + 4.5 * ref_stat.get("mad", 1.0))
                lower_bound = max(0.0, ref_stat.get("mad_lower_bound", 0.0))

            # Exceeded counts
            extreme_mask = (series > upper_bound) | (series < lower_bound)
            extreme_count = int(extreme_mask.sum())
            extreme_ratio = float(extreme_count / len(series))
            max_val = float(series.max())

            # Hard ceiling check (e.g. 100x scale error)
            hard_max_mask = series > absolute_hard_max
            hard_max_count = int(hard_max_mask.sum())

            if extreme_ratio > max_extreme_ratio or hard_max_count > 0:
                anomalies.append({
                    "check_type": "EXTREME_OUTLIERS",
                    "feature": col,
                    "severity": base_severity,
                    "message": (
                        f"Extreme outliers detected in '{col}': {extreme_count} rows ({extreme_ratio:.1%}) "
                        f"exceed reference outer fence [{lower_bound:.2f}, {upper_bound:.2f}]. "
                        f"Max observed value: {max_val:.2f}."
                    ),
                    "metric_value": round(extreme_ratio, 4),
                    "max_observed": round(max_val, 2),
                    "reference_upper_bound": round(upper_bound, 2),
                    "threshold": max_extreme_ratio,
                    "hard_ceiling_violations": hard_max_count
                })

        return anomalies
