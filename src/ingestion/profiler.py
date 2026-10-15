import os
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "reference")
PROFILE_PATH = os.path.join(REFERENCE_DIR, "baseline_profile.json")
REFERENCE_DATA_PATH = os.path.join(REFERENCE_DIR, "reference_data.csv")
FUTURE_POOL_PATH = os.path.join(REFERENCE_DIR, "future_pool.csv")


def compute_column_numeric_profile(series: pd.Series, bins: int = 10) -> dict:
    """Computes comprehensive non-parametric and parametric statistics for a numeric feature."""
    valid_series = series.dropna()
    if len(valid_series) == 0:
        return {}

    q1 = float(valid_series.quantile(0.25))
    median = float(valid_series.quantile(0.50))
    q3 = float(valid_series.quantile(0.75))
    iqr = float(q3 - q1)
    mad = float(np.median(np.abs(valid_series - median)))

    # Compute quantile-based bin edges for PSI calculation (10 equal quantiles)
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(valid_series, quantiles)).tolist()
    # Ensure minimum 2 edges
    if len(bin_edges) < 2:
        bin_edges = [float(valid_series.min()), float(valid_series.max() + 1e-5)]
    else:
        # Extend bounds to infinity/safe ranges
        bin_edges[0] = -1e9
        bin_edges[-1] = 1e9

    # Reference frequencies in these bins
    counts, _ = np.histogram(valid_series, bins=bin_edges)
    proportions = (counts / len(valid_series)).tolist()

    return {
        "count": int(len(valid_series)),
        "null_count": int(series.isna().sum()),
        "null_rate": float(series.isna().mean()),
        "mean": float(valid_series.mean()),
        "std": float(valid_series.std()),
        "min": float(valid_series.min()),
        "p01": float(valid_series.quantile(0.01)),
        "p05": float(valid_series.quantile(0.05)),
        "p10": float(valid_series.quantile(0.10)),
        "p25": q1,
        "median": median,
        "p75": q3,
        "p90": float(valid_series.quantile(0.90)),
        "p95": float(valid_series.quantile(0.95)),
        "p99": float(valid_series.quantile(0.99)),
        "max": float(valid_series.max()),
        "iqr": iqr,
        "mad": mad,
        "iqr_lower_inner": float(q1 - 1.5 * iqr),
        "iqr_upper_inner": float(q3 + 1.5 * iqr),
        "iqr_lower_outer": float(q1 - 3.0 * iqr),
        "iqr_upper_outer": float(q3 + 3.0 * iqr),
        "mad_lower_bound": float(median - 3.0 * 1.4826 * mad),
        "mad_upper_bound": float(median + 3.0 * 1.4826 * mad),
        "bin_edges": bin_edges,
        "bin_proportions": proportions
    }


def build_reference_profile(df: pd.DataFrame, train_ratio: float = 0.70) -> dict:
    """
    Splits data chronologically into Reference and Future Pool,
    then generates the baseline reference statistical profile.
    """
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.sort_values("InvoiceDate").reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    ref_df = df.iloc[:split_idx].copy()
    future_df = df.iloc[split_idx:].copy()

    logger.info(f"Split {len(df)} total rows: {len(ref_df)} reference ({ref_df['InvoiceDate'].min()} to {ref_df['InvoiceDate'].max()}), "
                f"{len(future_df)} future ({future_df['InvoiceDate'].min()} to {future_df['InvoiceDate'].max()})")

    # Save slices for simulation
    ref_df.to_csv(REFERENCE_DATA_PATH, index=False)
    future_df.to_csv(FUTURE_POOL_PATH, index=False)

    profile = {
        "metadata": {
            "version": "1.0.0",
            "dataset": "UCI Online Retail",
            "total_records": int(len(ref_df)),
            "date_range_start": str(ref_df["InvoiceDate"].min()),
            "date_range_end": str(ref_df["InvoiceDate"].max()),
            "columns": list(ref_df.columns)
        },
        "schema": {
            col: str(ref_df[col].dtype) for col in ref_df.columns
        },
        "null_rates": {
            col: float(ref_df[col].isna().mean()) for col in ref_df.columns
        },
        "categorical_vocabularies": {
            "Country": {
                "unique_values": sorted([str(x) for x in ref_df["Country"].dropna().unique().tolist()]),
                "frequencies": {str(k): float(v) for k, v in ref_df["Country"].value_counts(normalize=True).head(20).items()}
            },
            "StockCode": {
                "sample_codes": sorted([str(x) for x in ref_df["StockCode"].dropna().unique()[:50].tolist()]),
                "total_unique": int(ref_df["StockCode"].nunique())
            }
        },
        "numeric_profiles": {
            "UnitPrice": compute_column_numeric_profile(ref_df["UnitPrice"]),
            "Quantity": compute_column_numeric_profile(ref_df["Quantity"])
        }
    }

    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)

    logger.info(f"Saved baseline reference profile to {PROFILE_PATH}")
    return profile


if __name__ == "__main__":
    from src.ingestion.downloader import load_or_fetch_dataset
    df = load_or_fetch_dataset()
    profile = build_reference_profile(df)
    print("Baseline Profile Generated Successfully:")
    print(f"- Total Reference Records: {profile['metadata']['total_records']}")
    print(f"- UnitPrice Median: {profile['numeric_profiles']['UnitPrice']['median']}, IQR: {profile['numeric_profiles']['UnitPrice']['iqr']}")
    print(f"- Quantity Median: {profile['numeric_profiles']['Quantity']['median']}, IQR: {profile['numeric_profiles']['Quantity']['iqr']}")
    print(f"- Country Vocabulary Size: {len(profile['categorical_vocabularies']['Country']['unique_values'])}")
