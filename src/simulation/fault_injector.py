import os
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SIMULATED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "simulated_batches")
MANIFEST_PATH = os.path.join(SIMULATED_DIR, "manifest.json")
FUTURE_POOL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "reference", "future_pool.csv")


class FaultInjector:
    def __init__(self, future_pool_df: pd.DataFrame, batch_size: int = 350, seed: int = 42):
        self.future_df = future_pool_df.copy()
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        os.makedirs(SIMULATED_DIR, exist_ok=True)

    def sample_batch(self, batch_idx: int) -> pd.DataFrame:
        """Samples a representative future batch across multiple invoices."""
        # Use deterministic seed per batch index to ensure 100% reproducibility
        batch = self.future_df.sample(n=self.batch_size, random_state=1000 + batch_idx).copy()
        return batch.reset_index(drop=True)

    def inject_missing_prices(self, batch: pd.DataFrame, null_ratio: float = 0.20) -> pd.DataFrame:
        """Injects null values into the critical UnitPrice feature."""
        corrupted = batch.copy()
        num_nulls = int(len(corrupted) * null_ratio)
        null_indices = self.rng.choice(corrupted.index, size=num_nulls, replace=False)
        corrupted.loc[null_indices, "UnitPrice"] = np.nan
        return corrupted

    def inject_extreme_prices(self, batch: pd.DataFrame, outlier_ratio: float = 0.08, scale_factor: float = 100.0) -> pd.DataFrame:
        """Injects extreme price values (unit conversion errors or 100x scale shift)."""
        corrupted = batch.copy()
        num_outliers = max(1, int(len(corrupted) * outlier_ratio))
        outlier_indices = self.rng.choice(corrupted.index, size=num_outliers, replace=False)
        corrupted.loc[outlier_indices, "UnitPrice"] = corrupted.loc[outlier_indices, "UnitPrice"] * scale_factor
        return corrupted

    def inject_unexpected_categories(self, batch: pd.DataFrame, anomaly_ratio: float = 0.12) -> pd.DataFrame:
        """Injects unexpected country labels or corrupt categorical entries."""
        corrupted = batch.copy()
        num_anomalies = max(1, int(len(corrupted) * anomaly_ratio))
        anomaly_indices = self.rng.choice(corrupted.index, size=num_anomalies, replace=False)
        corrupted.loc[anomaly_indices, "Country"] = self.rng.choice(
            ["UNKNOWN_TERRITORY", "ATLANTIS_DEPOT", "INVALID_ZONE_99", "NULL_ISLAND"],
            size=num_anomalies
        )
        return corrupted

    def inject_quantity_distribution_drift(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Shifts quantity distribution significantly (e.g., bulk wholesale shift)."""
        corrupted = batch.copy()
        # Exponential bulk shift: transform normal retail quantities to heavy wholesale orders
        bulk_shift = self.rng.exponential(scale=35.0, size=len(corrupted)) + 25.0
        corrupted["Quantity"] = (corrupted["Quantity"] + bulk_shift).astype(int)
        return corrupted

    def generate_full_simulation_suite(self, num_clean: int = 12, num_corrupted_per_type: int = 6) -> dict:
        """
        Generates clean control batches and distinct corrupted batches across all 4 fault types
        plus combined multi-fault batches. Returns manifest dict.
        """
        manifest = {}
        batch_counter = 1
        pool_cursor = 0

        # 1. Clean Control Batches (Ground truth: clean, expected_action: ALLOW)
        logger.info(f"Generating {num_clean} Clean Control Batches...")
        for _ in range(num_clean):
            batch_id = f"batch_{batch_counter:03d}"
            batch_df = self.sample_batch(batch_counter)

            file_path = os.path.join(SIMULATED_DIR, f"{batch_id}.csv")
            batch_df.to_csv(file_path, index=False)

            manifest[batch_id] = {
                "batch_id": batch_id,
                "file_name": f"{batch_id}.csv",
                "is_clean": True,
                "row_count": len(batch_df),
                "injected_fault_types": [],
                "affected_features": [],
                "expected_action": "ALLOW",
                "description": "Clean production baseline batch without injected anomalies"
            }
            batch_counter += 1

        # 2. Injected Fault: Missing Prices
        logger.info(f"Generating {num_corrupted_per_type} Missing Price Batches...")
        for _ in range(num_corrupted_per_type):
            batch_id = f"batch_{batch_counter:03d}"
            base_batch = self.sample_batch(batch_counter)
            corrupted = self.inject_missing_prices(base_batch, null_ratio=float(self.rng.uniform(0.15, 0.28)))

            file_path = os.path.join(SIMULATED_DIR, f"{batch_id}.csv")
            corrupted.to_csv(file_path, index=False)

            manifest[batch_id] = {
                "batch_id": batch_id,
                "file_name": f"{batch_id}.csv",
                "is_clean": False,
                "row_count": len(corrupted),
                "injected_fault_types": ["MISSING_VALUES"],
                "affected_features": ["UnitPrice"],
                "expected_action": "BLOCK",
                "description": "High missingness injected in UnitPrice feature"
            }
            batch_counter += 1

        # 3. Injected Fault: Extreme Price Values (Outliers / 100x scale shift)
        logger.info(f"Generating {num_corrupted_per_type} Extreme Price Batches...")
        for _ in range(num_corrupted_per_type):
            batch_id = f"batch_{batch_counter:03d}"
            base_batch = self.sample_batch(batch_counter)
            corrupted = self.inject_extreme_prices(base_batch, outlier_ratio=float(self.rng.uniform(0.06, 0.12)), scale_factor=100.0)

            file_path = os.path.join(SIMULATED_DIR, f"{batch_id}.csv")
            corrupted.to_csv(file_path, index=False)

            manifest[batch_id] = {
                "batch_id": batch_id,
                "file_name": f"{batch_id}.csv",
                "is_clean": False,
                "row_count": len(corrupted),
                "injected_fault_types": ["EXTREME_OUTLIERS"],
                "affected_features": ["UnitPrice"],
                "expected_action": "BLOCK",
                "description": "Unit scaling (100x) and extreme IQR/MAD outlier values in UnitPrice"
            }
            batch_counter += 1

        # 4. Injected Fault: Unexpected Categories / Schema Violations
        logger.info(f"Generating {num_corrupted_per_type} Category Anomaly Batches...")
        for _ in range(num_corrupted_per_type):
            batch_id = f"batch_{batch_counter:03d}"
            base_batch = self.sample_batch(batch_counter)
            corrupted = self.inject_unexpected_categories(base_batch, anomaly_ratio=float(self.rng.uniform(0.08, 0.15)))

            file_path = os.path.join(SIMULATED_DIR, f"{batch_id}.csv")
            corrupted.to_csv(file_path, index=False)

            manifest[batch_id] = {
                "batch_id": batch_id,
                "file_name": f"{batch_id}.csv",
                "is_clean": False,
                "row_count": len(corrupted),
                "injected_fault_types": ["SCHEMA_CATEGORICAL_VIOLATION"],
                "affected_features": ["Country"],
                "expected_action": "WARN",
                "description": "Unregistered / corrupt categorical labels in Country"
            }
            batch_counter += 1

        # 5. Injected Fault: Quantity Distribution Drift (PSI / KS)
        logger.info(f"Generating {num_corrupted_per_type} Quantity Distribution Drift Batches...")
        for _ in range(num_corrupted_per_type):
            batch_id = f"batch_{batch_counter:03d}"
            base_batch = self.sample_batch(batch_counter)
            corrupted = self.inject_quantity_distribution_drift(base_batch)

            file_path = os.path.join(SIMULATED_DIR, f"{batch_id}.csv")
            corrupted.to_csv(file_path, index=False)

            manifest[batch_id] = {
                "batch_id": batch_id,
                "file_name": f"{batch_id}.csv",
                "is_clean": False,
                "row_count": len(corrupted),
                "injected_fault_types": ["DISTRIBUTION_DRIFT"],
                "affected_features": ["Quantity"],
                "expected_action": "WARN",
                "description": "Substantial distribution drift in Quantity (High PSI & low KS p-value)"
            }
            batch_counter += 1

        # 6. Combined Multi-Fault Batches
        logger.info("Generating Combined Multi-Fault Batches...")
        for _ in range(4):
            batch_id = f"batch_{batch_counter:03d}"
            base_batch = self.sample_batch(batch_counter)
            corrupted = self.inject_missing_prices(base_batch, null_ratio=0.18)
            corrupted = self.inject_quantity_distribution_drift(corrupted)

            file_path = os.path.join(SIMULATED_DIR, f"{batch_id}.csv")
            corrupted.to_csv(file_path, index=False)

            manifest[batch_id] = {
                "batch_id": batch_id,
                "file_name": f"{batch_id}.csv",
                "is_clean": False,
                "row_count": len(corrupted),
                "injected_fault_types": ["MISSING_VALUES", "DISTRIBUTION_DRIFT"],
                "affected_features": ["UnitPrice", "Quantity"],
                "expected_action": "BLOCK",
                "description": "Multi-fault compound failure: price missingness + demand distribution drift"
            }
            batch_counter += 1

        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Generated {len(manifest)} total simulated batches ({num_clean} clean, {len(manifest) - num_clean} corrupted).")
        logger.info(f"Manifest written to {MANIFEST_PATH}")
        return manifest


if __name__ == "__main__":
    if not os.path.exists(FUTURE_POOL_PATH):
        raise FileNotFoundError(f"Future pool not found at {FUTURE_POOL_PATH}. Run profiler first.")
    future_df = pd.read_csv(FUTURE_POOL_PATH)
    injector = FaultInjector(future_df, batch_size=500)
    manifest = injector.generate_full_simulation_suite()
    print(f"Simulation generation complete: {len(manifest)} batches created.")
