# Runbook RB-001: Missing Pricing Telemetry & Catalog Ingestion Failures

**ID:** RB-001  
**Category:** Data Quality / Missingness  
**Affected Feature(s):** `UnitPrice`  
**Severity Level:** CRITICAL  
**Owner:** Core Data Ingestion & Catalog Engineering  

---

## 1. Problem Description
The `UnitPrice` feature represents the transactional item price in GBP. In normal production operations, the baseline null rate is < 0.05%. An unexpected spike in missing values (> 4% delta) indicates a failure in upstream product pricing synchronization or catalog database lookups.

## 2. Likely Root Causes
1. **Catalog Service Drop:** The upstream Product Information Management (PIM) API experienced a timeout, causing newly listed SKUs to be ingested with `NULL` price payloads.
2. **Third-Party Merchant Feed Format Shift:** A partner merchant updated their catalog format, omitting the price field or passing empty strings.
3. **Database Migration / Schema Desynchronization:** An incomplete database migration in the catalog service unmapped foreign keys between `StockCode` and the price master table.

## 3. Recommended Investigation Steps
1. **Compare Against Baseline Profile:** Check `data/reference/baseline_profile.json` to confirm baseline null rate vs. current batch delta.
2. **Inspect Upstream Ingestion Logs:** Check logs for the `catalog-sync-service` in the time window matching the batch timestamp.
3. **Audit Unpriced SKUs:** Query the affected batch for which `StockCode` values have missing prices:
   ```sql
   SELECT StockCode, Description, COUNT(*) 
   FROM batch_transactions 
   WHERE UnitPrice IS NULL 
   GROUP BY StockCode, Description;
   ```
4. **Quarantine Batch:** If null rate delta exceeds the critical block threshold (> 8%), enforce the **`BLOCK`** gate to halt downstream demand forecasting.

## 4. Remediation & Recovery
- Trigger a re-sync of the catalog pricing cache: `python -m services.catalog_sync --force-refresh`.
- If missing prices cannot be backfilled within SLA, apply fallback median price imputation tagged with `imputed_flag=True` before re-running validation.
