# Runbook RB-005: ML Data Contract SLA & Release Gating Policies

**ID:** RB-005  
**Category:** Governance & Release Gate  
**Affected Feature(s):** All Features (`InvoiceNo`, `StockCode`, `UnitPrice`, `Quantity`, `Country`)  
**Severity Level:** POLICY  
**Owner:** ML Platform Engineering & Data Governance  

---

## 1. Purpose & Gating Philosophy
The ML Data Failure Early-Warning Platform enforces a **deterministic, evidence-based release gate** to prevent silent model performance degradation. Decisions must remain strictly rule-based and audited, independent of generative AI copilots.

## 2. Release Decision Matrix
1. **`ALLOW` (Clean / Low Risk):**
   - Aggregate Impact Score: $< 20$.
   - Zero critical schema violations, null rate deltas within SLA ($\le 4\%$), PSI $< 0.10$.
   - Downstream demand forecasting models execute without manual intervention.
2. **`WARN` (Moderate Anomaly / High Monitoring):**
   - Aggregate Impact Score: $20 \le \text{Score} < 50$.
   - Non-critical category expansions or moderate distribution drift ($0.10 \le \text{PSI} < 0.25$).
   - Downstream models are allowed to proceed, but an incident warning is logged for review.
3. **`BLOCK` (Critical Quality Failure / Release Halt):**
   - Aggregate Impact Score: $\ge 50$ OR any single **CRITICAL** rule triggered (e.g., missing prices $> 8\%$, severe outliers $> 3\%$).
   - Downstream predictions are strictly paused to prevent serving corrupted demand forecasts.

## 3. Emergency Bypass Protocol
If an authorized on-call engineer determines a false alarm (e.g., legitimate Black Friday volume surge):
1. Document justification in the platform audit log.
2. Apply an ephemeral override token with an expiration of 6 hours.
3. Notify the on-call ML Engineer and schedule a baseline re-profiling window.

## 4. Audit Trail Standards
Every evaluated batch must permanently record:
- Batch ID, record count, and evaluation timestamp.
- Computed metrics: null rates, outlier counts, PSI score, KS p-value.
- List of triggered check codes and final gate action (`ALLOW`, `WARN`, `BLOCK`).
