# Runbook RB-004: Quantity Distribution Drift & Demand Volume Shifting

**ID:** RB-004  
**Category:** Statistical Drift & Workload Volume  
**Affected Feature(s):** `Quantity`  
**Severity Level:** HIGH  
**Owner:** Demand Forecasting & Workload Analytics Team  

---

## 1. Problem Description
The distribution of transactional order quantities exhibited significant statistical drift relative to the historical reference baseline. The Population Stability Index (PSI) exceeded the critical threshold (> 0.25) or the 2-sample Kolmogorov-Smirnov test yielded $p < 0.005$, indicating that the current batch comes from a fundamentally distinct distribution.

## 2. Likely Root Causes
1. **Bulk B2B Wholesale Spike:** High-volume wholesale buyers placed large bulk orders (e.g., 50–200 items per line item), heavily skewing the right tail of consumer retail orders.
2. **Flash Sale or Promotional Campaign:** A marketing flash sale ("Buy 10 Get 5 Free") shifted the typical single-item basket size toward higher multi-packs.
3. **Automated Scraping or Arbitrage Bot Activity:** Unauthenticated inventory scraping bots triggered automated cart reservations with abnormal quantity increments.
4. **Unit-of-Measure Multiplier Error:** Upstream inventory ERP system emitted box counts instead of individual piece quantities.

## 3. Recommended Investigation Steps
1. **Compare PSI and Quantile Shifts:** Review the PSI score in the incident report and compare batch percentiles ($p_{50}, p_{90}, p_{99}$) against `baseline_profile.json`:
   - Normal retail median is 4.0; bulk shifts often elevate median to > 25.0.
2. **Segment B2B vs. B2C Invoices:** Inspect whether high quantities originate from a small number of wholesale customer accounts (`CustomerID`):
   ```sql
   SELECT CustomerID, COUNT(DISTINCT InvoiceNo), SUM(Quantity), AVG(Quantity) 
   FROM batch_transactions 
   GROUP BY CustomerID 
   ORDER BY SUM(Quantity) DESC LIMIT 5;
   ```
3. **Inspect Campaign Schedule:** Check whether marketing or merchandising launched a major volume discount campaign in the batch timeframe.
4. **Enforce Safe Release Decision:** Release gate emits **`WARN`** or **`BLOCK`** depending on whether downstream forecasting models support high-volume bulk inputs.

## 4. Remediation & Recovery
- Separate B2B wholesale transactions into a dedicated bulk modeling pipeline.
- If bot traffic is confirmed: Apply WAF rate-limiting and filter out automated reservation records before passing data to ML demand models.
