# Runbook RB-002: Extreme Price Values & 100x Unit Scaling Shifts

**ID:** RB-002  
**Category:** Data Quality / Outliers & Scaling  
**Affected Feature(s):** `UnitPrice`  
**Severity Level:** HIGH  
**Owner:** Billing & Currency Conversion Pipeline  

---

## 1. Problem Description
The platform detected extreme price outliers exceeding the reference outer IQR fence ($Q_3 + 3.0 \times \text{IQR}$) or exceeding the absolute price ceiling. Typically, normal unit prices are < £25.00, but corrupted records exhibit values between £200.00 and £5,000.00.

## 2. Likely Root Causes
1. **Cents vs. Dollars/Pounds Unit Error (100x Scale Shift):** The upstream POS (Point of Sale) or payment gateway emitted amounts in pence/cents (e.g., 255 pence), but the ingestion pipeline parsed the field as whole currency (£255.00) without dividing by 100.
2. **Locale Decimal Separator Parsing Bug:** Regional feeds using commas as decimal separators (e.g., `2,55` in European locales) were parsed by standard float converters as `255.00`.
3. **Wholesale Case Pricing Injected into Single-Unit Stream:** Case-pack or pallet prices were erroneously mapped to single consumer unit price attributes.

## 3. Recommended Investigation Steps
1. **Inspect Multiplier Ratio:** Check if the observed prices in the batch are approximately 100x their historical median in `baseline_profile.json`.
2. **Review Ingestion Transformations:** Verify the decimal parsing logic in the raw transaction ingestion adapter:
   ```bash
   grep -rn "locale" src/ingestion/ or git diff HEAD~1 src/ingestion/
   ```
3. **Audit High-Value Transactions:** Filter records with `UnitPrice > 500`:
   ```sql
   SELECT InvoiceNo, StockCode, UnitPrice, Quantity, Country 
   FROM batch_transactions 
   WHERE UnitPrice > 500.0 
   ORDER BY UnitPrice DESC LIMIT 10;
   ```
4. **Enforce Deterministic Gate:** Mark the batch as **`BLOCK`** if severe outlier ratio > 3% or if hard ceiling is violated.

## 4. Remediation & Recovery
- Apply unit conversion correction: If values are verified to be in pence, divide `UnitPrice` by 100.0.
- Update data ingestion contract parser to enforce explicit ISO currency code and locale-aware decimal formatting.
