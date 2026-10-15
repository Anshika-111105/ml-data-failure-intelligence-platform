# Runbook RB-003: Unregistered Categorical Labels & Geographical Schema Drift

**ID:** RB-003  
**Category:** Schema & Categorical Contract  
**Affected Feature(s):** `Country`  
**Severity Level:** MEDIUM  
**Owner:** Growth & Internationalization Data Operations  

---

## 1. Problem Description
The platform detected categorical values in `Country` that do not exist within the baseline reference vocabulary established in `baseline_profile.json`. Common examples include `UNKNOWN_TERRITORY`, `ATLANTIS_DEPOT`, or unmapped ISO-3166 territory codes.

## 2. Likely Root Causes
1. **New Market Expansion Without Contract Update:** Business operations launched shipping into a new international market without registering the territory in the ML data contract vocabulary.
2. **Reverse IP Geolocation Failure:** A customer checkout IP lookup failed, writing a default placeholder string (e.g., `NULL_ISLAND` or `INVALID_ZONE`) into the transaction record.
3. **Encoding / Text Corruption:** Character encoding mismatch (e.g., UTF-8 read as ASCII) corrupting accented country names.

## 3. Recommended Investigation Steps
1. **Extract Distinct Unknown Labels:** Identify all unregistered labels and their row percentages in the batch:
   ```python
   unknown_set = set(batch_df['Country']) - set(baseline_profile['categorical_vocabularies']['Country']['unique_values'])
   print("Unregistered Country values:", unknown_set)
   ```
2. **Check Geolocation Microservice:** Review logs from the address validation and checkout geolocation services.
3. **Verify Business Context:** Contact the international sales team to confirm whether a new regional launch took place.
4. **Evaluate Release Decision:**
   - If unknown category ratio < 3%: Gate outputs **`ALLOW`** or **`WARN`**.
   - If unknown category ratio >= 3%: Gate outputs **`WARN`** to allow investigation without blocking operational downstream pipelines.

## 4. Remediation & Recovery
- If the new country is legitimate: Update `configs/detection_rules.yaml` and re-profile `baseline_profile.json` to register the new territory.
- If caused by geolocation pipeline drop: Route unmapped transactions to `Other International` or trigger address normalization.
