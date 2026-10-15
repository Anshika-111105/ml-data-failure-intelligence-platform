# ML Data Failure Early-Warning & Root-Cause Intelligence Platform

A production-oriented ML engineering system combining **deterministic data quality gates**, **statistical anomaly & drift detection**, **impact-aware release gating**, and an **evidence-grounded GenAI root-cause copilot**.

---

## The Specific Problem

In production machine learning, a model can be mathematically sound, perfectly trained, and cleanly served—yet silently emit erroneous, unreliable predictions because its **upstream data pipeline silently changed**:
- Product prices arrive with high missingness rates.
- Units scale by 100x (pence parsed as pounds or decimal shifts).
- Category fields introduce unexpected, unregistered labels.
- Workload demand or quantity distributions shift dramatically.

Most monitoring systems either passively compute drift or rely on disconnected LLM chatbots that hallucinate root causes. This platform solves this by providing:
1. **Multi-Signal Incident Detection**: Explicit deterministic thresholds and statistical tests (PSI, KS test, IQR/MAD, missingness deltas).
2. **Impact-Aware Safe ML Release Gate**: Deterministic **`ALLOW`**, **`WARN`**, or **`BLOCK`** decisions strictly separated from generative AI.
3. **Evidence-Grounded GenAI Copilot**: FAISS dense vector search over verified operational runbooks, synthesizing root-cause hypotheses with exact source citations and explicit uncertainty caveats.

---

## System Architecture

```text
                                  ┌────────────────────────────────────────┐
                                  │ UCI Online Retail (530k+ Transactions) │
                                  └───────────────────┬────────────────────┘
                                                      │
                            ┌─────────────────────────┴─────────────────────────┐
                            ▼                                                   ▼
                 [Reference Baseline Slice]                          [Future Evaluation Pool]
                   (70% Historical Split)                              (30% Future Split)
                            │                                                   │
                            ▼                                                   ▼
                 [Baseline Profile Engine]                           [Controlled Fault Injector]
               • Outer IQR & MAD Bounds                            • 12 Clean Control Batches
               • Reference PSI Bin Edges                           • 28 Corrupted Batches:
               • Null-Rate Baselines                                 - Missing Prices (Nulls > 15%)
               • Category Vocabularies                               - Extreme Outliers (100x Scale)
                            │                                        - Schema / Category Violations
                            │                                        - Quantity Distribution Drift
                            │                                                   │
                            └─────────────────────────┬─────────────────────────┘
                                                      │
                                                      ▼
                                  ┌────────────────────────────────────────┐
                                  │      Multi-Signal Detection Engine     │
                                  ├────────────────────────────────────────┤
                                  │ • Schema & Categorical Checker         │
                                  │ • Delta Null-Rate Missingness Checker  │
                                  │ • Robust Non-Parametric Outlier (IQR)  │
                                  │ • Population Stability Index (PSI)     │
                                  │ • Kolmogorov-Smirnov 2-Sample Test     │
                                  └───────────────────┬────────────────────┘
                                                      │
                                                      ▼
                                  ┌────────────────────────────────────────┐
                                  │     Deterministic Safe Release Gate    │
                                  ├────────────────────────────────────────┤
                                  │      ALLOW   │   WARN   │   BLOCK      │
                                  └───────────────────┬────────────────────┘
                                                      │
                                    ┌─────────────────┴─────────────────┐
                                    ▼                                   ▼
        ┌──────────────────────────────────────────────┐    ┌───────────────────────────────────┐
        │        FastAPI REST Service (Port 8000)      │    │  Streamlit Dashboard (Port 8501)  │
        ├──────────────────────────────────────────────┤    ├───────────────────────────────────┤
        │ • GET  /health                               │    │ • Release Gate Status Banner      │
        │ • GET  /profile/reference                    │    │ • Side-by-Side Distribution Plots │
        │ • GET  /batches                              │    │ • Interactive RAG Copilot Tab     │
        │ • POST /batches/analyze                      │    │ • Batch Audit Registry & Metrics  │
        │ • POST /investigate                          │    └───────────────────────────────────┘
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │   Evidence-Grounded RAG Assistant (FAISS)    │
        ├──────────────────────────────────────────────┤
        │ • 5 Verified Operational Runbooks (RB-001..5)│
        │ • Sentence-Transformers (all-MiniLM-L6-v2)   │
        │ • Exact Excerpt Citations: [RB-001: Sec 2]   │
        │ • Actionable Diagnostic SQL/CLI Checklists   │
        │ • Zero-Hallucination Uncertainty Caveats     │
        └──────────────────────────────────────────────┘
```

---

## Measured Benchmark Acceptance KPIs

Evaluated against the ground-truth manifest across **40 simulated batches** (12 clean, 28 corrupted):

| KPI / Acceptance Metric | How Measured | Initial Target | Actual Measured | Gate Status |
|---|---|:---:|:---:|:---:|
| **Fault-Detection Recall** | $\frac{\text{Detected Injected Faults}}{\text{All Injected Faults}}$ | $\ge 90.0\%$ | **100.0%** (28/28) | **PASSED [OK]** |
| **Fault-Detection Precision** | $\frac{\text{True Detected Faults}}{\text{All Flagged Faults}}$ | $\ge 85.0\%$ | **96.6%** (28/29) | **PASSED [OK]** |
| **Clean-Batch False-Alarm Rate** | $\frac{\text{Clean Batches Flagged}}{\text{Total Clean Batches}}$ | $\le 10.0\%$ | **8.3%** (1/12) | **PASSED [OK]** |
| **Detection Latency (P95)** | Batch processing time | $\le 5.0\text{ s}$ | **0.005 s (4.74 ms)** | **PASSED [OK]** |
| **RAG Citation Relevance** | Runbook retrieval relevance | $\ge 90.0\%$ | **100.0%** (5/5) | **PASSED [OK]** |
| **Reproducibility** | Deterministic configuration | $100\%$ | **100%** | **PASSED [OK]** |

---

## Operational Runbook Corpus (`docs/runbooks/`)

1. **`RB-001: Missing Pricing Telemetry & Catalog Ingestion Failures`**  
   *Covers:* Upstream catalog synchronization drops, unmapped merchant SKUs, price null propagation.
2. **`RB-002: Extreme Price Values & 100x Unit Scaling Shifts`**  
   *Covers:* Cents vs. dollars parsing bugs, European decimal comma separators, case-pack pricing anomalies.
3. **`RB-003: Unregistered Categorical Labels & Geographical Schema Drift`**  
   *Covers:* New market geographic expansions, reverse-IP geolocation dropouts (`NULL_ISLAND`, `ATLANTIS_DEPOT`).
4. **`RB-004: Quantity Distribution Drift & Demand Volume Shifting`**  
   *Covers:* Bulk B2B buyer ordering spikes, flash promotions, scraper bot reservation surges.
5. **`RB-005: ML Data Contract SLA & Release Gating Policies`**  
   *Covers:* Gating matrix (`ALLOW`/`WARN`/`BLOCK`), emergency bypass tokens, audit logging standards.

---

## How to Run Locally

### 1. Run Automated Test Suite (26 Tests)
```bash
# Runs all detection engine tests + FastAPI and RAG citation tests
python -m pytest tests/ -v
```

### 2. Run Quantitative Benchmark Runner
```bash
python -m evaluation.run_benchmark
```

### 3. Launch the FastAPI REST Service
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
# Interactive Swagger Documentation: http://127.0.0.1:8000/docs
```

### 4. Launch the Streamlit Incident Dashboard
```bash
streamlit run dashboard.py
# Web Interface: http://localhost:8501
```

---

## Interview & Portfolio Talking Points

1. **Why Not Use an LLM to Detect Numerical Anomalies?**  
   *Answer:* LLMs are non-deterministic, computationally expensive, and poor at calculating numerical quantiles or statistical divergences. We use explicit statistical tests (PSI, KS-test, IQR/MAD) for deterministic gating, reserving the GenAI component strictly for diagnostic retrieval and evidence synthesis.
2. **How is Zero-Hallucination Enforced in the RAG Copilot?**  
   *Answer:* The RAG assistant uses closed-domain passage retrieval over versioned runbooks. Every hypothesis is explicitly tied to a source tag (e.g., `[RB-001: 2. Likely Root Causes]`), and the report includes an explicit caveat stating causes are evidence-backed hypotheses requiring upstream log corroboration.
3. **Difference Between Data Defect vs. Legitimate Business Shift:**  
   *Answer:* Missing prices or 100x decimal scale jumps are unambiguous data pipeline defects (`BLOCK`). Sudden bulk order spikes or high quantity PSI can be legitimate business shifts (e.g., Black Friday). The platform supports an emergency override bypass protocol with audit logging.
4. **Proposed Production Extensions:**  
   - Kafka / EventBridge streaming ingestion for micro-batch validation.
   - Centralized metadata store (e.g., SQLite/PostgreSQL) tracking long-term drift trajectories.
   - Slack / PagerDuty webhook alerting for `BLOCK` decisions.
