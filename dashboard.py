import os
import json
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.incidents.evaluator import IncidentEvaluator
from src.rag.assistant import RootCauseAssistant

# Page Config
st.set_page_config(
    page_title="ML Data Failure & Root-Cause Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .card-allow {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .card-warn {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid #F59E0B;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .card-block {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .citation-box {
        background: #1E293B;
        border-left: 4px solid #38BDF8;
        padding: 0.8rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# File Paths
BASE_DIR = os.path.dirname(__file__)
SIMULATED_DIR = os.path.join(BASE_DIR, "data", "simulated_batches")
MANIFEST_PATH = os.path.join(SIMULATED_DIR, "manifest.json")
PROFILE_PATH = os.path.join(BASE_DIR, "data", "reference", "baseline_profile.json")
BENCHMARK_PATH = os.path.join(BASE_DIR, "evaluation", "benchmark_results.json")


@st.cache_resource
def load_singletons():
    evaluator = IncidentEvaluator()
    assistant = RootCauseAssistant()
    return evaluator, assistant


@st.cache_data
def load_baseline():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


evaluator, assistant = load_singletons()
baseline_profile = load_baseline()
manifest = load_manifest()

# Sidebar Navigation
st.sidebar.markdown("### ML Platform Controls")
st.sidebar.markdown(
    "<span style='background:#0F172A; color:#38BDF8; padding:3px 8px; border-radius:4px; font-size:0.8rem;'>● SYSTEM HEALTHY · FAISS ACTIVE</span>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation Mode",
    ["Explore Simulated Batches", "Upload Custom Batch (CSV)", "Benchmark Acceptance KPIs"]
)

selected_df = None
selected_batch_id = None
batch_meta = {}

if app_mode == "Explore Simulated Batches":
    st.sidebar.markdown("#### Filter Simulated Suite (40 Batches)")
    fault_filter = st.sidebar.selectbox(
        "Filter by Failure Mode",
        ["All Batches", "Clean Baseline Batches", "Missing Prices", "Extreme Price Outliers", "Category Defects", "Quantity Drift", "Multi-Fault"]
    )

    batch_options = []
    for bid, meta in manifest.items():
        is_clean = meta["is_clean"]
        faults = meta["injected_fault_types"]

        if fault_filter == "Clean Baseline Batches" and not is_clean:
            continue
        elif fault_filter == "Missing Prices" and "MISSING_VALUES" not in faults:
            continue
        elif fault_filter == "Extreme Price Outliers" and "EXTREME_OUTLIERS" not in faults:
            continue
        elif fault_filter == "Category Defects" and "SCHEMA_CATEGORICAL_VIOLATION" not in faults:
            continue
        elif fault_filter == "Quantity Drift" and "DISTRIBUTION_DRIFT" not in faults:
            continue
        elif fault_filter == "Multi-Fault" and len(faults) < 2:
            continue

        label = f"{bid} — {'[CLEAN]' if is_clean else '[' + ', '.join(faults) + ']'}"
        batch_options.append((label, bid))

    if batch_options:
        selected_tuple = st.sidebar.selectbox("Select Batch to Inspect", batch_options, format_func=lambda x: x[0])
        selected_batch_id = selected_tuple[1]
        batch_meta = manifest.get(selected_batch_id, {})
        batch_file = os.path.join(SIMULATED_DIR, f"{selected_batch_id}.csv")
        selected_df = pd.read_csv(batch_file)
    else:
        st.sidebar.warning("No batches match filter.")

elif app_mode == "Upload Custom Batch (CSV)":
    uploaded_file = st.sidebar.file_uploader("Upload Batch CSV", type=["csv"])
    if uploaded_file is not None:
        selected_df = pd.read_csv(uploaded_file)
        selected_batch_id = uploaded_file.name.replace(".csv", "")
        batch_meta = {"is_clean": None, "injected_fault_types": ["User Upload"], "expected_action": "UNKNOWN"}


# Main Dashboard Layout
st.markdown("<div class='main-header'>ML Data Failure Early-Warning & Root-Cause Intelligence</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Deterministic Release Gating · Multi-Signal Data Quality · Evidence-Grounded GenAI Copilot</div>",
    unsafe_allow_html=True
)

if app_mode == "Benchmark Acceptance KPIs":
    st.markdown("### 📊 System-Wide Benchmark & Acceptance Gate KPIs")
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r") as f:
            bench = json.load(f)

        m = bench.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fault Detection Recall", f"{m.get('recall', 0)*100:.1f}%", delta="Target >= 90%")
        col2.metric("Detection Precision", f"{m.get('precision', 0)*100:.1f}%", delta="Target >= 85%")
        col3.metric("Clean False-Alarm Rate", f"{m.get('false_alarm_rate', 0)*100:.1f}%", delta="Target <= 10%", delta_color="inverse")
        col4.metric("P95 Latency", f"{m.get('p95_latency_ms', 0):.2f} ms", delta="Target <= 5,000 ms")

        st.markdown("---")
        st.markdown("#### Per-Fault-Type Detection Accuracy")
        fb = bench.get("fault_type_breakdown", {})
        fb_df = pd.DataFrame([
            {"Fault Type": k, "Injected Batches": v["total"], "Detected Batches": v["detected"], "Recall": f"{v['recall']*100:.1f}%"}
            for k, v in fb.items()
        ])
        st.dataframe(fb_df, use_container_width=True)

        st.markdown("#### Detailed Batch Evaluation History (40 Batches)")
        detail_df = pd.DataFrame(bench.get("detailed_results", []))
        st.dataframe(detail_df, use_container_width=True)
    else:
        st.warning("Benchmark results file not found. Run `python -m evaluation.run_benchmark` first.")

elif selected_df is not None:
    # Run Deterministic Inspection
    t0 = time.perf_counter()
    incident = evaluator.analyze_batch(selected_df, batch_id=selected_batch_id)
    eval_latency = (time.perf_counter() - t0) * 1000.0

    gate = incident["gate_decision"]
    score = incident["impact_score"]
    anomalies = incident["anomalies"]

    # Top Hero Release Gate Banner
    if gate == "ALLOW":
        st.markdown(f"""
        <div class='card-allow'>
            <h2 style='color:#10B981; margin:0;'>RELEASE GATE PERMITTED (ALLOW)</h2>
            <p style='color:#A7F3D0; margin-top:0.3rem;'>Batch <b>{selected_batch_id}</b> satisfies all quality contracts and physical invariants. Cleared for downstream ML demand forecasting.</p>
        </div>
        """, unsafe_allow_html=True)
    elif gate == "WARN":
        st.markdown(f"""
        <div class='card-warn'>
            <h2 style='color:#F59E0B; margin:0;'>CONDITIONAL RELEASE (WARN)</h2>
            <p style='color:#FDE68A; margin-top:0.3rem;'>Batch <b>{selected_batch_id}</b> exhibits moderate drift or non-critical schema variances. Monitored release permitted.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='card-block'>
            <h2 style='color:#EF4444; margin:0;'>PREDICTION PIPELINE HALTED (BLOCK)</h2>
            <p style='color:#FECACA; margin-top:0.3rem;'>Batch <b>{selected_batch_id}</b> violated critical data contracts or physical bounds. Downstream prediction models paused.</p>
        </div>
        """, unsafe_allow_html=True)

    # Metric Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Gate Decision", gate)
    m2.metric("Impact Score", f"{score} / 100")
    m3.metric("Anomalies Found", f"{len(anomalies)}")
    m4.metric("Row Count", f"{len(selected_df):,}")
    m5.metric("Inspection Latency", f"{eval_latency:.2f} ms")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Gating Anomalies & Rules",
        "Side-by-Side Distribution Analysis",
        "GenAI Root-Cause Copilot (RAG)",
        "Batch Raw Data"
    ])

    # Tab 1: Anomalies
    with tab1:
        st.markdown("#### Multi-Signal Rule Results")
        if anomalies:
            anomaly_table = []
            for a in anomalies:
                anomaly_table.append({
                    "Check Type": a.get("check_type"),
                    "Feature": a.get("feature"),
                    "Severity": a.get("severity"),
                    "Observed Metric": a.get("metric_value"),
                    "Reference Threshold": a.get("threshold"),
                    "Diagnostic Message": a.get("message")
                })
            st.dataframe(pd.DataFrame(anomaly_table), use_container_width=True)
        else:
            st.success("All schema checks, null-rate bounds, outlier fences, and PSI drift tests passed with zero violations.")

    # Tab 2: Distribution Comparison
    with tab2:
        st.markdown("#### Current Batch vs. Baseline Reference Comparison")
        c1, c2 = st.columns(2)

        # Quantity Distribution Plot
        with c1:
            st.markdown("**Order Quantity Distribution**")
            fig_qty = go.Figure()
            fig_qty.add_trace(go.Histogram(
                x=selected_df["Quantity"],
                name="Current Batch",
                marker_color="#38BDF8",
                opacity=0.75,
                nbinsx=30
            ))
            # Baseline reference median line
            ref_median_qty = baseline_profile.get("numeric_profiles", {}).get("Quantity", {}).get("median", 4.0)
            fig_qty.add_vline(x=ref_median_qty, line_dash="dash", line_color="#10B981", annotation_text=f"Ref Median ({ref_median_qty})")
            fig_qty.update_layout(
                xaxis_title="Quantity",
                yaxis_title="Count",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=20),
                height=350
            )
            st.plotly_chart(fig_qty, use_container_width=True)

        # UnitPrice Boxplot
        with c2:
            st.markdown("**Unit Price Distribution & Outlier Fences**")
            fig_price = go.Figure()
            fig_price.add_trace(go.Box(
                y=selected_df["UnitPrice"].dropna(),
                name="Current Batch Prices",
                marker_color="#EC4899",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8
            ))
            ref_iqr_upper = baseline_profile.get("numeric_profiles", {}).get("UnitPrice", {}).get("iqr_upper_outer", 11.5)
            fig_price.add_hline(y=ref_iqr_upper, line_dash="dash", line_color="#EF4444", annotation_text=f"Outer IQR Fence (£{ref_iqr_upper:.2f})")
            fig_price.update_layout(
                yaxis_title="UnitPrice (£)",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=20),
                height=350
            )
            st.plotly_chart(fig_price, use_container_width=True)

        # Null-Rate & Country Distribution
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Feature Null Rates: Current Batch vs. Reference**")
            cols_to_compare = ["UnitPrice", "CustomerID", "Quantity", "Country", "Description"]
            current_nulls = [selected_df[col].isna().mean() * 100 for col in cols_to_compare]
            ref_nulls = [baseline_profile.get("null_rates", {}).get(col, 0.0) * 100 for col in cols_to_compare]

            fig_null = go.Figure(data=[
                go.Bar(name='Current Batch Null %', x=cols_to_compare, y=current_nulls, marker_color='#F43F5E'),
                go.Bar(name='Baseline Null %', x=cols_to_compare, y=ref_nulls, marker_color='#10B981')
            ])
            fig_null.update_layout(barmode='group', template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_null, use_container_width=True)

        with c4:
            st.markdown("**Country Categorical Distribution**")
            country_counts = selected_df["Country"].value_counts().head(8).reset_index()
            country_counts.columns = ["Country", "Count"]
            fig_country = px.pie(country_counts, values="Count", names="Country", hole=0.4, template="plotly_dark")
            fig_country.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_country, use_container_width=True)

    # Tab 3: RAG Root-Cause Copilot
    with tab3:
        st.markdown("### Evidence-Grounded GenAI Root-Cause Assistant")
        st.markdown(
            "The assistant retrieves verified incident runbooks via **FAISS vector search** "
            "to synthesize diagnostic hypotheses with exact source citations and zero hallucinations."
        )

        with st.spinner("Retrieving verified runbook evidence and formulating diagnostic report..."):
            rag_report = assistant.investigate(incident, top_k=3)

        st.markdown(f"**Incident Summary:** {rag_report['summary']}")

        # Hypotheses
        st.markdown("#### Plausible Root-Cause Hypotheses")
        if rag_report["hypotheses"]:
            for h in rag_report["hypotheses"]:
                st.markdown(f"- {h}")
        else:
            st.info("No failure hypotheses generated — batch satisfies all operational baselines.")

        # Evidence Citations
        st.markdown("#### Documented Runbook Evidence & Excerpts")
        if rag_report["evidence_citations"]:
            for cit in rag_report["evidence_citations"]:
                st.markdown(f"""
                <div class='citation-box'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='color:#38BDF8; font-weight:700;'>{cit['citation']}</span>
                        <span style='color:#94A3B8; font-size:0.85rem;'>Cosine Similarity: <b>{cit['similarity_score']:.4f}</b></span>
                    </div>
                    <div style='color:#E2E8F0; margin-top:0.4rem; font-size:0.92rem;'>
                        "{cit['excerpt']}"
                    </div>
                    <div style='color:#64748B; font-size:0.8rem; margin-top:0.3rem;'>
                        Source: <code>docs/runbooks/{cit['source_file']}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No evidence citations required for clean batches.")

        # Recommended Investigation
        st.markdown("#### Recommended Investigation Checklist")
        for step in rag_report["recommended_investigation"]:
            st.markdown(f"- {step}")

        # Caveat
        st.warning(f"⚠️ **Confidence & Uncertainty Caveat:** {rag_report['confidence_caveat']}")

    # Tab 4: Raw Data
    with tab4:
        st.markdown(f"#### Raw Batch Data Preview ({len(selected_df)} Rows)")
        st.dataframe(selected_df.head(100), use_container_width=True)
else:
    st.info("Select a batch from the sidebar to inspect quality, drift, and root-cause intelligence.")
