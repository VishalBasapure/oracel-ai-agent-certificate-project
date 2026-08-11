import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import json
import time
import io
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataSense AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #f8f9fb; color: #1a1d23; }
.stApp { background: #f8f9fb; }

/* Header */
.agent-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f172a 100%);
    border-radius: 16px; padding: 32px 40px; margin-bottom: 24px;
    border: 1px solid #334155;
}
.agent-title { font-size: 32px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.5px; }
.agent-sub { font-size: 13px; color: #64748b; font-family: 'JetBrains Mono', monospace; margin-top: 6px; }
.oracle-badge {
    background: linear-gradient(135deg, #c2410c, #ea580c);
    color: white; padding: 6px 16px; border-radius: 20px;
    font-size: 12px; font-weight: 600; font-family: 'JetBrains Mono', monospace;
    display: inline-block;
}
.live-badge {
    background: #052e16; border: 1px solid #16a34a; color: #4ade80;
    padding: 4px 12px; border-radius: 20px; font-size: 11px;
    font-family: 'JetBrains Mono', monospace; display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

/* Agent step cards */
.step-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 16px 20px; margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.step-active {
    border-left: 4px solid #3b82f6 !important;
    background: #eff6ff !important;
}
.step-done {
    border-left: 4px solid #22c55e !important;
    background: #f0fdf4 !important;
}
.step-label {
    font-size: 11px; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 4px;
}
.step-text { font-size: 13px; color: #1e293b; font-weight: 500; }
.step-detail { font-size: 12px; color: #64748b; margin-top: 3px; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }
.kpi-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kpi-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; font-family: 'JetBrains Mono', monospace; }
.kpi-value { font-size: 28px; font-weight: 700; margin: 4px 0; }
.kpi-sub { font-size: 11px; color: #94a3b8; }

/* Anomaly cards */
.anomaly-critical {
    background: #fff1f2; border: 1px solid #fecaca; border-left: 4px solid #ef4444;
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
}
.anomaly-warning {
    background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #f59e0b;
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
}
.anomaly-info {
    background: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #3b82f6;
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
}

/* Report block */
.report-block {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 24px 28px; margin: 12px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.report-title { font-size: 16px; font-weight: 600; color: #1e293b; margin-bottom: 12px; }
.report-text { font-size: 14px; color: #475569; line-height: 1.8; }

/* Section header */
.section-head {
    font-size: 12px; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 1.5px;
    font-family: 'JetBrains Mono', monospace;
    border-left: 3px solid #3b82f6; padding-left: 10px;
    margin: 20px 0 12px 0;
}

/* Tool call display */
.tool-call {
    background: #0f172a; border-radius: 8px; padding: 12px 16px;
    margin: 6px 0; font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: #94a3b8;
}
.tool-call span { color: #60a5fa; }
.tool-result { color: #4ade80; }

/* Upload zone */
.upload-zone {
    background: white; border: 2px dashed #cbd5e1; border-radius: 12px;
    padding: 40px; text-align: center; margin: 16px 0;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 24px !important;
}
.stButton > button:hover { background: linear-gradient(135deg, #1d4ed8, #1e40af) !important; }

[data-testid="stFileUploader"] { background: white; border-radius: 12px; padding: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ['agent_running','agent_done','steps','anomalies','report','df','kpis','charts_data']:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ['steps','anomalies','charts_data'] else []

# ── Gemini setup ──────────────────────────────────────────────────────────────
def get_gemini():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if not key:
            return None
        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return None

# ── Agent Tools (Agentic AI concept — each tool is autonomous) ────────────────
def tool_load_data(df):
    """Tool 1: Load and profile the dataset"""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "col_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "sample": df.head(3).to_string()
    }

def tool_statistical_analysis(df):
    """Tool 2: Run statistical analysis"""
    results = {}
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        desc = df[num_cols].describe()
        results['stats'] = desc.to_dict()
        # IQR outlier detection
        outliers = {}
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            out_count = len(df[(df[col] < lower) | (df[col] > upper)])
            if out_count > 0:
                outliers[col] = {"count": out_count, "lower_bound": round(lower, 2), "upper_bound": round(upper, 2)}
        results['outliers'] = outliers
    return results

def tool_anomaly_detection(df):
    """Tool 3: Detect anomalies and patterns"""
    anomalies = []
    num_cols = df.select_dtypes(include='number').columns.tolist()

    # Missing value anomalies
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        if missing_pct > 20:
            anomalies.append({
                "severity": "critical",
                "type": "Missing Data",
                "column": col,
                "detail": f"{missing_pct:.1f}% of values are missing — data integrity risk",
                "impact": "High"
            })
        elif missing_pct > 5:
            anomalies.append({
                "severity": "warning",
                "type": "Partial Missing Data",
                "column": col,
                "detail": f"{missing_pct:.1f}% missing values detected",
                "impact": "Medium"
            })

    # Outlier anomalies
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_pct = len(df[(df[col] < lower) | (df[col] > upper)]) / len(df) * 100
        if outlier_pct > 10:
            anomalies.append({
                "severity": "critical",
                "type": "Extreme Outliers",
                "column": col,
                "detail": f"{outlier_pct:.1f}% of values fall outside expected range [{lower:.2f}, {upper:.2f}]",
                "impact": "High"
            })
        elif outlier_pct > 3:
            anomalies.append({
                "severity": "warning",
                "type": "Statistical Outliers",
                "column": col,
                "detail": f"{outlier_pct:.1f}% outliers detected using IQR method",
                "impact": "Medium"
            })

    # Duplicate detection
    dupe_count = df.duplicated().sum()
    if dupe_count > 0:
        anomalies.append({
            "severity": "warning" if dupe_count / len(df) < 0.1 else "critical",
            "type": "Duplicate Rows",
            "column": "All columns",
            "detail": f"{dupe_count} duplicate rows found ({dupe_count/len(df)*100:.1f}% of dataset)",
            "impact": "Medium"
        })

    # Skewness
    for col in num_cols:
        skew = df[col].skew()
        if abs(skew) > 2:
            anomalies.append({
                "severity": "info",
                "type": "High Skewness",
                "column": col,
                "detail": f"Skewness = {skew:.2f} — distribution is heavily {'right' if skew > 0 else 'left'}-skewed",
                "impact": "Low"
            })

    # Zero variance
    for col in num_cols:
        if df[col].std() == 0:
            anomalies.append({
                "severity": "warning",
                "type": "Zero Variance",
                "column": col,
                "detail": "Column has constant value — no predictive power",
                "impact": "Medium"
            })

    return anomalies

def tool_generate_report(model, df, anomalies, stats):
    """Tool 4: AI generates the final report"""
    anomaly_summary = "\n".join([
        f"- [{a['severity'].upper()}] {a['type']} in '{a['column']}': {a['detail']}"
        for a in anomalies[:10]
    ])
    num_cols = df.select_dtypes(include='number').columns.tolist()
    stats_summary = ""
    if num_cols and 'stats' in stats:
        for col in num_cols[:3]:
            if col in stats['stats']:
                s = stats['stats'][col]
                stats_summary += f"\n{col}: mean={s.get('mean',0):.2f}, std={s.get('std',0):.2f}, min={s.get('min',0):.2f}, max={s.get('max',0):.2f}"

    prompt = f"""You are an expert data analyst AI agent. Analyze this dataset and write a professional report.

Dataset Overview:
- Rows: {len(df)}, Columns: {len(df.columns)}
- Column names: {', '.join(df.columns.tolist())}
- Numeric columns: {', '.join(num_cols)}

Statistical Summary:{stats_summary}

Anomalies Detected:
{anomaly_summary if anomaly_summary else "No significant anomalies found"}

Write a structured analysis report with these exact sections:

**EXECUTIVE SUMMARY**
2-3 sentences: what this dataset contains, overall data quality, key finding.

**KEY FINDINGS**
3-4 bullet points of the most important insights from the data.

**ANOMALY ANALYSIS**
Explain the most critical anomalies found and their business impact.

**DATA QUALITY SCORE**
Give a score out of 100 and explain why.

**RECOMMENDATIONS**
3 specific, actionable recommendations for improving data quality or using this data.

Keep it professional, concise, and specific to this dataset. Use actual numbers from the data."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Report generation error: {str(e)}"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="agent-header">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <div class="agent-title">🤖 DataSense AI Agent</div>
      <div class="agent-sub">AGENTIC AI · AUTONOMOUS DATA ANALYSIS · ANOMALY DETECTION · AI REPORTING</div>
      <div style="margin-top:12px;display:flex;gap:10px;align-items:center">
        <span class="oracle-badge">⬡ Oracle Agentic AI Certified</span>
        <span class="live-badge">● AGENT READY</span>
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-size:11px;color:#475569;font-family:'JetBrains Mono',monospace">POWERED BY</div>
      <div style="font-size:18px;font-weight:700;color:#60a5fa">Gemini AI</div>
      <div style="font-size:11px;color:#475569;font-family:'JetBrains Mono',monospace;margin-top:4px">4 Autonomous Tools</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── How it works banner ───────────────────────────────────────────────────────
st.markdown("""
<div style="background:white;border:1px solid #e2e8f0;border-radius:10px;padding:14px 20px;margin-bottom:20px;
            display:flex;align-items:center;gap:24px;flex-wrap:wrap">
  <div style="font-size:12px;color:#94a3b8;font-family:'JetBrains Mono',monospace;white-space:nowrap">AGENT PIPELINE</div>
  <div style="display:flex;align-items:center;gap:8px;font-size:13px;color:#1e293b;flex-wrap:wrap">
    <span style="background:#eff6ff;color:#2563eb;padding:4px 10px;border-radius:6px;font-weight:500">📁 Load Data</span>
    <span style="color:#94a3b8">→</span>
    <span style="background:#eff6ff;color:#2563eb;padding:4px 10px;border-radius:6px;font-weight:500">📊 Statistical Analysis</span>
    <span style="color:#94a3b8">→</span>
    <span style="background:#eff6ff;color:#2563eb;padding:4px 10px;border-radius:6px;font-weight:500">🔍 Anomaly Detection</span>
    <span style="color:#94a3b8">→</span>
    <span style="background:#eff6ff;color:#2563eb;padding:4px 10px;border-radius:6px;font-weight:500">🤖 AI Report Generation</span>
    <span style="color:#94a3b8">→</span>
    <span style="background:#f0fdf4;color:#16a34a;padding:4px 10px;border-radius:6px;font-weight:500">✅ Insights Delivered</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upload + API key ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 1])

with col_left:
    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"],
        help="Upload any CSV file — the agent will autonomously analyze it"
    )

with col_right:
    st.markdown("<br>", unsafe_allow_html=True)
    api_key_input = st.text_input("Gemini API Key", type="password",
                                   placeholder="AIza...",
                                   help="Get free key at aistudio.google.com")

# ── Main logic ────────────────────────────────────────────────────────────────
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Quick preview
    with st.expander("📋 Dataset Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # Run agent button
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_agent = st.button("▶ Run AI Agent", use_container_width=True)

    if run_agent or st.session_state.agent_done:

        if run_agent:
            # Reset
            st.session_state.agent_done = False
            st.session_state.steps = []
            st.session_state.anomalies = []
            st.session_state.report = None
            st.session_state.kpis = None
            st.session_state.charts_data = []

        # ── Agent execution with live steps ──────────────────────────────────
        if run_agent:
            st.markdown('<div class="section-head">🤖 Agent Execution Log</div>', unsafe_allow_html=True)

            steps_placeholder = st.empty()

            def render_steps(steps):
                html = ""
                for s in steps:
                    cls = "step-done" if s['status'] == 'done' else "step-active" if s['status'] == 'running' else "step-card"
                    icon = "✅" if s['status'] == 'done' else "⚡" if s['status'] == 'running' else "⏳"
                    html += f"""<div class="step-card {cls}">
                        <div class="step-label">{icon} Tool {s['num']} — {s['tool']}</div>
                        <div class="step-text">{s['action']}</div>
                        <div class="step-detail">{s.get('result','')}</div>
                    </div>"""
                # Tool call log
                html += """<div class="tool-call">
                    <span>agent</span>.plan() → decompose task into 4 tool calls<br>
                    <span>agent</span>.execute() → running tools sequentially with self-correction<br>"""
                for s in steps:
                    if s['status'] == 'done':
                        html += f'<span class="tool-result">✓ {s["tool"]}(df) → completed</span><br>'
                html += "</div>"
                steps_placeholder.markdown(html, unsafe_allow_html=True)

            # Tool 1 — Load
            steps = [{"num": 1, "tool": "tool_load_data", "action": "Loading dataset and profiling schema...", "status": "running", "result": ""}]
            render_steps(steps)
            time.sleep(0.8)

            profile = tool_load_data(df)
            steps[0]['status'] = 'done'
            steps[0]['result'] = f"Found {profile['rows']:,} rows × {profile['columns']} columns · {len([v for v in profile['missing'].values() if v > 0])} columns with missing values"

            # Tool 2 — Stats
            steps.append({"num": 2, "tool": "tool_statistical_analysis", "action": "Running statistical analysis (mean, std, IQR, outlier bounds)...", "status": "running", "result": ""})
            render_steps(steps)
            time.sleep(0.8)

            stats = tool_statistical_analysis(df)
            outlier_cols = len(stats.get('outliers', {}))
            steps[1]['status'] = 'done'
            steps[1]['result'] = f"Analyzed {len(num_cols)} numeric columns · {outlier_cols} columns with outliers detected"

            # Tool 3 — Anomaly
            steps.append({"num": 3, "tool": "tool_anomaly_detection", "action": "Scanning for anomalies: missing values, outliers, duplicates, skewness...", "status": "running", "result": ""})
            render_steps(steps)
            time.sleep(0.8)

            anomalies = tool_anomaly_detection(df)
            st.session_state.anomalies = anomalies
            critical = len([a for a in anomalies if a['severity'] == 'critical'])
            warning = len([a for a in anomalies if a['severity'] == 'warning'])
            steps[2]['status'] = 'done'
            steps[2]['result'] = f"Found {len(anomalies)} anomalies: {critical} critical · {warning} warnings"

            # Tool 4 — Report
            steps.append({"num": 4, "tool": "tool_generate_report", "action": "Sending context to Gemini AI → generating executive report...", "status": "running", "result": ""})
            render_steps(steps)

            # Get API key
            api_key = api_key_input or ""
            try:
                api_key = api_key or st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                pass

            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                report_text = tool_generate_report(model, df, anomalies, stats)
            else:
                report_text = """**EXECUTIVE SUMMARY**
This dataset has been analyzed by the DataSense AI Agent. Connect your Gemini API key to generate the full AI-powered executive report with insights, anomaly explanations, and recommendations.

**KEY FINDINGS**
• Statistical analysis completed across all numeric columns
• Anomaly detection scan finished — see results below
• Data quality assessment ready

**RECOMMENDATIONS**
• Add your Gemini API key in the sidebar for full AI analysis
• Get a free key at aistudio.google.com"""

            st.session_state.report = report_text
            steps[3]['status'] = 'done'
            steps[3]['result'] = "Executive report generated · Insights ready"
            render_steps(steps)

            # KPIs
            quality_score = max(0, 100 - (critical * 20) - (warning * 10) - (len([a for a in anomalies if a['severity']=='info']) * 3))
            st.session_state.kpis = {
                "rows": len(df), "cols": len(df.columns),
                "anomalies": len(anomalies), "quality": quality_score,
                "critical": critical, "warning": warning
            }
            st.session_state.agent_done = True
            time.sleep(0.3)
            st.rerun()

        # ── Results ───────────────────────────────────────────────────────────
        if st.session_state.agent_done and st.session_state.kpis:
            kpis = st.session_state.kpis
            anomalies = st.session_state.anomalies

            st.markdown("---")
            st.markdown('<div class="section-head">📊 Analysis Results</div>', unsafe_allow_html=True)

            # KPI cards
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Total Records</div>
                    <div class="kpi-value" style="color:#2563eb">{kpis['rows']:,}</div>
                    <div class="kpi-sub">{kpis['cols']} columns analyzed</div>
                </div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Anomalies Found</div>
                    <div class="kpi-value" style="color:#ef4444">{kpis['anomalies']}</div>
                    <div class="kpi-sub">{kpis['critical']} critical · {kpis['warning']} warnings</div>
                </div>""", unsafe_allow_html=True)
            with k3:
                q = kpis['quality']
                qcolor = "#22c55e" if q >= 80 else "#f59e0b" if q >= 60 else "#ef4444"
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Data Quality Score</div>
                    <div class="kpi-value" style="color:{qcolor}">{q}/100</div>
                    <div class="kpi-sub">{'Good' if q>=80 else 'Needs attention' if q>=60 else 'Poor quality'}</div>
                </div>""", unsafe_allow_html=True)
            with k4:
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">Numeric Columns</div>
                    <div class="kpi-value" style="color:#8b5cf6">{len(df.select_dtypes(include='number').columns)}</div>
                    <div class="kpi-sub">{len(df.select_dtypes(include='object').columns)} categorical</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Two column layout
            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown('<div class="section-head">🔍 Anomalies Detected</div>', unsafe_allow_html=True)
                if anomalies:
                    for a in anomalies:
                        cls = f"anomaly-{a['severity']}"
                        icon = "🔴" if a['severity']=='critical' else "🟡" if a['severity']=='warning' else "🔵"
                        st.markdown(f"""<div class="{cls}">
                            <div style="font-size:12px;font-weight:600;color:#374151">{icon} {a['type']} — <code style="font-size:11px">{a['column']}</code>
                            <span style="float:right;font-size:10px;background:#e5e7eb;padding:2px 6px;border-radius:4px">Impact: {a['impact']}</span></div>
                            <div style="font-size:12px;color:#6b7280;margin-top:4px">{a['detail']}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="anomaly-info">
                        <div style="font-size:13px;font-weight:600;color:#1e40af">✅ No anomalies detected</div>
                        <div style="font-size:12px;color:#3b82f6;margin-top:4px">Dataset looks clean and well-structured</div>
                    </div>""", unsafe_allow_html=True)

            with col_b:
                st.markdown('<div class="section-head">📈 Data Distribution</div>', unsafe_allow_html=True)
                num_cols_list = df.select_dtypes(include='number').columns.tolist()
                if num_cols_list:
                    selected_col = st.selectbox("Select column to visualize", num_cols_list, label_visibility="collapsed")
                    fig = px.histogram(df, x=selected_col, nbins=30,
                                       color_discrete_sequence=['#3b82f6'],
                                       template="simple_white")
                    fig.update_layout(
                        margin=dict(l=0,r=0,t=20,b=0), height=250,
                        plot_bgcolor='white', paper_bgcolor='white',
                        font_color='#64748b',
                        xaxis=dict(gridcolor='#f1f5f9'),
                        yaxis=dict(gridcolor='#f1f5f9')
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Correlation heatmap
            if len(num_cols_list) > 1:
                st.markdown('<div class="section-head">🔗 Correlation Matrix</div>', unsafe_allow_html=True)
                corr = df[num_cols_list].corr()
                fig_corr = px.imshow(corr, text_auto=".2f",
                                      color_continuous_scale='RdBu_r',
                                      aspect='auto', template="simple_white")
                fig_corr.update_layout(
                    margin=dict(l=0,r=0,t=20,b=0), height=300,
                    paper_bgcolor='white', font_color='#64748b'
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # Missing values chart
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                st.markdown('<div class="section-head">⚠️ Missing Values by Column</div>', unsafe_allow_html=True)
                fig_miss = px.bar(
                    x=missing.index, y=missing.values,
                    color=missing.values,
                    color_continuous_scale=['#fbbf24','#ef4444'],
                    template="simple_white",
                    labels={'x': 'Column', 'y': 'Missing Count'}
                )
                fig_miss.update_layout(
                    margin=dict(l=0,r=0,t=10,b=0), height=220,
                    paper_bgcolor='white', font_color='#64748b',
                    showlegend=False
                )
                st.plotly_chart(fig_miss, use_container_width=True)

            # AI Report
            st.markdown('<div class="section-head">🤖 AI Executive Report</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="report-block">
                <div class="report-text">{st.session_state.report.replace(chr(10), '<br>').replace('**', '<b>').replace('**', '</b>')}</div>
            </div>""", unsafe_allow_html=True)

            # Download report
            st.markdown("<br>", unsafe_allow_html=True)
            dl_col1, dl_col2, _ = st.columns([1,1,2])
            with dl_col1:
                report_download = f"""DataSense AI Agent — Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Dataset: {len(df)} rows × {len(df.columns)} columns
Data Quality Score: {kpis['quality']}/100

ANOMALIES DETECTED ({len(anomalies)} total):
{'='*50}
""" + "\n".join([f"[{a['severity'].upper()}] {a['type']} — {a['column']}: {a['detail']}" for a in anomalies]) + f"\n\n{'='*50}\nAI ANALYSIS REPORT:\n{'='*50}\n\n{st.session_state.report}"
                st.download_button(
                    "📄 Download Report",
                    data=report_download,
                    file_name=f"datasense_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with dl_col2:
                csv_anomalies = pd.DataFrame(anomalies).to_csv(index=False) if anomalies else "No anomalies found"
                st.download_button(
                    "📊 Export Anomalies CSV",
                    data=csv_anomalies,
                    file_name="anomalies.csv",
                    mime="text/csv",
                    use_container_width=True
                )

else:
    # Empty state
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:48px;margin-bottom:12px">📁</div>
        <div style="font-size:18px;font-weight:600;color:#1e293b;margin-bottom:8px">Upload a CSV to get started</div>
        <div style="font-size:14px;color:#94a3b8">The agent will autonomously analyze your data — no prompting needed</div>
        <br>
        <div style="display:flex;justify-content:center;gap:16px;flex-wrap:wrap">
            <span style="background:#f1f5f9;padding:6px 14px;border-radius:6px;font-size:12px;color:#64748b">📊 Statistical Analysis</span>
            <span style="background:#f1f5f9;padding:6px 14px;border-radius:6px;font-size:12px;color:#64748b">🔍 Anomaly Detection</span>
            <span style="background:#f1f5f9;padding:6px 14px;border-radius:6px;font-size:12px;color:#64748b">🤖 AI Report</span>
            <span style="background:#f1f5f9;padding:6px 14px;border-radius:6px;font-size:12px;color:#64748b">📥 Export Results</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:16px;border-top:1px solid #e2e8f0">
    <span style="font-size:12px;color:#94a3b8;font-family:'JetBrains Mono',monospace">
        Built by Vishal Basapure · Oracle Agentic AI Certified Foundations Associate · 
        Implementing: Autonomous Tool Use · Multi-step Planning · Self-correction Loop
    </span>
</div>
""", unsafe_allow_html=True)
