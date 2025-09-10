import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import IsolationForest
import plotly.express as px
import io
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="SOC Dashboard",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and SIEM-like styling
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #1E1E1E; color: #E0E0E0; }
    .stMetric { background-color: #2C2C2C; border: 1px solid #4A4A4A; border-radius: 8px; padding: 15px; }
    .stDataFrame { border: 1px solid #4A4A4A; }
    .stAlert { background-color: #333333; }
    h1, h2, h3 { color: #FFFFFF; }
    .stButton > button { background-color: #4A90E2; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Default Values ---
RANDOM_STATE = 42
DEFAULT_PCT_THRESHOLD = 99.0
DEFAULT_W1 = 0.70
DEFAULT_RULE_WEIGHTS = {
    "wikileaks_flag": 1.0,
    "offhour_usb_flag": 0.7,
    "offhour_http_flag": 0.3,
    "offhour_logon_flag": 0.3,
}

# --- Caching Functions (Pure Computation) ---

@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the uploaded CSV file. Returns (df, error_message)."""
    df = pd.read_csv(uploaded_file)
    required_cols = {"user", "day", "is_malicious"}
    if not required_cols.issubset(df.columns):
        return None, f"CSV must contain the columns: {required_cols}"
    df["user"] = df["user"].astype(str)
    df["day"] = pd.to_datetime(df["day"]).dt.floor("D")
    return df, None

def time_split_train_test(df):
    """Splits data and returns dataframes and a status message."""
    ID_COLS = ['user','day']
    df['day'] = pd.to_datetime(df['day'])
    available_years = sorted(df['day'].dt.year.unique())
    split_year = None
    for year in available_years:
        split_date = pd.Timestamp(f"{year}-06-30")
        has_train_data = (df['day'] <= split_date).any()
        has_test_data = (df['day'] > split_date).any()
        if has_train_data and has_test_data:
            split_year = year
            break
    if split_year is None:
        return None, None, None, "Could not find a year with data both before/on June 30 and after July 1."
    
    toast_message = f"✅ Data split complete. Training on data up to {split_year}-06-30."
    
    final_split_date = pd.Timestamp(f"{split_year}-06-30")
    train_mask = df['day'] <= final_split_date
    keys_train = df.loc[train_mask, ID_COLS].copy().reset_index(drop=True)
    test_mask = df['day'] > final_split_date
    keys_test = df.loc[test_mask, ID_COLS].copy().reset_index(drop=True)
    train_df = df.merge(keys_train, on=ID_COLS, how='inner')
    test_df = df.merge(keys_test, on=ID_COLS, how='inner')
    return train_df, test_df, toast_message, None

@st.cache_data
def train_and_score_model(df):
    """
    Splits data, trains model, and computes scores.
    Returns ( (train_ben, test_df, total_tp), (toast_msg, error_msg) )
    """
    train_df, test_df, toast_message, error = time_split_train_test(df)
    if error:
        return None, (None, error)
        
    train_ben = train_df[train_df["is_malicious"] == 0].reset_index(drop=True)
    if train_ben.empty:
        return None, (None, "No benign rows found in the training data split. Cannot train the model.")

    per_user_z_cols = [c for c in df.columns if ('_user_z' in c)]
    if not per_user_z_cols:
        return None, (None, "No per-user normalized feature columns found (expected names with '_user_z').")

    X_train_if = train_ben[per_user_z_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test_if = test_df[per_user_z_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    clf = IsolationForest(n_estimators=500, contamination="auto", max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_if)

    train_scores = -clf.decision_function(X_train_if)
    test_scores = -clf.decision_function(X_test_if)
    train_ben["anomaly_score"] = train_scores
    test_df["anomaly_score"] = test_scores

    results = (train_ben, test_df.copy(), int(test_df["is_malicious"].sum()))
    status = (toast_message, None)
    return results, status

@st.cache_data
def apply_thresholding(test_df_scored, train_ben_scored, pct_threshold, use_per_user_threshold):
    df = test_df_scored.copy()
    train_scores = train_ben_scored['anomaly_score']
    global_threshold = np.percentile(train_scores, pct_threshold)
    if use_per_user_threshold:
        per_user_thr = train_ben_scored.groupby("user")["anomaly_score"].quantile(pct_threshold / 100.0)
        df["anomaly_threshold"] = df["user"].map(per_user_thr).fillna(global_threshold)
    else:
        df["anomaly_threshold"] = global_threshold
    df["is_model_anomaly"] = (df["anomaly_score"] >= df["anomaly_threshold"]).astype(int)
    return df[df["is_model_anomaly"] == 1].copy()

@st.cache_data
def calculate_hybrid_scores(model_anomalies_df, rule_weights, w1, train_scores):
    if model_anomalies_df.empty:
        return pd.DataFrame()
    df = model_anomalies_df.copy()
    df["rule_score"] = df.apply(lambda r: compute_rule_score_row(r, rule_weights), axis=1)
    max_possible_rule = sum(w for w in rule_weights.values() if w > 0.0) or 1.0
    df["rule_norm"] = (df["rule_score"] / max_possible_rule).clip(0, 1)
    score_min, score_max = train_scores.min(), train_scores.max()
    den = max(1e-12, (score_max - score_min))
    df["anomaly_norm"] = ((df["anomaly_score"] - score_min) / den).clip(0, 1)
    df["weighted_anomaly_score"] = w1 * df["anomaly_norm"]
    df["weighted_rule_score"] = (1.0 - w1) * df["rule_norm"]
    df["hybrid_score"] = df["weighted_anomaly_score"] + df["weighted_rule_score"]
    return df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

# --- Helper Functions (No Caching) ---
def compute_rule_score_row(row, rule_weights):
    score = 0.0
    if int(row.get("wikileaks_flag", 0)) == 1: score += rule_weights["wikileaks_flag"]
    trio = {
        "offhour_usb_flag": int(row.get("offhour_usb_flag", 0)),
        "offhour_http_flag": int(row.get("offhour_http_flag", 0)),
        "offhour_logon_flag": int(row.get("offhour_logon_flag", 0)),
    }
    if trio["offhour_usb_flag"] == 1 or sum(trio.values()) >= 2:
        score += sum(rule_weights[k] for k, v in trio.items() if v == 1)
    return score

def compute_budget_metrics(ranked_df, percents, total_tp):
    rows = []
    for p in percents:
        frac = p / 100.0
        k = max(1, int(math.floor(frac * len(ranked_df))))
        head = ranked_df.head(k)
        n_alerts = len(head)
        tp_in_budget = int(head["is_malicious"].sum())
        precision = (tp_in_budget / n_alerts * 100.0) if n_alerts > 0 else 0.0
        coverage = (tp_in_budget / total_tp * 100.0) if total_tp > 0 else 0.0
        rows.append({"Budget Top %": p, "Top N Alerts": k, "TPs in Budget": tp_in_budget, "Precision (%)": precision, "Coverage of Total TPs (%)": coverage})
    return pd.DataFrame(rows)

def display_alert_card(alert_data, rank, rule_weights):
    is_malicious = alert_data['is_malicious'] == 1
    triggered_rules = ", ".join([flag for flag, weight in rule_weights.items() if alert_data.get(flag, 0) == 1 and weight > 0]) or "None"
    with st.container(border=True):
        c1, c2, c3 = st.columns([0.25, 0.5, 0.25])
        with c1:
            st.markdown(f"**🏅 Rank #{rank}**")
            st.markdown(f"**User:** `{alert_data['user']}`")
            st.markdown(f"**Date:** {alert_data['day'].strftime('%Y-%m-%d')}")
        with c2:
            st.metric(label="Hybrid Score", value=f"{alert_data['hybrid_score']:.4f}")
            st.caption(f"Breakdown: ({alert_data['weighted_anomaly_score']:.3f}) + ({alert_data['weighted_rule_score']:.3f}) | Scaled Anomaly: {alert_data['anomaly_norm']:.3f} | Scaled Rule: {alert_data['rule_norm']:.3f}")
            st.caption(f"Triggered Rules: {triggered_rules}")
        with c3:
            if is_malicious:
                st.error("🔥 **Confirmed Malicious**", icon="🔥")
            else:
                st.success("✅ **Benign**", icon="✅")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload Log", type=["csv"])
    st.markdown("---")
    st.subheader("1. Anomaly Detection")
    pct_threshold = st.slider("Percentile Threshold", 90.0, 100.0, DEFAULT_PCT_THRESHOLD, step=0.1, key="pct")
    use_per_user_threshold = st.toggle("Use Per-User Threshold", value=True, help="If on, uses per-user thresholds. If off, uses a single global threshold.")
    st.subheader("2. Hybrid Reranking")
    w1 = st.slider("Hybrid Weight (w1 for Anomaly Score)", 0.0, 1.0, DEFAULT_W1, step=0.05, key="w1")
    st.caption(f"Hybrid Score = **{w1:.2f}** * Anomaly + **{1-w1:.2f}** * Rule")
    with st.expander("Tune Rule Weights"):
        wikileaks_w = st.number_input("wikileaks_flag", value=DEFAULT_RULE_WEIGHTS["wikileaks_flag"], step=0.1)
        offhour_usb_w = st.number_input("offhour_usb_flag", value=DEFAULT_RULE_WEIGHTS["offhour_usb_flag"], step=0.1)
        offhour_http_w = st.number_input("offhour_http_flag", value=DEFAULT_RULE_WEIGHTS["offhour_http_flag"], step=0.1)
        offhour_logon_w = st.number_input("offhour_logon_flag", value=DEFAULT_RULE_WEIGHTS["offhour_logon_flag"], step=0.1)
    st.markdown("---")
    run_button = st.button("▶️ Run Pipeline", use_container_width=True)

# Initialize a session state flag to show the dashboard after the first run
if 'run_complete' not in st.session_state:
    st.session_state['run_complete'] = False

# --- Main Application Logic ---
if not uploaded_file:
    st.info("Upload a features CSV, adjust controls in the sidebar, and click **Run Pipeline**.")
    st.stop()

# Load data once
df_loaded, error = load_data(uploaded_file)
if error:
    st.error(error)
    st.stop()

# Train model once (per uploaded file)
results, status = train_and_score_model(df_loaded)
toast_msg, error_msg = status
if error_msg:
    st.error(error_msg)
    st.stop()
train_ben_scored, test_df_scored, total_tp = results

# If the button is clicked, show a toast and set the flag to display the dashboard
if run_button:
    if toast_msg:
        st.toast(toast_msg)
    st.session_state.run_complete = True

# After the button has been clicked once, the main dashboard will always display
if st.session_state.run_complete:
    
    # --- DYNAMIC CALCULATION SECTION 1: THRESHOLDING ---
    # This section re-runs whenever the threshold sliders change
    model_anomalies_df = apply_thresholding(
        test_df_scored, train_ben_scored, pct_threshold, use_per_user_threshold
    )
    
    # --- DYNAMIC CALCULATION SECTION 2: HYBRID RANKING ---
    # This section re-runs whenever the hybrid ranking controls change
    rule_weights_dict = {
        "wikileaks_flag": wikileaks_w, "offhour_usb_flag": offhour_usb_w,
        "offhour_http_flag": offhour_http_w, "offhour_logon_flag": offhour_logon_w,
    }
    
    hybrid_ranked_df = calculate_hybrid_scores(
        model_anomalies_df, rule_weights_dict, w1, train_ben_scored['anomaly_score']
    )
    
    model_ranked_df_for_comparison = model_anomalies_df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    # --- Tabbed Interface ---
    st.title("🛡️ SIEM-Style SOC Alert Triage Dashboard")
    st.markdown("##### Anomaly Detection with Isolation Forest & Rule-Based Reranking")

    tab1, tab2, tab3 = st.tabs(["Overview", "Alert Triage", "Investigations"])

    with tab1:
        st.info("Upload data and run the pipeline to triage alerts like a pro SIEM.")
        
        # --- DYNAMIC DISPLAY 1: PERFORMANCE METRICS ---
        st.header("📊 Performance Metrics")
        st.write("These metrics update when the Anomaly Detection controls are changed.")
        model_alerts_count = len(model_anomalies_df)
        model_tp = int(model_anomalies_df["is_malicious"].sum())
        model_coverage = (model_tp / total_tp * 100.0) if total_tp > 0 else 0.0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Malicious Events (in Test Set)", f"{total_tp:,}")
        c2.metric("Total Model-Detected Alerts", f"{model_alerts_count:,}")
        c3.metric("TPs Found by Model (Coverage)", f"{model_coverage:.2f}%")

    with tab2:
        # --- DYNAMIC DISPLAY 2: INTERACTIVE RESULTS ---
        st.markdown("---")
        st.header("🚨 Top 10 Reranked Alerts")
        st.write("The highest priority alerts after applying the hybrid reranking model.")
        display_cols_top10 = [
            "user", "day", "hybrid_score", "anomaly_score", "anomaly_norm", "weighted_anomaly_score",
            "rule_score", "rule_norm", "weighted_rule_score", "is_malicious"
        ]
        if not hybrid_ranked_df.empty and all(col in hybrid_ranked_df.columns for col in display_cols_top10):
            st.dataframe(hybrid_ranked_df[display_cols_top10].head(10).style.format(formatter={
                'hybrid_score': '{:.4f}', 'anomaly_score': '{:.4f}', 'anomaly_norm': '{:.4f}',
                'weighted_anomaly_score': '{:.4f}', 'rule_score': '{:.2f}', 'rule_norm': '{:.4f}',
                'weighted_rule_score': '{:.4f}'
            }), use_container_width=True)
        else:
            st.info("No alerts to display for the Top 10 table.")

        st.markdown("---")
        st.header("📈 Investigation Budget & Model Comparison")
        st.write("Performance at different investigation budget levels. This table updates as you change any control.")
        percents_to_show = [1, 2, 5, 10]
        col1_eval, col2_eval = st.columns(2, gap="large")
        with col1_eval:
            st.subheader("Model Score Ranking")
            st.dataframe(compute_budget_metrics(model_ranked_df_for_comparison, percents_to_show, total_tp).style.format({'Precision (%)': '{:.2f}', 'Coverage of Total TPs (%)': '{:.2f}'}), use_container_width=True)
        with col2_eval:
            st.subheader("Hybrid Score Reranking")
            st.dataframe(compute_budget_metrics(hybrid_ranked_df, percents_to_show, total_tp).style.format({'Precision (%)': '{:.2f}', 'Coverage of Total TPs (%)': '{:.2f}'}), use_container_width=True)
        
        st.markdown("---")
        st.header("🏆 Top 10 Users by True Positive (TP) Count")
        st.write("Users with the highest number of confirmed malicious alerts within the generated alerts.")
        if hybrid_ranked_df.empty:
            st.info("No alerts were generated to summarize user activity.")
        else:
            user_summary = hybrid_ranked_df.groupby('user').agg(
                alert_count=('user', 'size'),
                tp_count=('is_malicious', 'sum'),
            ).reset_index()
            user_summary = user_summary.sort_values(by=['tp_count', 'alert_count'], ascending=[False, False]).head(10)
            user_summary = user_summary.set_index('user')
            st.bar_chart(user_summary[['tp_count']], color=["#BF616A"])
            st.caption("Bars show the number of confirmed malicious alerts (TPs) for each user.")

        st.markdown("---")
        st.header("📋 Alert Triage Queue (Hybrid-Ranked)")
        if hybrid_ranked_df.empty:
            st.warning("No alerts generated by the model with the current settings.")
        else:
            max_alerts = len(hybrid_ranked_df)
            default_display = min(10, max_alerts)
            num_to_display = st.number_input(
                "Number of alerts to display", min_value=1, max_value=max_alerts, value=default_display, step=5)
            st.write(f"Displaying top **{num_to_display}** of **{len(hybrid_ranked_df)}** total alerts.")
            alerts_to_show = hybrid_ranked_df.head(num_to_display)
            
            for i in range(0, len(alerts_to_show), 2):
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    alert_row = alerts_to_show.iloc[i]
                    display_alert_card(alert_row, rank=alert_row.name + 1, rule_weights=rule_weights_dict)
                with col2:
                    if i + 1 < len(alerts_to_show):
                        alert_row_2 = alerts_to_show.iloc[i + 1]
                        display_alert_card(alert_row_2, rank=alert_row_2.name + 1, rule_weights=rule_weights_dict)

        # --- Exports ---
        st.markdown("---")
        st.subheader("Export Options")
        # CSV Export
        csv = hybrid_ranked_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Ranked Alerts as CSV", csv, "ranked_alerts.csv", "text/csv")

        # PDF Export (simple summary)
        def generate_pdf(df):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="SIEM-Style SOC Alert Report", ln=True, align='C')
            for i, row in df.head(10).iterrows():
                pdf.cell(200, 10, txt=f"Rank {i+1}: User {row['user']}, Score {row['hybrid_score']:.4f}, Malicious: {row['is_malicious']}", ln=True)
            return pdf

        pdf_output = generate_pdf(hybrid_ranked_df)
        pdf_str = pdf_output.output(dest='S')
        pdf_bytes = io.BytesIO(pdf_str.encode('latin-1'))
        pdf_bytes.seek(0)
        st.download_button("📄 Download Report as PDF", pdf_bytes, "alert_report.pdf", "application/pdf")

    with tab3:
        st.header("🔍 Advanced Investigations")
        if not hybrid_ranked_df.empty:
            # Hybrid score timeline as dot plot with custom colors
            st.subheader("Hybrid Scores Over Time")
            fig_hybrid = px.scatter(hybrid_ranked_df.sort_values('day'), x="day", y="hybrid_score", color="is_malicious",
                                    title="Hybrid Scores Timeline", labels={"is_malicious": "Malicious"},
                                    color_discrete_map={0: 'orange', 1: 'white'})
            fig_hybrid.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#2C2C2C", font_color="#E0E0E0")
            st.plotly_chart(fig_hybrid, use_container_width=True)

            # Anomaly score timeline as dot plot with custom colors
            st.subheader("Anomaly Scores Over Time")
            fig_anomaly = px.scatter(hybrid_ranked_df.sort_values('day'), x="day", y="anomaly_score", color="is_malicious",
                                     title="Anomaly Scores Timeline", labels={"is_malicious": "Malicious"},
                                     color_discrete_map={0: 'orange', 1: 'white'})
            fig_anomaly.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#2C2C2C", font_color="#E0E0E0")
            st.plotly_chart(fig_anomaly, use_container_width=True)

            # Bar chart for most triggered rules
            st.subheader("Most Triggered Rules")
            rule_columns = list(rule_weights_dict.keys())
            rule_triggers = hybrid_ranked_df[rule_columns].sum().reset_index()
            rule_triggers.columns = ['Rule', 'Trigger Count']
            rule_triggers = rule_triggers.sort_values('Trigger Count', ascending=False)
            fig_bar = px.bar(rule_triggers, x='Rule', y='Trigger Count', title="Rule Trigger Counts",
                             color='Trigger Count', color_continuous_scale="Reds")
            fig_bar.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#2C2C2C", font_color="#E0E0E0")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Run the pipeline to generate visualizations.")
