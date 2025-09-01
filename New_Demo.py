import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import IsolationForest

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="SOC Dashboard - Isolation Forest",
    page_icon="🛡️"
)

# --- Default Values (from notebook) ---
RANDOM_STATE = 42
DEFAULT_PCT_THRESHOLD = 99.0
DEFAULT_W1 = 0.70
DEFAULT_RULE_WEIGHTS = {
    "wikileaks_flag": 1.0,
    "offhour_usb_flag": 0.7,
    "offhour_http_flag": 0.3,
    "offhour_logon_flag": 0.3,
}

# --- Caching Functions to Prevent Re-computation ---

@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the uploaded CSV file. Cached to prevent re-reading."""
    df = pd.read_csv(uploaded_file)
    required_cols = {"user", "day", "is_malicious"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain the columns: {required_cols}")
        return None
    df["user"] = df["user"].astype(str)
    df["day"] = pd.to_datetime(df["day"]).dt.floor("D")
    return df

@st.cache_data
def run_pipeline(df, pct_threshold, w1, rule_weights, use_per_user_threshold):
    """
    Runs the entire data processing and modeling pipeline.
    This function is cached, so it only re-runs when a control parameter in the sidebar changes.
    """
    per_user_z_cols = [c for c in df.columns if ('_user_z' in c)]
    if not per_user_z_cols:
        st.error("No per-user normalized feature columns found (expected names with '_user_z').")
        return None, None, None

    train_df, test_df = time_split_train_test(df)
    train_ben = train_df[train_df["is_malicious"] == 0].reset_index(drop=True)
    if train_ben.empty:
        st.error("No benign rows found in the training data split. Cannot train the model.")
        return None, None, None

    X_train_if = train_ben[per_user_z_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test_if = test_df[per_user_z_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    clf = IsolationForest(n_estimators=500, contamination="auto", max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_if)

    train_scores = -clf.decision_function(X_train_if)
    test_scores = -clf.decision_function(X_test_if)
    train_ben["anomaly_score"] = train_scores
    test_df["anomaly_score"] = test_scores

    # --- THRESHOLDING LOGIC ---
    global_threshold = np.percentile(train_scores, pct_threshold)
    if use_per_user_threshold:
        per_user_thr = train_ben.groupby("user")["anomaly_score"].quantile(pct_threshold / 100.0)
        test_df["anomaly_threshold"] = test_df["user"].map(per_user_thr).fillna(global_threshold)
    else:
        test_df["anomaly_threshold"] = global_threshold

    test_df["is_model_anomaly"] = (test_df["anomaly_score"] >= test_df["anomaly_threshold"]).astype(int)

    # --- RULE & HYBRID SCORING ---
    test_df["rule_score"] = test_df.apply(lambda r: compute_rule_score_row(r, rule_weights), axis=1)
    anomalies_all = test_df[test_df["is_model_anomaly"] == 1].copy()

    max_possible_rule = sum(w for w in rule_weights.values() if w > 0.0) or 1.0
    anomalies_all["rule_norm"] = (anomalies_all["rule_score"] / max_possible_rule).clip(0, 1)

    score_min, score_max = train_scores.min(), train_scores.max()
    den = max(1e-12, (score_max - score_min))
    anomalies_all["anomaly_norm"] = ((anomalies_all["anomaly_score"] - score_min) / den).clip(0, 1)

    anomalies_all["hybrid_score"] = (w1 * anomalies_all["anomaly_norm"]) + ((1.0 - w1) * anomalies_all["rule_norm"])
    anomalies_all = anomalies_all.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

    total_tp = int(test_df["is_malicious"].sum())
    model_ranked_df = test_df[test_df["is_model_anomaly"] == 1].sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    return model_ranked_df, anomalies_all, total_tp

# --- Helper Functions (No Caching Needed) ---
def time_split_train_test(df):
    ID_COLS     = ['user','day']
    df['day'] = pd.to_datetime(df['day'])
    
    # Find all unique years present in the data, sorted chronologically
    available_years = sorted(df['day'].dt.year.unique())
    split_year = None

    # 💡 Automatically find a year that can be split
    for year in available_years:
        split_date = pd.Timestamp(f"{year}-06-30")
        
        # Check if there is data both before/on the split date AND after it
        has_train_data = (df['day'] <= split_date).any()
        has_test_data = (df['day'] > split_date).any()
        
        if has_train_data and has_test_data:
            split_year = year
            break # Use the first suitable year found

    # If no such year is found, raise an error
    if split_year is None:
        raise ValueError("Could not find a year with data both before/on June 30 and after July 1.")

    print(f"✅ Found suitable split year: {split_year}. Training up to {split_year}-06-30.")

    # --- Perform the split ---
    final_split_date = pd.Timestamp(f"{split_year}-06-30")
    
    # Training keys are from rows on or before the split date
    train_mask = df['day'] <= final_split_date
    keys_train = df.loc[train_mask, ID_COLS].copy().reset_index(drop=True)
    
    # Testing keys are from rows after the split date
    test_mask = df['day'] > final_split_date
    keys_test = df.loc[test_mask, ID_COLS].copy().reset_index(drop=True)

    train_df   = df.merge(keys_train, on=ID_COLS, how='inner')
    test_df    = df.merge(keys_test,  on=ID_COLS, how='inner')
    
    return train_df, test_df

def compute_rule_score_row(row, rule_weights):
    """Calculates the rule score."""
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
    """Computes precision and coverage tables."""
    rows = []
    for p in percents:
        frac = p / 100.0
        k = max(1, int(math.floor(frac * len(ranked_df))))
        head = ranked_df.head(k)
        n_alerts = len(head)
        tp_in_budget = int(head["is_malicious"].sum())
        precision = (tp_in_budget / n_alerts * 100.0) if n_alerts > 0 else 0.0
        coverage = (tp_in_budget / total_tp * 100.0) if total_tp > 0 else 0.0
        rows.append({
            "Budget Top %": p, "Top N Alerts": k, "TPs in Budget": tp_in_budget,
            "Precision (%)": precision, "Coverage of Total TPs (%)": coverage,
        })
    return pd.DataFrame(rows)

def display_alert_card(alert_data, rank, rule_weights, w1_weight):
    """Displays a single alert in a formatted card with score breakdown."""
    is_malicious = alert_data['is_malicious'] == 1
    triggered_rules = ", ".join([flag for flag, weight in rule_weights.items() if alert_data.get(flag, 0) == 1]) or "None"

    with st.container(border=True):
        c1, c2, c3 = st.columns([0.25, 0.5, 0.25])
        with c1:
            st.markdown(f"**🏅 Rank #{rank}**")
            st.markdown(f"**User:** `{alert_data['user']}`")
            st.markdown(f"**Date:** {alert_data['day'].strftime('%Y-%m-%d')}")
        with c2:
            st.metric(label="Hybrid Score", value=f"{alert_data['hybrid_score']:.4f}")
            st.caption(f"""
                Breakdown: ({alert_data['weighted_anomaly_score']:.3f}) + ({alert_data['weighted_rule_score']:.3f})
                | Scaled Anomaly: {alert_data['anomaly_norm']:.3f}
                | Scaled Rule: {alert_data['rule_norm']:.3f}
            """)
            st.markdown(f"**Triggered Rules:** `{triggered_rules}`")
        with c3:
            if is_malicious:
                st.error("🔥 **Confirmed Malicious**", icon="🔥")
            else:
                st.success("✅ **Benign**", icon="✅")


# --- UI Layout ---
st.markdown("""<style>.stMetric{border:1px solid #4A5568;border-radius:8px;padding:15px;background-color:#2D3748;}.stDataFrame{border:1px solid #4A5568;}</style>""", unsafe_allow_html=True)
st.title("🛡️ SOC Alert Triage Dashboard")
st.markdown("##### Anomaly Detection with Isolation Forest & Rule-Based Reranking")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️Controls")
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

# --- Main Application Logic ---
if not run_button:
    st.info("Upload a features CSV, adjust controls in the sidebar, and click **Run Pipeline**.")
    st.stop()
if uploaded_file is None:
    st.error("❌ Please upload the per-user features CSV file to begin.")
    st.stop()

df_loaded = load_data(uploaded_file)
if df_loaded is None: st.stop()

rule_weights_dict = {
    "wikileaks_flag": wikileaks_w, "offhour_usb_flag": offhour_usb_w,
    "offhour_http_flag": offhour_http_w, "offhour_logon_flag": offhour_logon_w,
}

model_ranked_df, hybrid_ranked_df, total_tp = run_pipeline(
    df_loaded, pct_threshold, w1, rule_weights_dict, use_per_user_threshold
)
if model_ranked_df is None: st.stop()

# --- Display Results ---
st.header("📊 Performance Metrics")
model_alerts_count = len(model_ranked_df)
model_tp = int(model_ranked_df["is_malicious"].sum())
model_coverage = (model_tp / total_tp * 100.0) if total_tp > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Total Malicious Events (in Test Set)", f"{total_tp:,}")
c2.metric("Total Model-Detected Alerts", f"{model_alerts_count:,}")
c3.metric("TPs Found by Model (Coverage)", f"{model_coverage:.2f}%")

st.markdown("---")
st.header("🚨 Top 10 Reranked Alerts")
st.write("The highest priority alerts after applying the hybrid reranking model, with a full score breakdown.")

hybrid_ranked_df['weighted_anomaly_score'] = w1 * hybrid_ranked_df['anomaly_norm']
hybrid_ranked_df['weighted_rule_score'] = (1 - w1) * hybrid_ranked_df['rule_norm']

display_cols_top10 = [
    "user", "day", "hybrid_score", "anomaly_score", "anomaly_norm", "weighted_anomaly_score",
    "rule_score", "rule_norm", "weighted_rule_score", "is_malicious"
]
st.dataframe(hybrid_ranked_df[display_cols_top10].head(10).style.format(formatter={
    'hybrid_score': '{:.4f}', 'anomaly_score': '{:.4f}', 'anomaly_norm': '{:.4f}',
    'weighted_anomaly_score': '{:.4f}', 'rule_score': '{:.2f}', 'rule_norm': '{:.4f}',
    'weighted_rule_score': '{:.4f}'
}), use_container_width=True)


st.markdown("---")
st.header("📈 Investigation Budget & Model Comparison")
st.write("Performance at different investigation budget levels, comparing the model-only ranking to the hybrid reranking.")

percents_to_show = [1, 2, 5, 10]
col1_eval, col2_eval = st.columns(2, gap="large")
with col1_eval:
    st.subheader("Model Score Ranking")
    st.dataframe(compute_budget_metrics(model_ranked_df, percents_to_show, total_tp).style.format({'Precision (%)': '{:.2f}', 'Coverage of Total TPs (%)': '{:.2f}'}), use_container_width=True)

with col2_eval:
    st.subheader("Hybrid Score Reranking")
    st.dataframe(compute_budget_metrics(hybrid_ranked_df, percents_to_show, total_tp).style.format({'Precision (%)': '{:.2f}', 'Coverage of Total TPs (%)': '{:.2f}'}), use_container_width=True)

st.markdown("---")
st.header("🏆 Top 10 Users by True Positive (TP) Count")
st.write("Users with the highest number of confirmed malicious alerts.")

if hybrid_ranked_df.empty:
    st.info("No alerts were generated to summarize.")
else:
    user_summary = hybrid_ranked_df.groupby('user').agg(
        alert_count=('user', 'size'),
        tp_count=('is_malicious', 'sum'),
    ).reset_index()
    # Sort by TP count first, then by total alert count as a tie-breaker
    user_summary = user_summary.sort_values(by=['tp_count', 'alert_count'], ascending=[False, False]).head(10)
    user_summary = user_summary.set_index('user')
    
    st.bar_chart(user_summary[['tp_count']], color=["#BF616A"])
    st.caption("Bars show the number of confirmed malicious alerts (TPs) for each user.")

st.markdown("---")
st.header("📋 Alert Triage Queue (Hybrid-Ranked)")

if hybrid_ranked_df.empty:
    st.warning("No alerts generated by the model with the current settings.")
else:
    
    # Control for number of alerts to display, without the user filter
    max_alerts = len(hybrid_ranked_df)
    default_display = min(10, max_alerts if max_alerts > 0 else 10)
    num_to_display = st.number_input(
        "Number of alerts to display", 
        min_value=1, 
        max_value=max_alerts if max_alerts > 0 else 1, 
        value=default_display, 
        step=5
    )

    st.write(f"Displaying top **{num_to_display}** of **{len(hybrid_ranked_df)}** total alerts.")
    alerts_to_show = hybrid_ranked_df.head(num_to_display)
    
    # Display alerts in a 2-column layout
    for i in range(0, len(alerts_to_show), 2):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            # Display the first alert in the pair
            alert_row = alerts_to_show.iloc[i]
            display_alert_card(alert_row, rank=alert_row.name + 1, rule_weights=rule_weights_dict, w1_weight=w1)
        with col2:
            # Display the second alert if it exists
            if i + 1 < len(alerts_to_show):
                alert_row_2 = alerts_to_show.iloc[i + 1]
                display_alert_card(alert_row_2, rank=alert_row_2.name + 1, rule_weights=rule_weights_dict, w1_weight=w1)
