"""
Streamlit Anomaly Detection Dashboard — updated logic to match Capstone notebook
Save as: streamlit_anomaly_dashboard_updated.py
Run: streamlit run streamlit_anomaly_dashboard_updated.py

Changes in this edition:
- layout changed to `centered` to avoid full-screen expansion
- column-name cleanup to collapse duplicate segments like `name_name` -> `name`
- inserted requested rule_z / scaling / hybrid rerank block (HYBRID_ALPHA = 0.5)
- prevented dataframes from auto-expanding to full width by setting explicit width
- SHOW RULE SCORE AS SCALED Z (rule_z_scaled) IN TABLES / TOP10
- RENAMED DISPLAY COLUMNS TO FRIENDLY NAMES
- EXPANDED TOP10 AND PERCENTILE TABLES
"""

import io
import math
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc

# ----------------------------- Config ------------------------------------
RANDOM_STATE = 42

# NOTE: changed layout to centered so the app doesn't expand to full screen
st.set_page_config(layout="centered", page_title="Anomaly Detection (Dark Centered)")

# ----------------------------- Dark + Centered CSS -----------------------------
st.markdown("""
<style>
/* Center main container and limit width */
.main .block-container, .block-container {
  margin-left: auto !important;
  margin-right: auto !important;
  max-width: 1200px;
  padding-top: 12px;
  padding-bottom: 12px;
}

/* Dark background for the whole app */
body, .stApp, .block-container {
  background: linear-gradient(180deg, #020712 0%, #071122 100%) !important;
  color: #cfe6ff !important;
  text-align: center !important;
}

/* SOC header & KPI alignment */
.soc-header {
  background: linear-gradient(90deg, rgba(10,20,30,0.65), rgba(8,18,30,0.55));
  padding: 12px;
  border-radius: 8px;
  margin-bottom: 12px;
  display: flex;
  justify-content: center;
  align-items: center;
  text-align: center;
  border: 1px solid rgba(255,255,255,0.03);
}

/* KPI row: center the cards and allow wrap on small screens */
.kpi-row {
  display: flex;
  gap: 12px;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
  margin: 0 auto;
}

/* KPI card styling (center content inside) */
.kpi-card {
  background: linear-gradient(180deg, rgba(8,12,18,0.65), rgba(6,10,14,0.55));
  padding: 12px;
  border-radius: 8px;
  min-width: 200px;
  max-width: 260px;
  box-shadow: 0 4px 10px rgba(0,0,0,0.6);
  margin: 6px auto;
  text-align: center;
  border: 1px solid rgba(255,255,255,0.03);
}

/* subtle color accents */
.card-green { border-left: 6px solid rgba(34,197,94,0.9); }
.card-amber { border-left: 6px solid rgba(245,158,11,0.9); }
.card-red   { border-left: 6px solid rgba(239,68,68,0.9); }
.card-blue  { border-left: 6px solid rgba(59,130,246,0.9); }

/* make each KPI value centered and muted */
.kpi-title { font-size:13px; color:#9fb3ff; text-align:center; opacity:0.95; }
.kpi-value { font-size:20px; font-weight:700; margin-top:6px; color:#e6f4ff; }

/* DataFrame and table containers: dark theme */
div[data-testid="stDataFrameContainer"] table {
  background-color: rgba(10,14,18,0.75) !important;
  color: #dff0ff !important;
}
.stDataFrameContainer, .stTable, .element-container, .dataframe {
  margin-left: auto !important;
  margin-right: auto !important;
  display: block;
}

/* small table card */
.table-container {
  background: rgba(255,255,255,0.02);
  padding:8px;
  border-radius:6px;
  margin-left: auto;
  margin-right: auto;
  text-align: center;
  border: 1px solid rgba(255,255,255,0.02);
}

/* Sidebar tweaks: dark & slightly translucent */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(5,8,12,0.95), rgba(4,6,10,0.95)) !important;
  color: #cfe6ff !important;
  border-right: 1px solid rgba(255,255,255,0.02);
}

/* Buttons */
.stDownloadButton>button, button[kind="secondary"] {
  background-color: rgba(40,90,140,0.95);
  color: white;
  border: none;
}

/* monospace for small tables and top10 */
.monofont { font-family: monospace; font-size:13px; }

/* Responsive tweaks */
@media (max-width: 900px) {
  .kpi-card { min-width: 140px; max-width: 90%; }
  .kpi-row { gap: 8px; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Helper functions -----------------------------
def per_user_time_split_train_test(df, train_frac=0.5, min_rows=3):
    train_parts, test_parts = [], []
    for user, g in df.groupby('user'):
        g = g.sort_values('day')
        n = len(g)
        if n < min_rows:
            train_parts.append(g)
            continue
        i1 = int(math.floor(train_frac * n))
        if i1 < 1:
            i1 = 1
        if i1 >= n:
            i1 = n - 1
        train_parts.append(g.iloc[:i1])
        test_parts.append(g.iloc[i1:])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df  = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, test_df


def compute_rule_score_row(row, rule_weights):
    # deterministic rule scoring (keeps behaviour from notebook)
    score = 0.0
    if int(row.get('wikileaks_flag', 0)) == 1:
        score += rule_weights['wikileaks_flag']
    trio = {
        'offhour_usb_flag': int(row.get('offhour_usb_flag', 0)),
        'offhour_http_flag': int(row.get('offhour_http_flag', 0)),
        'offhour_logon_flag': int(row.get('offhour_logon_flag', 0))
    }
    # require usb flag present and at least two of the trio to add their weights
    if trio['offhour_usb_flag'] == 1 and sum(trio.values()) >= 2:
        score += sum(rule_weights[k] for k, v in trio.items() if v == 1)
    return score


def list_triggered_rules(row, rule_flags):
    triggered = [f for f in rule_flags if int(row.get(f, 0)) == 1]
    return ",".join(triggered) if triggered else "none"


def compute_norm_stats(train_ben_df, raw_like):
    if len(raw_like) == 0:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
    medians = train_ben_df[raw_like].median()
    user_med = train_ben_df.groupby('user')[raw_like].median()
    user_std = train_ben_df.groupby('user')[raw_like].std().replace(0, 1.0)
    global_med = train_ben_df[raw_like].median()
    global_std = train_ben_df[raw_like].std().replace(0, 1.0)
    return medians, user_med, user_std, global_med, global_std


def norm_df(input_df, features=None):
    if features is None:
        features = RAW_LIKE
    out = pd.DataFrame(index=input_df.index, columns=features, dtype=float)
    for feat in features:
        med_map = user_med[feat]
        std_map = user_std[feat]
        user_means = input_df['user'].map(med_map).fillna(global_med[feat])
        user_stds  = input_df['user'].map(std_map).fillna(global_std[feat])
        vals = input_df[feat].fillna(medians[feat])
        out[feat] = (vals - user_means) / user_stds
    return out.fillna(0.0)


def assemble_features(df_rows, raw_scaled_df, derived_keep):
    derived = df_rows[derived_keep].reset_index(drop=True)
    assembled = pd.concat([raw_scaled_df.reset_index(drop=True), derived.reset_index(drop=True)], axis=1)
    assembled = assembled.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return assembled


def compute_budgets_table(ranked_df, percent_budgets, total_tp):
    rows = []
    for p in percent_budgets:
        head = ranked_df.head(10)
        n_alerts = len(head)
        tp_in_budget = int(head['is_malicious'].sum())
        precision_pct = (tp_in_budget / n_alerts * 100.0) if n_alerts > 0 else 0.0
        coverage_pct = (tp_in_budget / total_tp * 100.0) if total_tp > 0 else float('nan')
        rows.append({'budget_type':'top_percent','budget':p,'precision_pct':precision_pct,'coverage_pct':coverage_pct,'tp_in_budget':tp_in_budget,'alerts_in_budget':n_alerts})
    return pd.DataFrame(rows)


# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.header("Controls & Pipeline settings")
    uploaded = st.file_uploader("Upload processed features CSV (user, day, is_malicious)", type=['csv'])
    run_button = st.button("Run pipeline")

    st.markdown("---")
    train_frac = st.slider("Train fraction (per-user)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    min_rows = st.number_input("Min rows for split (users with fewer go to train)", value=3, min_value=1, step=1)

    pct_threshold = st.slider("Percentile threshold for detection (train benign)", 90.0, 99.99, 99.0, step=0.1)
    # By default follow the notebook behaviour and use per-user thresholds (with global fallback)
    per_user_threshold = st.checkbox("Use per-user threshold (fallback to global)", value=True)

    st.markdown("---")
    st.header("Hybrid")
    hybrid_alpha = st.slider("Hybrid alpha (anomaly + alpha * rule_z)", 0.0, 1.0, 0.5, step=0.1)


    st.markdown("---")
    st.write("Rule weights")
    wikileaks_w = st.number_input("wikileaks_flag weight", value=1.0, step=0.1)
    offhour_usb_w = st.number_input("offhour_usb_flag weight", value=0.7, step=0.1)
    offhour_http_w = st.number_input("offhour_http_flag weight", value=0.3, step=0.1)
    offhour_logon_w = st.number_input("offhour_logon_flag weight", value=0.3, step=0.1)

# Centered title (dark)
st.markdown('<h1 style="text-align:center; color:#dff6ff; margin-bottom:6px;">User Behaviour Analytics — Insider Detection</h1>', unsafe_allow_html=True)

# ----------------------------- Run guard & load -----------------------------
if not run_button:
    st.info("Upload a CSV in the sidebar, adjust settings, and click 'Run pipeline'.")
    st.stop()

if uploaded is None:
    st.error("Please upload a CSV file (processed features).")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

required_cols = {'user','day','is_malicious'}
if not required_cols.issubset(set(raw_df.columns)):
    st.error(f"Uploaded CSV must contain columns: {required_cols}. Found: {raw_df.columns.tolist()}")
    st.stop()

raw_df['day'] = pd.to_datetime(raw_df['day'])

# ----------------------------- Features -----------------------------
EXPLICIT_DROP = ['user','day','is_malicious','scenario']
NO_NORMALIZE_COLUMNS = [
    # flags / categorical / label
    "is_weekend","wikileaks_flag","offhour_logon_flag","offhour_usb_flag","offhour_http_flag",
    "is_first_time_usb_user","rare_usb_user","scenario","is_malicious",'user','day'
    # z-scores (already normalized)
    "usb_connect_disconnect_zscore","http_offhour_requests_zscore",
    "session_duration_hours_zscore","total_http_requests_zscore",
    # rolling percentiles & stds (already relative)
    "logon_online_duration_rollpct_7d","logon_online_duration_rollpct_30d",
    "logon_offhour_count_rollpct_7d","logon_offhour_count_rollpct_30d",
    "logon_distinct_pcs_rollpct_7d","logon_distinct_pcs_rollpct_30d",
    "http_total_requests_rollpct_7d","http_total_requests_rollpct_30d",
    "http_unique_domains_rollpct_7d","http_unique_domains_rollpct_30d",
    "device_usb_count_rollpct_7d","device_usb_count_rollpct_30d",
    "device_total_usb_duration_rollpct_7d","device_total_usb_duration_rollpct_30d",
    "device_afterhours_usb_rollpct_7d","device_afterhours_usb_rollpct_30d",
    "logon_online_duration_rollstd_7d","logon_online_duration_rollstd_30d",
    "http_total_requests_rollstd_7d","http_total_requests_rollstd_30d",
    "device_usb_count_rollstd_7d","device_usb_count_rollstd_30d",
]
exclude_from_derived = {'is_malicious','scenario','user','day','is_weekend','wikileaks_flag','offhour_logon_flag','offhour_usb_flag','offhour_http_flag'}
DERIVED_KEEP = [c for c in raw_df.columns if c in NO_NORMALIZE_COLUMNS and c not in exclude_from_derived]
all_features = [c for c in raw_df.columns if c not in EXPLICIT_DROP]
RAW_LIKE = [c for c in all_features if c not in NO_NORMALIZE_COLUMNS]
st.sidebar.write(f"Detected features: raw_like={len(RAW_LIKE)}, derived_keep={len(DERIVED_KEEP)}")

# ----------------------------- Split -----------------------------
train_df, test_df = per_user_time_split_train_test(raw_df, train_frac=0.5, min_rows=min_rows)
st.write("### Data split (per-user time-respecting)")
col1, col2, col3 = st.columns([1,1,1])
col1.metric("Train rows", f"{len(train_df)}")
col2.metric("Test rows", f"{len(test_df)}")
col3.metric("Unique users (train/test)", f"{train_df['user'].nunique()}/{test_df['user'].nunique()}")
#col3.metric("Unique users (train/test)", f"{train_df['user'].nunique()}/{test_df['user'].nunique()}")

# ----------------------------- Train IF -----------------------------
train_ben_df = train_df[train_df['is_malicious']==0].reset_index(drop=True)
if train_ben_df.empty:
    st.error("No benign rows in training partition — cannot fit IsolationForest.")
    st.stop()

medians, user_med, user_std, global_med, global_std = compute_norm_stats(train_ben_df, RAW_LIKE)

if len(RAW_LIKE) > 0:
    X_train_scaled = norm_df(train_ben_df, RAW_LIKE)
    X_test_scaled = norm_df(test_df, RAW_LIKE)
else:
    X_train_scaled = pd.DataFrame(index=train_ben_df.index)
    X_test_scaled = pd.DataFrame(index=test_df.index)

assembled_train = assemble_features(train_ben_df, X_train_scaled, DERIVED_KEEP)
assembled_test  = assemble_features(test_df, X_test_scaled, DERIVED_KEEP)

st.write(f"Assembled train shape (benign rows used to train IF): {assembled_train.shape}")
st.write(f"Assembled test shape: {assembled_test.shape}")

clf = IsolationForest(n_estimators=200, max_samples='auto', contamination='auto', random_state=RANDOM_STATE)
with st.spinner("Training IsolationForest on benign training rows..."):
    clf.fit(assembled_train)

train_scores = -clf.decision_function(assembled_train)
test_scores  = -clf.decision_function(assembled_test)

train_ben_df = train_ben_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_ben_df['anomaly_score'] = train_scores
test_df['anomaly_score'] = test_scores

# ----------------------------- Thresholding -----------------------------
# compute global threshold from benign train scores
global_threshold = float(np.percentile(train_scores, pct_threshold))
if per_user_threshold:
    pupct = pct_threshold/100.0
    per_user_thresh = train_ben_df.groupby('user')['anomaly_score'].quantile(pupct)
    # fallback to global for users missing in train-benign
    test_df['anomaly_threshold'] = test_df['user'].map(per_user_thresh).fillna(global_threshold)
else:
    test_df['anomaly_threshold'] = global_threshold

# mark model-detected anomalies (anomaly-only detection)
test_df['is_model_anomaly'] = (test_df['anomaly_score'] >= test_df['anomaly_threshold']).astype(int)

# ----------------------------- Rule scoring -----------------------------
RULE_FLAGS = ['wikileaks_flag','offhour_usb_flag','offhour_http_flag','offhour_logon_flag']
RULE_WEIGHTS = {
    'wikileaks_flag': wikileaks_w,
    'offhour_usb_flag': offhour_usb_w,
    'offhour_http_flag': offhour_http_w,
    'offhour_logon_flag': offhour_logon_w
}

train_ben_df['rule_score'] = train_ben_df.apply(lambda r: compute_rule_score_row(r, RULE_WEIGHTS), axis=1)
# ensure rule_score exists even if some cols missing
if any(c not in test_df.columns for c in ['wikileaks_flag','offhour_usb_flag','offhour_http_flag','offhour_logon_flag']):
    # create cols with zeros if missing
    for c in ['wikileaks_flag','offhour_usb_flag','offhour_http_flag','offhour_logon_flag']:
        if c not in test_df.columns:
            test_df[c] = 0

test_df['rule_score'] = test_df.apply(lambda r: compute_rule_score_row(r, RULE_WEIGHTS), axis=1)



# ----------------------------- rule_z scaling + hybrid (inserted block) -----------------------------
rmean = train_ben_df['rule_score'].mean()
rstd = train_ben_df['rule_score'].std() if train_ben_df['rule_score'].std() > 0 else 1.0
test_df['rule_z'] = (test_df['rule_score'] - rmean) / rstd
rule_z_std = test_df['rule_z'].replace([np.inf,-np.inf], np.nan).dropna().std()
anomaly_std = train_ben_df['anomaly_score'].std()
scale = anomaly_std / (rule_z_std if rule_z_std>0 else 1.0)
test_df['rule_z_scaled'] = test_df['rule_z'] * scale

# ----------------- Re-rank only the model-detected anomalies using hybrid score -----------------
# Note: we DO NOT use hybrid for detection; detection is anomaly-only thresholding above.
anomalies_all = test_df[test_df['is_model_anomaly'] == 1].copy()
HYBRID_ALPHA = hybrid_alpha   # use user-selected slider value
anomalies_all['hybrid_score'] = anomalies_all['anomaly_score'] + HYBRID_ALPHA * anomalies_all['rule_z_scaled']

# final re-ranked alerts (descending hybrid_score)
anomalies_all = anomalies_all.sort_values('hybrid_score', ascending=False).reset_index(drop=True)

# ----------------------------- Budgets parsing -----------------------------
percent_budgets = [0.01,0.02,0.05,0.10,0.15]

total_tp = int(test_df['is_malicious'].sum())
model_table = compute_budgets_table(test_df[test_df['is_model_anomaly']==1], percent_budgets, total_tp)
hybrid_table = compute_budgets_table(anomalies_all, percent_budgets, total_tp)

# ----------------------------- Friendly display name mapping -----------------------------
display_name_map = {
    'user':'User', 'day':'Day', 'hybrid_score':'Hybrid Score', 'anomaly_score':'Anomaly Score',
     'rule_z':'Rule Z', 'rule_z_scaled':'Rule Z (scaled)',
    'is_malicious':'Is Malicious', 'triggered_rules':'Triggered Rules'
}

# small helper to rename for display
def rename_for_display(df):
    df = df.copy()
    # convert day to ISO string for nicer display
    if 'day' in df.columns:
        try:
            df['day'] = pd.to_datetime(df['day']).dt.strftime('%Y-%m-%d')
        except Exception:
            pass
    return df.rename(columns=display_name_map)

# ----------------------------- KPI cards -----------------------------
st.markdown('<div class="soc-header"><h2 style="margin:4px 0 0 0; color:#dff6ff;">SOC Anomaly Detection — Overview</h2></div>', unsafe_allow_html=True)

model_tp_in_alerts = int(test_df[test_df['is_model_anomaly']==1]['is_malicious'].sum()) if len(test_df[test_df['is_model_anomaly']==1])>0 else 0
model_tp_pct_of_total = (model_tp_in_alerts / total_tp * 100.0) if total_tp>0 else 0.0

kpi_html = f"""
<div class="kpi-row">
  <div class="kpi-card card-blue">
    <div class="kpi-title">Train rows</div>
    <div class="kpi-value">{len(train_df)}</div>
  </div>
  <div class="kpi-card card-amber">
    <div class="kpi-title">Test rows</div>
    <div class="kpi-value">{len(test_df)}</div>
  </div>
  <div class="kpi-card card-blue">
    <div class="kpi-title">Total Users</div>
    <div class="kpi-value">{train_df['user'].nunique()}</div>
  </div>
  <div class="kpi-card card-green">
    <div class="kpi-title">Total true positives (test)</div>
    <div class="kpi-value">{total_tp}</div>
  </div>
</div>
<br/>
<div class="kpi-row">
  <div class="kpi-card card-red">
    <div class="kpi-title">Model-detected alerts</div>
    <div class="kpi-value">{len(test_df[test_df['is_model_anomaly']==1])}</div>
  </div>
  <div class="kpi-card card-amber">
    <div class="kpi-title">Re-ranked (hybrid) alerts</div>
    <div class="kpi-value">{len(anomalies_all)}</div>
  </div>
  <div class="kpi-card card-green">
    <div class="kpi-title">TP% in alerts (of total TP)</div>
    <div class="kpi-value">{model_tp_pct_of_total:.2f}%</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)
st.markdown("---")

# ----------------------------- Top 10 hybrid alerts -----------------------------
st.subheader("Top 10 anomalous alerts (by hybrid score)")
if len(anomalies_all) == 0:
    st.write("No model-detected anomalies found at the selected threshold.")
else:
    anomalies_all['triggered_rules'] = anomalies_all.apply(lambda r: list_triggered_rules(r, RULE_FLAGS), axis=1)
    # Expand the top10 to show more useful columns and use the scaled z rule score
    display_cols = ['user','day','hybrid_score','anomaly_score','rule_z_scaled','triggered_rules','is_malicious']
    # keep only cols that exist (robust)
    display_cols = [c for c in display_cols if c in anomalies_all.columns]
    top10_rows = anomalies_all.head(10)[display_cols]
    # rename for human-friendly display
    top10_rows_disp = rename_for_display(top10_rows)

    st.markdown('<div class="table-container monofont">', unsafe_allow_html=True)
    st.dataframe(top10_rows_disp.reset_index(drop=True).style.set_properties(**{"font-family":"monospace"}), width=1100)
    st.markdown('</div>', unsafe_allow_html=True)
    csv_buf = io.StringIO()
    top10_rows.to_csv(csv_buf, index=False)
    st.download_button('Download top10 rows CSV', csv_buf.getvalue().encode('utf-8'), file_name='top10_hybrid_rows.csv')

st.markdown("---")

# ----------------------------- Percentile tables (1,2,5,10,15%) -----------------------------
percentiles = [1,2,5,10,15]
percent_fractions = [p/100.0 for p in percentiles]


def compute_percentile_table(ranked_df, percent_fractions, total_tp):
    rows = []
    for p, frac in zip(percentiles, percent_fractions):
        k = max(1, int(math.floor(frac * len(ranked_df))))
        head = ranked_df.head(k)
        n_alerts = len(head)
        tp_in_budget = int(head['is_malicious'].sum())
        tp_pct_of_total = (tp_in_budget / total_tp * 100.0) if total_tp>0 else float('nan')
        tp_pct_in_alert = (tp_in_budget / n_alerts * 100.0) if n_alerts>0 else float('nan')
        rows.append({'Percentile (%)': p, 'Top N': k, 'Alerts in budget': n_alerts,
                     'TP in budget': tp_in_budget, 'TP% in alerts': tp_pct_in_alert, 'TP% of total': tp_pct_of_total})
    return pd.DataFrame(rows)

model_percent_table = compute_percentile_table(test_df[test_df['is_model_anomaly']==1], percent_fractions, total_tp)
hybrid_percent_table = compute_percentile_table(anomalies_all, percent_fractions, total_tp)

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Model-only TP% at percentiles")
    st.dataframe(model_percent_table.style.format({"Percentile (%)":"{:.0f}", "TP% of total":"{:.2f}", "TP% in alerts":"{:.2f}"}), width=1100)
    st.download_button("Download model percentiles CSV", model_percent_table.to_csv(index=False).encode('utf-8'), file_name='model_percentiles.csv')
with col_right:
    st.subheader("Hybrid TP% at percentiles")
    st.dataframe(hybrid_percent_table.style.format({"Percentile (%)":"{:.0f}", "TP% of total":"{:.2f}", "TP% in alerts":"{:.2f}"}), width=1100)
    st.download_button("Download hybrid percentiles CSV", hybrid_percent_table.to_csv(index=False).encode('utf-8'), file_name='hybrid_percentiles.csv')

st.markdown("---")

# End of file
