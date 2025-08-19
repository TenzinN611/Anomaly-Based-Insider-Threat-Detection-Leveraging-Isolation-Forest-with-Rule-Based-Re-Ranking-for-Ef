#!/usr/bin/env python3
# coding: utf-8

import cudf
import pandas as pd
import numpy as np
import os
import io
from tqdm import tqdm
import time
import re
import gc
from collections import defaultdict, Counter

# -----------------------
# Config
# -----------------------
WORK_START = 8   # 8:00 AM
WORK_END = 18    # 6:00 PM

SUSPICIOUS_KEYWORDS = ["free-keylogger", "keylog", "computer-monitoring", "monitoring", "wikileaks", "spy"]
SUSPICIOUS_URL_RE = re.compile('|'.join(re.escape(k.lower()) for k in SUSPICIOUS_KEYWORDS))

# -----------------------
# Chunked CSV reader (works for cudf and pandas)
# -----------------------
def read_csv_chunks(file_path, chunk_bytes=1024**3, **kwargs):
    """
    Read large CSV in chunks, returning pandas or cudf frames depending on engine param.
    Usage: read_csv_chunks(path, chunk_bytes, engine='cudf'|'pandas', **pd.read_csv kwargs)
    """
    engine = kwargs.pop('engine', 'cudf')
    try:
        header_df = cudf.read_csv(file_path, nrows=0) if engine == 'cudf' else pd.read_csv(file_path, nrows=0)
    except (FileNotFoundError, StopIteration):
        return
    cols = list(header_df.columns)
    kwargs.update({'names': cols, 'header': 0})
    leftover = b''
    with open(file_path, 'rb') as f:
        f.readline()  # Skip header
        while True:
            data = f.read(chunk_bytes)
            if not data and not leftover:
                break
            buf = leftover + data
            nl = buf.rfind(b"\n")
            if nl >= 0:
                part, leftover = buf[:nl+1], buf[nl+1:]
            else:
                part, leftover = b"", buf
            if part:
                buf_io = io.BytesIO(part)
                kwargs['header'] = None
                yield cudf.read_csv(buf_io, **kwargs) if engine == 'cudf' else pd.read_csv(buf_io, **kwargs)
    if leftover.strip():
        buf_io = io.BytesIO(leftover)
        yield cudf.read_csv(buf_io, **kwargs) if engine == 'cudf' else pd.read_csv(buf_io, **kwargs)


# -----------------------
# Helpers
# -----------------------
def is_offhour(ts_series):
    if isinstance(ts_series, pd.Series):
        return (ts_series.dt.hour < WORK_START) | (ts_series.dt.hour >= WORK_END)
    else:
        return (ts_series.dt.hour < WORK_START) | (ts_series.dt.hour >= WORK_END)


def extract_domain(url_series):
    s = url_series.fillna('').astype('str')
    # use pandas string ops for robustness
    if isinstance(s, pd.Series):
        domains = s.str.extract(r'^[^:/]+://([^/]+)')[0].fillna('')
        missing = domains == ''
        if missing.any():
            parts = s[missing].str.split('/', 2)
            domains.loc[missing] = parts.str.get(1).fillna('')
        return domains.fillna('')
    else:
        s_pd = s.to_pandas()
        domains = s_pd.str.extract(r'^[^:/]+://([^/]+)')[0].fillna('')
        missing = domains == ''
        if missing.any():
            parts = s_pd[missing].str.split('/', 2)
            domains.loc[missing] = parts.str.get(1).fillna('')
        return cudf.from_pandas(domains)


# -----------------------
# Process Logon (chunked, cudf)
# -----------------------
def process_logon(logon_path, chunk_bytes=1024**3):
    parts = []
    raw_parts = []
    if not os.path.exists(logon_path):
        return cudf.DataFrame(), pd.DataFrame()
    for chunk in tqdm(read_csv_chunks(logon_path, chunk_bytes, engine='cudf'), desc='Logon'):
        # ensure date
        chunk['date'] = cudf.from_pandas(pd.to_datetime(chunk['date'].to_pandas(), errors='coerce'))
        chunk = chunk.dropna(subset=['date'])
        chunk['day'] = chunk['date'].dt.floor('D')
        chunk['user'] = chunk['user'].astype('str')
        # counts
        # ensure no-NaN strings before contains (cudf doesn't support na= in contains)
        act_series = chunk['activity'].fillna('').str.lower()
        logon_mask = act_series.str.contains('logon')
        logoff_mask = act_series.str.contains('logoff')
        logon_count = chunk[logon_mask].groupby(['user','day']).size().reset_index(name='logon_count')
        logoff_count = chunk[logoff_mask].groupby(['user','day']).size().reset_index(name='logoff_count')
        unique_pcs = chunk.groupby(['user','day'])['pc'].nunique().reset_index(name='distinct_pcs')
        offhour_mask = is_offhour(chunk['date']) & act_series.str.contains('logon|logoff')
        offhour_events = chunk[offhour_mask]
        offhour_count = offhour_events.groupby(['user','day']).size().reset_index(name='offhour_logonlogoff_count')

        # weekend flag
        chunk_pd = chunk.to_pandas()
        weekend_flag = (chunk_pd['date'].dt.weekday >= 5).groupby([chunk_pd['user'], chunk_pd['day']]).any().reset_index()
        weekend_flag.columns = ['user','day','is_weekend']
        weekend_flag = cudf.from_pandas(weekend_flag)

        agg = logon_count.merge(logoff_count, on=['user','day'], how='outer')
        agg = agg.merge(unique_pcs, on=['user','day'], how='outer')
        agg = agg.merge(offhour_count, on=['user','day'], how='outer')
        agg = agg.merge(weekend_flag, on=['user','day'], how='outer')
        agg = agg.fillna(0)
        parts.append(agg)
        raw_parts.append(chunk[['user','day','date','pc','activity']].to_pandas())
        del chunk, chunk_pd, logon_count, logoff_count, unique_pcs, offhour_count, weekend_flag, agg
        gc.collect()
    if not parts:
        return cudf.DataFrame(), pd.DataFrame()
    agg_all = cudf.concat(parts).groupby(['user','day']).sum().reset_index()
    raw_all = pd.concat(raw_parts).sort_values(['user','day','date']).reset_index(drop=True)

    # compute total online duration (sum of paired logon->logoff durations) per user-day using raw_all
    durations = []
    for (u,d), g in raw_all.groupby(['user','day']):
        g2 = g.sort_values('date')
        total_seconds = 0.0
        stack = []
        for _, row in g2.iterrows():
            act = str(row['activity']).lower()
            ts = row['date']
            if 'logon' in act:
                stack.append(ts)
            elif 'logoff' in act:
                if stack:
                    start = stack.pop(0)
                    total_seconds += (ts - start).total_seconds()
        # any unmatched logon - ignore or cap at 0
        total_hours = total_seconds / 3600.0
        durations.append({'user':u, 'day':pd.Timestamp(d), 'online_duration_hours': total_hours})

    dur_df = pd.DataFrame(durations) if durations else pd.DataFrame(columns=['user','day','online_duration_hours'])
    agg_pd = agg_all.to_pandas()
    agg_pd = agg_pd.merge(dur_df, on=['user','day'], how='left').fillna({'online_duration_hours':0})
    agg_pd['is_weekend'] = agg_pd.get('is_weekend',0).astype(int)

    # ensure numeric types
    for c in ['logon_count','logoff_count','distinct_pcs','offhour_logonlogoff_count']:
        if c in agg_pd.columns:
            agg_pd[c] = agg_pd[c].fillna(0).astype(int)
    agg_pd['online_duration_hours'] = agg_pd['online_duration_hours'].astype(float)

    return cudf.from_pandas(agg_pd), agg_pd


# -----------------------
# Process HTTP (chunked, cudf)
# -----------------------
def process_http(http_path, chunk_bytes=1024**3):
    parts = []
    raw_parts = []
    if not os.path.exists(http_path):
        return cudf.DataFrame(), pd.DataFrame()
    for chunk in tqdm(read_csv_chunks(http_path, chunk_bytes, engine='cudf'), desc='HTTP'):
        chunk['date'] = cudf.from_pandas(pd.to_datetime(chunk['date'].to_pandas(), errors='coerce'))
        chunk = chunk.dropna(subset=['date'])
        chunk['day'] = chunk['date'].dt.floor('D')
        chunk['user'] = chunk['user'].astype('str')
        # domain extraction via pandas roundtrip
        chunk_pd = chunk.to_pandas()
        chunk_pd['domain'] = extract_domain(chunk_pd['url'])
        chunk_pd['is_wikileaks'] = chunk_pd['domain'].fillna('').str.lower().str.contains('wikileaks').astype(int)
        # counts
        total_req = chunk_pd.groupby(['user','day']).size().reset_index(name='total_http_requests')
        unique_dom = chunk_pd.groupby(['user','day'])['domain'].nunique().reset_index(name='unique_domains')
        # session duration hours (max-min)
        session_span = chunk_pd.groupby(['user','day'])['date'].agg(['min','max']).reset_index()
        session_span['session_duration_hours'] = (session_span['max'] - session_span['min']).dt.total_seconds() / 3600.0
        session_span = session_span[['user','day','session_duration_hours']]
        wik = chunk_pd.groupby(['user','day'])['is_wikileaks'].max().reset_index(name='wikileaks_flag')

        agg = total_req.merge(unique_dom, on=['user','day'], how='outer')
        agg = agg.merge(session_span, on=['user','day'], how='outer')
        agg = agg.merge(wik, on=['user','day'], how='outer')
        agg = agg.fillna({'session_duration_hours':0})
        # compute avg requests per session
        agg['avg_requests_per_session'] = agg.apply(lambda r: (r['total_http_requests'] / (r['session_duration_hours'] if r['session_duration_hours']>0 else 1.0)), axis=1)
        parts.append(cudf.from_pandas(agg))
        raw_parts.append(chunk_pd[['user','day','date','domain']])
        del chunk, chunk_pd, total_req, unique_dom, session_span, wik, agg
        gc.collect()
    if not parts:
        return cudf.DataFrame(), pd.DataFrame()
    http_all = cudf.concat(parts).groupby(['user','day']).agg({'total_http_requests':'sum', 'unique_domains':'sum', 'session_duration_hours':'sum', 'avg_requests_per_session':'mean', 'wikileaks_flag':'max'}).reset_index()
    raw_all = pd.concat(raw_parts).sort_values(['user','day','date']).reset_index(drop=True)
    return http_all, raw_all


# -----------------------
# Process Device (USB) (chunked, cudf)
# -----------------------
def process_device(device_path, chunk_bytes=1024**3):
    parts = []
    raw_parts = []
    if not os.path.exists(device_path):
        return cudf.DataFrame(), pd.DataFrame()
    for chunk in tqdm(read_csv_chunks(device_path, chunk_bytes, engine='cudf'), desc='Device'):
        chunk['date'] = cudf.from_pandas(pd.to_datetime(chunk['date'].to_pandas(), errors='coerce'))
        chunk = chunk.dropna(subset=['date'])
        chunk['day'] = chunk['date'].dt.floor('D')
        chunk['user'] = chunk['user'].astype('str')
        chunk_pd = chunk.to_pandas()
        # counts of connect/disconnect
        conn_mask = chunk_pd['activity'].fillna('').str.lower().str.contains('connect')
        disc_mask = chunk_pd['activity'].fillna('').str.lower().str.contains('disconnect')

        usb_count = chunk_pd[conn_mask | disc_mask].groupby(['user','day']).size().reset_index(name='usb_connect_disconnect_count')
        # after-hours usb connects
        offhour_conn = chunk_pd[conn_mask & (chunk_pd['date'].dt.hour < WORK_START) | (chunk_pd['date'].dt.hour >= WORK_END)]
        offhour_conn = offhour_conn.groupby(['user','day']).size().reset_index(name='afterhours_usb_connects')
        # collect raw for session pairing
        parts.append(cudf.from_pandas(usb_count))
        raw_parts.append(chunk_pd[['user','day','date','activity']])
        del chunk, chunk_pd, usb_count, offhour_conn
        gc.collect()
    if not parts:
        return cudf.DataFrame(), pd.DataFrame()
    dev_agg = cudf.concat(parts).groupby(['user','day']).sum().reset_index()
    raw_all = pd.concat(raw_parts).sort_values(['user','day','date']).reset_index(drop=True)

    # compute session durations by pairing connect->disconnect per user-day
    durations = []
    for (u,d), g in raw_all.groupby(['user','day']):
        g2 = g.sort_values('date')
        total_seconds = 0.0
        session_counts = 0
        start = None
        for _, row in g2.iterrows():
            act = str(row['activity']).lower()
            ts = row['date']
            if 'connect' in act:
                start = ts
            elif 'disconnect' in act and start is not None:
                total_seconds += (ts - start).total_seconds()
                session_counts += 1
                start = None
        avg_dur = (total_seconds / session_counts / 3600.0) if session_counts>0 else 0.0
        durations.append({'user':u, 'day':pd.Timestamp(d), 'total_usb_session_hours': total_seconds/3600.0, 'avg_usb_session_hours': avg_dur, 'usb_session_count': session_counts})

    dur_df = pd.DataFrame(durations) if durations else pd.DataFrame(columns=['user','day','total_usb_session_hours','avg_usb_session_hours','usb_session_count'])
    dev_pd = dev_agg.to_pandas()
    dev_pd = dev_pd.merge(dur_df, on=['user','day'], how='left').fillna({'total_usb_session_hours':0,'avg_usb_session_hours':0,'usb_session_count':0})
    # After-hours usb connects count
    # recompute from raw_all
    raw_all['hour'] = raw_all['date'].dt.hour
    offhour_conn = raw_all[raw_all['activity'].fillna('').str.lower().str.contains('connect') & ((raw_all['hour'] < WORK_START) | (raw_all['hour'] >= WORK_END))]

    offhour_cnt = offhour_conn.groupby(['user','day']).size().reset_index(name='afterhours_usb_connects')
    dev_pd = dev_pd.merge(offhour_cnt, on=['user','day'], how='left').fillna({'afterhours_usb_connects':0})

    return cudf.from_pandas(dev_pd), raw_all


# -----------------------
# Scenario mapping from answers4.csv
# -----------------------
def process_scenarios(answers_path):
    # default scenario 0
    if not os.path.exists(answers_path):
        return pd.DataFrame(columns=['user','day','scenario'])
    df = pd.read_csv(answers_path, parse_dates=['date'], low_memory=True)
    df = df.dropna(subset=['date'])
    df['day'] = df['date'].dt.floor('D')
    if 'scenario' not in df.columns:
        df['scenario'] = 0
    out = df[['user','day','scenario']].drop_duplicates()
    return out


# -----------------------
# Merge and compute rolling features
# -----------------------
def merge_all(data_dir, answers_path, chunk_bytes=1024**8):
    t0 = time.time()
    print('Processing logon/http/device (chunked cudf) ...')
    logon_cudf, logon_raw_pd = process_logon(os.path.join(data_dir,'logon.csv'), chunk_bytes)
    http_cudf, http_raw_pd = process_http(os.path.join(data_dir,'http.csv'), chunk_bytes)
    dev_cudf, dev_raw_pd = process_device(os.path.join(data_dir,'device.csv'), chunk_bytes)

    # convert cudf aggregates to pandas for merging/rolling
    parts = []
    for df in [logon_cudf, http_cudf, dev_cudf]:
        if isinstance(df, cudf.DataFrame) and len(df)>0:
            parts.append(df.to_pandas())
    if not parts:
        print('No data found')
        return pd.DataFrame()
    merged = parts[0]
    for df in parts[1:]:
        merged = merged.merge(df, on=['user','day'], how='outer')
    merged = merged.fillna(0)

    # ensure day dtype
    merged['day'] = pd.to_datetime(merged['day'])

    # -----------------------
    # Remove all per-user-per-day rows for users that appear with scenario 2 or 3
    # We call process_scenarios once here and reuse it later.
    scenarios = process_scenarios(answers_path)
    if not scenarios.empty:
        bad_users = scenarios.loc[scenarios['scenario'].isin([2,3]), 'user'].unique().tolist()
        if bad_users:
            print(f"Removing {len(bad_users)} users with scenario 2 or 3 from merged data")
            # merged is pandas here (we converted above); support cudf if needed
            if isinstance(merged, cudf.DataFrame):
                merged = merged[~merged['user'].isin(bad_users)]
            else:
                merged = merged[~merged['user'].isin(bad_users)]

    # cross-modal features
    merged['total_activity_count'] = (
        merged.get('logon_count',0).fillna(0) + merged.get('total_http_requests',0).fillna(0) + merged.get('usb_connect_disconnect_count',0).fillna(0)
    )
    # offhour flags as boolean
    merged['offhour_logon_flag'] = (merged.get('offhour_logonlogoff_count',0) > 0).astype(int)
    merged['offhour_usb_flag'] = (merged.get('afterhours_usb_connects',0) > 0).astype(int)
    # for http we approximate offhour http as session durations outside work hours by checking raw http timestamps
    if not http_raw_pd.empty:
        http_raw_pd['hour'] = http_raw_pd['date'].dt.hour
        off_http = http_raw_pd[(http_raw_pd['hour'] < WORK_START) | (http_raw_pd['hour'] >= WORK_END)]
        off_http_cnt = off_http.groupby(['user','day']).size().reset_index(name='offhour_http_count')
        merged = merged.merge(off_http_cnt, on=['user','day'], how='left').fillna({'offhour_http_count':0})
    else:
        merged['offhour_http_count'] = 0
    merged['offhour_http_flag'] = (merged['offhour_http_count'] > 0).astype(int)

    # de-duplicate any accidental duplicated columns (e.g. is_weekend)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # is_weekend - if not present use logon is_weekend or derive from day
    if 'is_weekend' not in merged.columns:
        merged['is_weekend'] = merged['day'].dt.weekday >= 5
    merged['is_weekend'] = merged['is_weekend'].astype(int)

    # -----------------------
    # NEW: requested additional features (z-scores, offhour_http_frac, pc_switch_rate, is_first_time_usb_user, rare_usb_user)
    # Only adding these — no other logic changes.
    # -----------------------

    # ensure sorting for per-user cumulative/transform operations
    merged = merged.sort_values(['user','day']).reset_index(drop=True)

    # offhour_http_frac
    merged['offhour_http_frac'] = (merged['offhour_http_count'] / merged['total_http_requests'].replace(0, np.nan)).fillna(0).astype(float)

    # pc_switch_rate = distinct_pcs / logon_count
    merged['pc_switch_rate'] = (merged.get('distinct_pcs',0) / merged.get('logon_count',0).replace(0, np.nan)).fillna(0).astype(float)

    # is_first_time_usb_user: True when user has usb activity today and no prior usb activity in earlier days
    merged['cum_usb_prev'] = merged.groupby('user')['usb_connect_disconnect_count'].cumsum() - merged['usb_connect_disconnect_count']
    merged['is_first_time_usb_user'] = ((merged['usb_connect_disconnect_count'] > 0) & (merged['cum_usb_prev'] == 0)).astype(int)
    merged.drop(columns=['cum_usb_prev'], inplace=True)

    # rare_usb_user: user-level mean usb usage is in the bottom 10th percentile
    user_mean_usb = merged.groupby('user')['usb_connect_disconnect_count'].transform('mean')
    global_q10 = merged.groupby('user')['usb_connect_disconnect_count'].mean().quantile(0.1)
    merged['rare_usb_user'] = (user_mean_usb < global_q10).astype(int)

    # z-score helper (per-user z using all historical rows for that user)
    def _user_z(df, col):
        grp = df.groupby('user')[col]
        mean = grp.transform('mean')
        std = grp.transform('std').replace(0, np.nan)
        z = (df[col] - mean) / std
        return z.fillna(0)

    z_cols = [
        ('usb_connect_disconnect_count', 'usb_connect_disconnect_zscore'),
        ('offhour_http_count', 'http_offhour_requests_zscore'),
        ('session_duration_hours', 'session_duration_hours_zscore'),
        ('total_http_requests', 'total_http_requests_zscore')
    ]
    for src, dst in z_cols:
        if src in merged.columns:
            merged[dst] = _user_z(merged, src).astype('float32')
        else:
            merged[dst] = 0.0

    # -----------------------
    # Rolling temporal features per-user (existing code)
    # -----------------------
    windows = [7,30]
    roll_feats = {
        'logon_online_duration': 'online_duration_hours',
        'logon_offhour_count': 'offhour_logonlogoff_count',
        'logon_distinct_pcs': 'distinct_pcs',
        'http_total_requests': 'total_http_requests',
        'http_unique_domains': 'unique_domains',
        'device_usb_count': 'usb_connect_disconnect_count',
        'device_total_usb_duration': 'total_usb_session_hours',
        'device_afterhours_usb': 'afterhours_usb_connects'
    }

    # ensure cols
    for col in roll_feats.values():
        if col not in merged.columns:
            merged[col] = 0

    merged = merged.sort_values(['user','day']).reset_index(drop=True)

    # prepare rolling result columns
    for name, col in roll_feats.items():
        for w in windows:
            merged[f'{name}_rollpct_{w}d'] = 0.0
            if name in ['logon_online_duration','http_total_requests','device_usb_count']:
                merged[f'{name}_rollmean_{w}d'] = 0.0
                merged[f'{name}_rollstd_{w}d'] = 0.0

    # rolling percentile function
    def _roll_pctile(arr):
        if len(arr)<=1:
            return 0.0
        last = arr[-1]
        cnt = np.sum(arr[:-1] <= last)
        return float(cnt) / max(1, (len(arr)-1))

    # compute per-user rolling windows
    for user, idx in merged.groupby('user').groups.items():
        i = list(idx)
        sub = merged.loc[i].sort_values('day')
        for name, col in roll_feats.items():
            vals = sub[col].astype('float32')
            for w in windows:
                roll_mean = vals.rolling(window=w, min_periods=1).mean()
                roll_std = vals.rolling(window=w, min_periods=1).std().replace(np.nan,0).replace(0,1)
                roll_pct = vals.rolling(window=w, min_periods=1).apply(_roll_pctile, raw=True).fillna(0)
                merged.loc[i, f'{name}_rollpct_{w}d'] = roll_pct.values
                if f'{name}_rollmean_{w}d' in merged.columns:
                    merged.loc[i, f'{name}_rollmean_{w}d'] = roll_mean.values
                if f'{name}_rollstd_{w}d' in merged.columns:
                    merged.loc[i, f'{name}_rollstd_{w}d'] = roll_std.values
        del sub
        gc.collect()

    # attach scenario mapping and set is_malicious = 1 where user/day match
    # (we already called process_scenarios earlier and stored `scenarios`)

    # normalize scenarios day
    if not scenarios.empty:
        scenarios['day'] = pd.to_datetime(scenarios['day']).dt.floor('D')

    # merge scenarios and create is_malicious flag
    if isinstance(merged, cudf.DataFrame):
        scen_cudf = cudf.from_pandas(scenarios) if not scenarios.empty else cudf.DataFrame(columns=['user','day','scenario'])
        merged = merged.merge(scen_cudf, on=['user','day'], how='left')
        merged['is_malicious'] = (~merged['scenario'].isnull()).astype('int8')
    else:
        # pandas path
        if not scenarios.empty:
            merged = merged.merge(scenarios, on=['user','day'], how='left')
        merged['is_malicious'] = (~merged.get('scenario').isna()).astype(int)

    # ensure columns exist and types are clean
    if 'scenario' not in merged.columns:
        merged['scenario'] = 0
    try:
        merged['scenario'] = merged['scenario'].fillna(0).astype(int)
    except Exception:
        merged['scenario'] = merged['scenario'].fillna(0)

    merged['is_malicious'] = merged['is_malicious'].fillna(0).astype('int8')


    # select and order final columns according to user's requested full set
    final_cols = [
        'user','day',
        # Logon base
        'logon_count','logoff_count','online_duration_hours','distinct_pcs','is_weekend','offhour_logonlogoff_count',
        # HTTP base
        'total_http_requests','unique_domains','avg_requests_per_session','wikileaks_flag',
        # Device base
        'usb_connect_disconnect_count','avg_usb_session_hours','total_usb_session_hours','afterhours_usb_connects',
        # Cross-Modal
        'total_activity_count','offhour_logon_flag','offhour_usb_flag','offhour_http_flag','offhour_http_count',
        # newly added features
        'offhour_http_frac','pc_switch_rate','is_first_time_usb_user','rare_usb_user',
        # scenario
        'scenario', 'is_malicious'
    ]
    # include any rolling columns
    roll_cols = [c for c in merged.columns if ('_rollpct_' in c) or ('_rollmean_' in c) or ('_rollstd_' in c)]
    # include zscore columns if present
    zscore_cols = [c for c in ['usb_connect_disconnect_zscore','http_offhour_requests_zscore','session_duration_hours_zscore','total_http_requests_zscore'] if c in merged.columns]
    final_cols += zscore_cols
    final_cols += roll_cols

    final = merged[[c for c in final_cols if c in merged.columns]].copy()

    # downcast numeric types
    for c in final.select_dtypes(include=['float64']).columns:
        try:
            final[c] = final[c].astype('float32')
        except Exception:
            pass

    # convert back to cudf for compatibility
    try:
        final_cudf = cudf.from_pandas(final)
    except Exception as e:
        print('Warning: could not convert to cudf, returning pandas. Error:', e)
        final_cudf = None

    print(f"Done Merge and features in {time.time() - t0:.2f} seconds")
    if final_cudf is not None:
        return final_cudf
    else:
        return final


if __name__ == '__main__':
    tstart = time.time()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'r4.2'))
    answers_path = os.path.join(os.path.dirname(__file__), 'answers4.csv')
    out = merge_all(data_dir, answers_path, chunk_bytes=256 * 1024 * 1024)
    # write pandas CSV
    if isinstance(out, cudf.DataFrame):
        out = out.to_pandas()
    out.to_csv(os.path.join(os.path.dirname(__file__), 'processed_features_scenario1p3n23f.csv'), index=False)
    print('Scenario 1 feature engineering complete. Output written to processed_features_scenario1p3n23f.csv')
    print(f"Done Everything in {time.time() - tstart:.2f} seconds")

