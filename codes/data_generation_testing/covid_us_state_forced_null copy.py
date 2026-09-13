# =============================================================================
# COVID-19 JHU → Null / Transcritical Residuals 
# =============================================================================
#
# Written by Maya Rangarajan
#
# Creates null and transcritical outbreak time series from JHU source
# Retrieves data from website + downloads it locally + processes to generate residual series
# Runs on multi-core processor
#  Adjust N_WORKERS based on number of CPUs available
#

import pandas as pd
import numpy as np
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
from concurrent.futures import ProcessPoolExecutor, as_completed
import zipfile
import warnings
from epyestim.covid19 import r_covid
import epyestim.estimate_r as er
from tqdm import tqdm 
import uuid 
import ewstools
import time
import csv

warnings.filterwarnings("ignore")

# -------------------------- CONFIG -----------------------------------------
URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
MIN_TOTAL_CASES = 10          # Skip very small counties
NULL_OFFSET = 28               # Days to truncate for null series
MIN_TRANS_LEN = 56             # Minimum length of transcritical series
OUTPUT_DIR = "real_data_covid2y_state"
N_WORKERS = 5                 # Number of parallel workers
MAX_ZERO_FRACTION = 0.3   
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "resids"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "timeseries"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "output_labels"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "output_groups"), exist_ok=True)

# -------------------------- FUNCTIONS --------------------------------------

EXIT_OK = "ok"
EXIT_TOO_FEW_CASES = "too_few_cases"
EXIT_R_FAIL = "r_failed"
EXIT_NO_WINDOWS = "no_transcritical_windows"
EXIT_WINDOWS_TOO_SHORT = "windows_too_short"
EXIT_TOO_MANY_ZEROS = "too_many_zeros"
EXIT_EXCEPTION = "exception"

ZERO_LOG_FILE = os.path.join(OUTPUT_DIR, "too_many_zeros_log.csv")
with open(ZERO_LOG_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['state_fips', 'zero_frac'])
    writer.writeheader()


def get_column_safe(df, col_name):
    if col_name in df.columns:
        return df[col_name].values
    return np.full(len(df), np.nan)  # fill with NaN if missing

def too_many_zeros(x, tol=1e-8, max_frac=MAX_ZERO_FRACTION, state_fips=None):
    x = np.asarray(x)
    zero_frac = np.mean(np.abs(x) < tol)
    if zero_frac >= max_frac:
        with open(ZERO_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([state_fips, zero_frac])
    return zero_frac >= max_frac


def find_transcritical_windows(r_means, cases_ts, min_len=56, max_len=300): 
    """
    Extract transcritical segments between downward and upward crossings of R=1.

    Upward crossing: R goes from <1 to >=1 (end of segment)
    Downward crossing: R goes from >=1 to <1 (start of segment)
    """

    windows = []

    # Identify upward and downward crossings
    upward_crossings = (r_means.shift(1) < 1) & (r_means >= 1)
    downward_crossings = (r_means.shift(1) >= 1) & (r_means < 1)

    upward_dates = r_means.index[upward_crossings]
    downward_dates = r_means.index[downward_crossings]

    # For each upward crossing, find the closest previous downward crossing
    for up_date in upward_dates:
        # Get all downward crossings before the upward crossing
        prior_downs = downward_dates[downward_dates < up_date]

        if len(prior_downs) == 0:
            # No previous downward crossing; optionally skip or use start of series
            start_date = cases_ts.index[0]
        else:
            start_date = prior_downs[-1]  # Last downward crossing before up_date

        # Extract segment between start_date (inclusive) and up_date (exclusive)
        segment = cases_ts.loc[(cases_ts.index >= start_date) & (cases_ts.index <= up_date)]

        # Enforce max_len by trimming from the start if needed
        if len(segment) > max_len:
            segment = segment.iloc[-max_len:]

        # Enforce minimum length
        if len(segment) >= min_len:
            windows.append((start_date, up_date, segment))

    return windows

def compute_residuals_and_ews(window):
    """
    Compute residuals, smoothed series, lag-1 autocorrelation, and variance
    using ewstools, matching the reference method.
    """
    var_name = 'I'
    rw = 0.25   # rolling window fraction
    span = 0.2  # lowess span
    lag = 1

    # Create ewstools TimeSeries object
    tem_series = window
    ews_dic = ewstools.core.TimeSeries(tem_series, transition=None)
    
    ews_dic.detrend(method='Lowess', span=span)
    
    ews_dic.compute_auto(rolling_window=rw, lag=lag)
    ews_dic.compute_var(rolling_window=rw)
    
    # Extract results
    state_df = ews_dic.state
    ews_df   = ews_dic.ews
    # raw_I = tem_series.values

    # Check if columns exist; if not, fill with NaN to maintain array length
    def get_column_safe(df, col_name):
        if col_name in df.columns:
            return df[col_name].values
        return np.full(len(df), np.nan)
    
    return {
        'raw_I': state_df['state'].values,
        'smoothed': state_df['smoothing'].values,
        'residuals': state_df['residuals'].values,
        'ac1': ews_df['ac1'].values,
        'variance': get_column_safe(ews_df, 'variance')
    }

def process_county_for_transcritical_windows(args):
    """
    Extracts transcritical windows from raw data. Uses epieyestim package
    For each TC series, truncate last 28 days to obtain null sequence
    """

    state_fips, group = args

    # print(f"Starting FIPS {fips}...")

    last_forced_start_idx = -np.inf
    total_cases = group['Daily'].sum()
    days_with_cases = (group['Daily'] > 0).sum()

    if total_cases < MIN_TOTAL_CASES or days_with_cases < 30:
        return [], EXIT_TOO_FEW_CASES

    df_daily = group.groupby('Date')['Daily'].sum().reset_index()
    cases_ts = pd.Series(df_daily['Daily'].values, index=df_daily['Date'])

    try:
        r_estimate = r_covid(cases_ts, r_window_size=14)
        if r_estimate.empty:
            return [], EXIT_R_FAIL # was return[]

        if isinstance(r_estimate.columns, pd.MultiIndex):
            r_means = r_estimate['R']['Mean(R)'].fillna(0)
        else:
            r_means = r_estimate['R_mean'].fillna(0)

    except Exception:
        return [], EXIT_R_FAIL 
    windows = find_transcritical_windows(r_means, cases_ts)
    if len(windows) == 0:
        return [], EXIT_NO_WINDOWS # was return[]
    
    segments = []

    NULL_OFFSET_DAYS = 28  # 4 weeks
    window_counter = 1

    for (start_date, end_date, segment_series) in windows:
        if len(segment_series) < MIN_TRANS_LEN or len(segment_series) <= NULL_OFFSET_DAYS:
            continue
        
        try:
            # 1. Compute BOTH first
            data_tc = compute_residuals_and_ews(segment_series)
            
            # Calculate Null (Label 0)
            null_window = segment_series.iloc[:-NULL_OFFSET_DAYS]
            data_null = compute_residuals_and_ews(null_window)

            # Checking residuals to ensure the Lowess smoothing didn't result in flat-lines
            if too_many_zeros(data_tc['residuals'], state_fips=state_fips) or too_many_zeros(data_null['residuals'], state_fips=state_fips):
                continue 

            local_id = window_counter
            window_counter += 1
            
            pair = [
                {
                    'residuals': data_tc['residuals'], 'raw_I': data_tc['raw_I'],
                    'smoothed': data_tc['smoothed'], 'ac1': data_tc['ac1'],
                    'variance': data_tc['variance'], 'label': 1, 'state_fips': state_fips,
                    'start_date': str(segment_series.index[0])[:10],
                    'end_date': str(segment_series.index[-1])[:10],
                    'local_pair_id': local_id
                },
                {
                    'residuals': data_null['residuals'], 'raw_I': data_null['raw_I'],
                    'smoothed': data_null['smoothed'], 'ac1': data_null['ac1'],
                    'variance': data_null['variance'], 'label': 0, 'state_fips': state_fips,
                    'start_date': str(null_window.index[0])[:10],
                    'end_date': str(null_window.index[-1])[:10],
                    'local_pair_id': local_id
                }
            ]
            segments.extend(pair)
        except Exception as e:
            print(f"[ERROR] STATE FIPS {state_fips}: {e}")
            continue
 

    return segments, EXIT_OK

# -------------------------- MAIN EXECUTION ---------------------------------
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Ensure all required output directories exist
    # ------------------------------------------------------------------
    required_dirs = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "resids"),
        os.path.join(OUTPUT_DIR, "timeseries"),
        os.path.join(OUTPUT_DIR, "output_labels"),
        os.path.join(OUTPUT_DIR, "output_groups"),
    ]

    for d in required_dirs:
        os.makedirs(d, exist_ok=True)

    print("✓ All required output directories verified")

    # --- Load & limit to 3 years ---
    print("Downloading and limiting to first 3 years...")
    df = pd.read_csv(URL, low_memory=False)

    date_cols = [c for c in df.columns if '/' in c]
    dates = pd.to_datetime(date_cols, format='%m/%d/%y')
    mask = dates <= '2023-01-21'
    # mask = (dates >= '2020-11-01') & (dates <= '2021-02-28')

    date_cols_2y = [date_cols[i] for i, keep in enumerate(mask) if keep]

    id_vars = ['FIPS']
    df = df[id_vars + date_cols_2y]

    df_long = df.melt(id_vars=id_vars, value_vars=date_cols_2y,
                      var_name='Date_Str', value_name='Cumulative')
    df_long['Date'] = pd.to_datetime(df_long['Date_Str'], format='%m/%d/%y')
    df_long = df_long.sort_values(['FIPS', 'Date'])

    df_long['Daily'] = df_long.groupby('FIPS')['Cumulative'].diff().fillna(0).clip(lower=0)

    # Derive state FIPS (first 2 digits of county FIPS)
    df_long = df_long.dropna(subset=['FIPS'])
    df_long['State_FIPS'] = (df_long['FIPS'].astype(int) // 1000)

    # Aggregate daily cases to state level
    df_state = (df_long.groupby(['State_FIPS', 'Date'], as_index=False)['Daily']
                    .sum())

    all_segments = []
    ews_forced_rows = []   # label = 1
    ews_null_rows = []


    # Filter state groups
    state_groups = [(sfips, group) for sfips, group in df_state.groupby('State_FIPS')]

    exit_counts = {}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(process_county_for_transcritical_windows, state_group) 
                for state_group in state_groups]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing all states"):
            segments, reason = future.result()
            all_segments.extend(segments)
            exit_counts[reason] = exit_counts.get(reason, 0) + 1

    if len(all_segments) == 0:
        print("No segments generated.")
        exit()
    print("\n=== EXIT REASONS SUMMARY ===")
    total = sum(exit_counts.values())
    for k, v in sorted(exit_counts.items(), key=lambda x: -x[1]):
        print(f"{k:30s}: {v:5d} ({v/total:.1%})")
        
    # 1. Sort to ensure pairs (label 0 and 1) are processed together
    all_segments.sort(key=lambda x: (x['state_fips'], x.get('local_pair_id', 0)))

    id_mapping = {}
    next_seq_id = 1

    for seg in all_segments:
        key = (seg['state_fips'], seg['local_pair_id'])

        if key not in id_mapping:
            id_mapping[key] = next_seq_id
            next_seq_id += 1

        seg['sequence_ID'] = id_mapping[key]

    # 2. Assign Global Sequential Integer IDs
    labels = []
    id_to_fips = []

    for seg in all_segments:
        residuals = seg['residuals']
        sid = seg['sequence_ID']
        suffix = "forced" if seg['label'] == 1 else "null"

        # Save residuals CSV
        pd.DataFrame({
            'Time': np.arange(len(residuals)),
            'residuals': residuals
        }).to_csv(f"{OUTPUT_DIR}/resids/resids_COVID_state_{suffix}{sid}.csv", index=False)

        # Save detailed timeseries CSV
        pd.DataFrame({
            'Time': np.arange(len(seg['raw_I'])),
            'raw_I': seg['raw_I'],
            'smoothed': seg['smoothed'],
            'ac1': seg['ac1'],
            'variance': seg['variance']
        }).to_csv(f"{OUTPUT_DIR}/timeseries/timeseries_COVID_state_{suffix}{sid}.csv", index=False)

        # Append mapping
        id_to_fips.append({
            'sequence_ID': sid,
            'STATE_FIPS': int(seg['state_fips']),
            'start_date': seg.get('start_date', 'unknown'),
            'end_date': seg.get('end_date', 'unknown'),
            'label': seg['label']
        })

        # Append EWS long-format rows
        ews_rows = []
        series_len = len(seg['raw_I'])
        for t in range(series_len):
            ews_rows.append({
                'sid': sid,
                'Variable': 'I',
                'Time': t,
                'state': seg['raw_I'][t],
                'smoothing': seg['smoothed'][t],
                'residuals': seg['residuals'][t],
                'ac1': seg['ac1'][t] if not np.isnan(seg['ac1'][t]) else '',
                'variance': seg['variance'][t] if not np.isnan(seg['variance'][t]) else ''
            })
        if seg['label'] == 1:
            ews_forced_rows.extend(ews_rows)
        else:
            ews_null_rows.extend(ews_rows)

        labels.append({'sequence_ID': sid, 'class_label': seg['label']})

    pd.DataFrame(id_to_fips).to_csv(
        f"{OUTPUT_DIR}/sequence_id_to_state_fips.csv", index=False
    )
    print(f"Saved mapping for {len(id_to_fips):,} sequences → {OUTPUT_DIR}/sequence_id_to_state_fips.csv")

    df_labels = pd.DataFrame(labels)
    df_labels.to_csv(f"{OUTPUT_DIR}/output_labels/labels.csv", index=False)
    
    pd.DataFrame(ews_forced_rows).to_csv(
    f"{OUTPUT_DIR}/df_ews_forced_COVID_state.csv",
    index=False
    )

    pd.DataFrame(ews_null_rows).to_csv(
        f"{OUTPUT_DIR}/df_ews_null_COVID_state.csv",
        index=False
    )

    print("Saved df_ews_forced_COVID_state.csv")
    print("Saved df_ews_null_COVID_state.csv")

    print(f"Saved EWS metrics → {OUTPUT_DIR}/df_ews_null_COVID_state.csv")

    # Train/val/test split
    np.random.seed(42)
    groups = np.random.choice([1, 2, 3], size=len(df_labels), p=[0.8, 0.1, 0.1])
    pd.DataFrame({
        'sequence_ID': df_labels['sequence_ID'],
        'dataset_ID': groups
    }).to_csv(f"{OUTPUT_DIR}/output_groups/groups.csv", index=False)


    # Zip residuals
    print("Creating output_resids.zip...")
    with zipfile.ZipFile(f"{OUTPUT_DIR}/output_resids.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(f"{OUTPUT_DIR}/resids"):
            for f in files:
                zf.write(os.path.join(root, f), arcname=f)

    print("Creating output_timeseries.zip...")
    with zipfile.ZipFile(f"{OUTPUT_DIR}/output_timeseries.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(f"{OUTPUT_DIR}/timeseries"):
            for f in files:
                zf.write(os.path.join(root, f), arcname=f)
    print("\n=== FINISHED ===")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Total segments: {len(df_labels):,}")
    final_forced = sum(1 for l in labels if l['class_label'] == 1)
    final_null   = sum(1 for l in labels if l['class_label'] == 0)

    print(f"Final forced: {final_forced:,}")
    print(f"Final null: {final_null:,}")
    print(f"Final total: {final_forced + final_null:,}")
    print(f"\nFiles saved:")
    print(f"  - {OUTPUT_DIR}/resids/ (residuals CSVs)")
    print(f"  - {OUTPUT_DIR}/timeseries/ (detailed time series CSVs)")
    print(f"  - {OUTPUT_DIR}/output_resids.zip")
    print(f"  - {OUTPUT_DIR}/output_timeseries.zip")
