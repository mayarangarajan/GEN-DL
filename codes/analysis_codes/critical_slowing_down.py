# =============================================================================
# PAIRED ANALYSIS: Null vs Forced Spectral Comparison
# Testing spectral shift in the last 4 weeks approaching bifurcation
# Developed by Maya Rangarajan, 2026
# =============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import mannwhitneyu, wilcoxon
import glob
import re
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binomtest


# -------------------------- CONFIG -----------------------------------------
DATA_DIR = "./"
OUTPUT_DIR = "../../output_charts/critical_slowing_down"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Disease configurations
DISEASES = {
    'COVID': {
        'data_dir': './COVID/data/resids',
        'pattern': 'resids*.csv',
        'label': 'COVID',
        'color': '#c0392b',
        'expected_gen_time': 5
    },
    'mpox': {
        'data_dir': './mpox/data/resids',
        'pattern': 'resids*.csv',
        'label': 'Mpox',
        'color': '#3498db',
        'expected_gen_time': 12
    },
    'Influenza': {
        'data_dir': './flu/data/resids',
        'pattern': 'resids*.csv',
        'label': 'Influenza',
        'color': '#2ecc71',
        'expected_gen_time': 3
    },
    'COVID U.S.': {
        'data_dir': './COVID_state/data/resids',
        'pattern': 'resids*.csv',
        'label': 'COVID U.S.',
        'color': '#34495e',
        'expected_gen_time': 5
    }
}

SAMPLING_RATE = 1.0

# -------------------------- HELPER FUNCTIONS -------------------------------

def save_table_data_to_csv(all_results, output_dir):
    """
    Save table data to CSV files
    """
    from scipy.stats import mannwhitneyu, ks_2samp
    
    # Define metrics
    metrics = [
        # --- Core spectral structure ---
        ('null_dominant_period', 'Dominant Period (Null)'),
        ('forced_dominant_period', 'Dominant Period (Forced)'),
        ('period_shift', 'Dominant Period Shift'),

        ('null_mean_freq', 'Mean Frequency (Null)'),
        ('forced_mean_freq', 'Mean Frequency (Forced)'),
        ('mean_freq_shift', 'Mean Frequency Shift'),

        # --- Power distribution ---
        ('null_low_freq', 'Low Freq Power (Null)'),
        ('forced_low_freq', 'Low Freq Power (Forced)'),
        ('low_freq_shift', 'Low Freq Shift'),

        ('null_mid_freq', 'Mid Freq Power (Null)'),
        ('forced_mid_freq', 'Mid Freq Power (Forced)'),
        ('mid_freq_shift', 'Mid Freq Shift'),

        ('null_high_freq', 'High Freq Power (Null)'),
        ('forced_high_freq', 'High Freq Power (Forced)'),
        ('high_freq_shift', 'High Freq Shift'),

        # --- Multi-scale / structure ---
        ('null_spectral_entropy', 'Spectral Entropy (Null)'),
        ('forced_spectral_entropy', 'Spectral Entropy (Forced)'),
        ('entropy_shift', 'Entropy Shift'),

        ('null_peak_sharpness', 'Peak Sharpness (Null)'),
        ('forced_peak_sharpness', 'Peak Sharpness (Forced)'),
        ('peak_sharpness_shift', 'Peak Sharpness Shift'),

        # --- Composite indices ---
        ('reddening_index', 'Reddening Index'),
        ('multiscale_index', 'Multiscale Index'),
    ]
    
    # ========== CSV 1: Summary Statistics ==========
    summary_rows = []
    
    for metric_key, metric_label in metrics:
        row = {'Metric': metric_label}
        
        for disease_key in all_results.keys():
            data = all_results[disease_key]
            pairs = data['pairs']
            disease_label = data['config']['label']
            
            if len(pairs) > 0:
                df_pairs = pd.DataFrame(pairs)
                
                if metric_key in df_pairs.columns:
                    values = df_pairs[metric_key].values
                    values = values[np.isfinite(values)]
                    
                    if len(values) > 0:
                        row[f'{disease_label}_Median'] = np.median(values)
                        row[f'{disease_label}_Mean'] = np.mean(values)
                        row[f'{disease_label}_Std'] = np.std(values)
                        row[f'{disease_label}_N'] = len(values)
                    else:
                        row[f'{disease_label}_Median'] = np.nan
                        row[f'{disease_label}_Mean'] = np.nan
                        row[f'{disease_label}_Std'] = np.nan
                        row[f'{disease_label}_N'] = 0
                else:
                    row[f'{disease_label}_Median'] = np.nan
                    row[f'{disease_label}_Mean'] = np.nan
                    row[f'{disease_label}_Std'] = np.nan
                    row[f'{disease_label}_N'] = 0
            else:
                row[f'{disease_label}_Median'] = np.nan
                row[f'{disease_label}_Mean'] = np.nan
                row[f'{disease_label}_Std'] = np.nan
                row[f'{disease_label}_N'] = 0
        
        summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = f"{output_dir}/disease_spectral_summary_statistics.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved summary statistics CSV: {summary_csv_path}")
    
    # ========== CSV 2: Pairwise Statistical Tests ==========
    disease_keys = list(all_results.keys())
    comparison_metrics = [
        ('null_dominant_period', 'Dominant Period (Null)'),
        ('forced_dominant_period', 'Dominant Period (Forced)'),
        ('low_freq_shift', 'Low Freq Shift'),
        ('high_freq_shift', 'High Freq Shift'),
        ('mean_freq_shift', 'Mean Freq Shift')
    ]
    
    pairwise_rows = []
    
    for metric_key, metric_label in comparison_metrics:
        for i in range(len(disease_keys)):
            for j in range(i + 1, len(disease_keys)):
                disease1 = disease_keys[i]
                disease2 = disease_keys[j]
                
                data1 = all_results[disease1]
                data2 = all_results[disease2]
                
                pairs1 = data1['pairs']
                pairs2 = data2['pairs']
                
                label1 = data1['config']['label']
                label2 = data2['config']['label']
                
                if len(pairs1) > 0 and len(pairs2) > 0:
                    df1 = pd.DataFrame(pairs1)
                    df2 = pd.DataFrame(pairs2)
                    
                    if metric_key in df1.columns and metric_key in df2.columns:
                        vals1 = df1[metric_key].values
                        vals2 = df2[metric_key].values
                        
                        vals1 = vals1[np.isfinite(vals1)]
                        vals2 = vals2[np.isfinite(vals2)]
                        
                        if len(vals1) >= 3 and len(vals2) >= 3:
                            try:
                                mw_stat, mw_p = mannwhitneyu(vals1, vals2, alternative='two-sided')
                                ks_stat, ks_p = ks_2samp(vals1, vals2)
                                
                                mw_sig = '***' if mw_p < 0.001 else '**' if mw_p < 0.01 else '*' if mw_p < 0.05 else 'ns'
                                ks_sig = '***' if ks_p < 0.001 else '**' if ks_p < 0.01 else '*' if ks_p < 0.05 else 'ns'
                                
                                pairwise_rows.append({
                                    'Metric': metric_label,
                                    'Comparison': f'{label1} vs {label2}',
                                    'Disease1': label1,
                                    'Disease2': label2,
                                    'Disease1_Median': np.median(vals1),
                                    'Disease2_Median': np.median(vals2),
                                    'Disease1_Mean': np.mean(vals1),
                                    'Disease2_Mean': np.mean(vals2),
                                    'MW_Statistic': mw_stat,
                                    'MW_PValue': mw_p,
                                    'MW_Significance': mw_sig,
                                    'KS_Statistic': ks_stat,
                                    'KS_PValue': ks_p,
                                    'KS_Significance': ks_sig
                                })
                            except Exception as e:
                                pass
    
    if pairwise_rows:
        df_pairwise = pd.DataFrame(pairwise_rows)
        pairwise_csv_path = f"{output_dir}/disease_pairwise_comparisons.csv"
        df_pairwise.to_csv(pairwise_csv_path, index=False)
        print(f"Saved pairwise comparisons CSV: {pairwise_csv_path}")
    
    return summary_csv_path, pairwise_csv_path if pairwise_rows else None

def load_residuals(filepath):
    """Load residuals from CSV file"""
    try:
        df = pd.read_csv(filepath)
        if 'residuals' in df.columns:
            return df['residuals'].values
        elif 'Residuals' in df.columns:
            return df['Residuals'].values
        else:
            return df.iloc[:, 1].values
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compute_spectral_features(residuals, fs=1.0):
    """Compute frequency domain features from residuals"""
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 20:
        return None
    
    # Compute power spectral density
    freqs, psd = signal.periodogram(residuals, fs=fs)
    freqs = freqs[1:]  # Skip DC
    psd = psd[1:]
    
    if len(psd) == 0:
        return None
    
    total_power = np.sum(psd)
    if total_power == 0:
        return None
    
    # Normalize PSD
    psd_norm = psd / total_power
    
    # Dominant frequency and period
    peak_idx = np.argmax(psd)
    dominant_freq = freqs[peak_idx]
    dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.inf
    
    # Mean frequency (spectral centroid)
    mean_freq = np.sum(freqs * psd_norm)
    
    # Frequency band powers
    high_freq_mask = freqs > 0.3
    high_freq_power = np.sum(psd_norm[high_freq_mask]) if np.any(high_freq_mask) else 0
    
    low_freq_mask = freqs < 0.05
    low_freq_power = np.sum(psd_norm[low_freq_mask]) if np.any(low_freq_mask) else 0
    
    mid_freq_mask = (freqs >= 0.05) & (freqs <= 0.3)
    mid_freq_power = np.sum(psd_norm[mid_freq_mask]) if np.any(mid_freq_mask) else 0
    
    # Spectral entropy
    psd_for_entropy = psd_norm[psd_norm > 0]
    if len(psd_for_entropy) > 0:
        spectral_entropy = -np.sum(psd_for_entropy * np.log2(psd_for_entropy))
    else:
        spectral_entropy = 0
    
    # Peak sharpness (ratio of peak to mean power)
    mean_power = np.mean(psd[1:]) if len(psd) > 1 else 0
    peak_sharpness = psd[peak_idx] / mean_power if mean_power > 0 else 0
    
    return {
        'dominant_freq': dominant_freq,
        'dominant_period': dominant_period,
        'mean_freq': mean_freq,
        'high_freq_power': high_freq_power,
        'low_freq_power': low_freq_power,
        'mid_freq_power': mid_freq_power,
        'spectral_entropy': spectral_entropy,
        'peak_sharpness': peak_sharpness,
        'length': len(residuals)
    }

def extract_tsid(filename):
    """Extract time series ID from filename"""
    match = re.search(r'(null|forced)(\d+)', filename)
    if match:
        return int(match.group(2))
    return None

# -------------------------- PAIRED ANALYSIS --------------------------------

def analyze_disease_paired(disease_key, disease_config, remove_outliers=True, iqr_multiplier=3):
    """Analyze paired null-forced comparisons for a disease with enhanced metrics and optional outlier removal"""
    
    pattern = os.path.join(disease_config['data_dir'], disease_config['pattern'])
    all_files = glob.glob(pattern)
    
    print(f"\n{'='*60}")
    print(f"Analyzing {disease_config['label']}")
    print(f"Found {len(all_files)} total files")
    
    # Separate null and forced files
    null_files = [f for f in all_files if 'null' in os.path.basename(f)]
    forced_files = [f for f in all_files if 'forced' in os.path.basename(f)]
    
    print(f"  Null files: {len(null_files)}")
    print(f"  Forced files: {len(forced_files)}")
    
    # Create paired data structure
    pairs = {}
    
    # Index null files by tsid
    for null_file in null_files:
        tsid = extract_tsid(os.path.basename(null_file))
        if tsid is not None:
            pairs[tsid] = {'null_file': null_file, 'forced_file': None}
    
    # Match forced files
    for forced_file in forced_files:
        tsid = extract_tsid(os.path.basename(forced_file))
        if tsid is not None and tsid in pairs:
            pairs[tsid]['forced_file'] = forced_file
    
    # Keep only complete pairs
    complete_pairs = {k: v for k, v in pairs.items() if v['forced_file'] is not None}
    
    print(f"  Complete pairs: {len(complete_pairs)}")
    
    if len(complete_pairs) == 0:
        print(f"  WARNING: No complete pairs found!")
        return None
    
    # Analyze each pair
    paired_results = []
    
    for tsid, files in complete_pairs.items():
        null_resid = load_residuals(files['null_file'])
        forced_resid = load_residuals(files['forced_file'])
        
        if null_resid is None or forced_resid is None:
            continue
        
        null_features = compute_spectral_features(null_resid)
        forced_features = compute_spectral_features(forced_resid)
        
        if null_features is None or forced_features is None:
            continue
        
        # Calculate shifts
        low_freq_shift = forced_features['low_freq_power'] - null_features['low_freq_power']
        high_freq_shift = forced_features['high_freq_power'] - null_features['high_freq_power']
        mid_freq_shift = forced_features['mid_freq_power'] - null_features['mid_freq_power']
        mean_freq_shift = forced_features['mean_freq'] - null_features['mean_freq']
        entropy_shift = forced_features['spectral_entropy'] - null_features['spectral_entropy']
        peak_sharpness_shift = forced_features['peak_sharpness'] - null_features['peak_sharpness']
        period_shift = forced_features['dominant_period'] - null_features['dominant_period']
        
        # Composite indices
        reddening_index = low_freq_shift - high_freq_shift - mean_freq_shift
        multiscale_index = entropy_shift - peak_sharpness_shift
        low_freq_pct_shift = (low_freq_shift / null_features['low_freq_power'] * 100) if null_features['low_freq_power'] > 0 else 0
        
        paired_results.append({
            'tsid': tsid,
            'null_length': null_features['length'],
            'forced_length': forced_features['length'],
            'null_low_freq': null_features['low_freq_power'],
            'forced_low_freq': forced_features['low_freq_power'],
            'low_freq_shift': low_freq_shift,
            'low_freq_pct_shift': low_freq_pct_shift,
            'null_high_freq': null_features['high_freq_power'],
            'forced_high_freq': forced_features['high_freq_power'],
            'high_freq_shift': high_freq_shift,
            'mid_freq_shift': mid_freq_shift,
            'entropy_shift': entropy_shift,
            'peak_sharpness_shift': peak_sharpness_shift,
            'period_shift': period_shift,
            'reddening_index': reddening_index,
            'multiscale_index': multiscale_index,
            'null_mean_freq': null_features['mean_freq'],
            'forced_mean_freq': forced_features['mean_freq'],
            'mean_freq_shift': mean_freq_shift,
            'null_dominant_period': null_features['dominant_period'],
            'forced_dominant_period': forced_features['dominant_period'],
            'null_spectral_entropy': null_features['spectral_entropy'],
            'forced_spectral_entropy': forced_features['spectral_entropy'],
            'null_peak_sharpness': null_features['peak_sharpness'],
            'forced_peak_sharpness': forced_features['peak_sharpness'],
        })
    
    # Convert to DataFrame
    df_pairs = pd.DataFrame(paired_results)
    
    # Optional outlier removal (3×IQR by default)
    if remove_outliers and not df_pairs.empty:
        for col in ['low_freq_shift', 'high_freq_shift', 'mid_freq_shift', 'entropy_shift', 'peak_sharpness_shift']:
            q1 = df_pairs[col].quantile(0.25)
            q3 = df_pairs[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            df_pairs = df_pairs[(df_pairs[col] >= lower) & (df_pairs[col] <= upper)]
    
    # Disease-level metrics
    n_pairs = len(df_pairs)
    frac_reddening = np.mean(df_pairs['low_freq_shift'] > 0) if n_pairs > 0 else np.nan
    frac_entropy_increase = np.mean(df_pairs['entropy_shift'] > 0) if n_pairs > 0 else np.nan
    
    # Statistical tests
    wilcoxon_results = {}
    ttest_results = {}
    
    if n_pairs >= 3:
        from scipy.stats import ttest_rel
        try:
            wilcoxon_results['low_freq'] = wilcoxon(df_pairs['null_low_freq'], df_pairs['forced_low_freq'], alternative='less')
            wilcoxon_results['mean_freq'] = wilcoxon(df_pairs['null_mean_freq'], df_pairs['forced_mean_freq'], alternative='greater')
            wilcoxon_results['entropy'] = wilcoxon(df_pairs['null_spectral_entropy'], df_pairs['forced_spectral_entropy'], alternative='less')
            
            ttest_results['low_freq'] = ttest_rel(df_pairs['forced_low_freq'], df_pairs['null_low_freq'])
            ttest_results['mean_freq'] = ttest_rel(df_pairs['forced_mean_freq'], df_pairs['null_mean_freq'])
            ttest_results['entropy'] = ttest_rel(df_pairs['forced_spectral_entropy'], df_pairs['null_spectral_entropy'])
        except Exception as e:
            print(f"Statistical test error: {e}")
    
    return {
        'pairs': df_pairs,
        'config': disease_config,
        'frac_reddening': frac_reddening,
        'frac_entropy_increase': frac_entropy_increase,
        'wilcoxon': wilcoxon_results,
        'ttest': ttest_results
    }

def aggregate_global_metrics(disease_configs):
    """Aggregate enhanced paired metrics across all diseases (no plotting)"""
    
    global_results = []
    
    for disease_key, disease_config in disease_configs.items():
        res = analyze_disease_paired(disease_key, disease_config)
        if res is None or res['pairs'].empty:
            continue
        
        df = res['pairs']
        
        metrics = {
            'disease': disease_config['label'],
            'n_pairs': len(df),
            'median_low_freq_shift': df['low_freq_shift'].median(),
            'mean_low_freq_shift': df['low_freq_shift'].mean(),
            'std_low_freq_shift': df['low_freq_shift'].std(),
            'max_low_freq_shift': df['low_freq_shift'].max(),
            'frac_reddening': res['frac_reddening'],
            'frac_entropy_increase': res['frac_entropy_increase'],
            'wilcoxon_low_freq_p': res['wilcoxon'].get('low_freq').pvalue if res['wilcoxon'].get('low_freq') else np.nan,
            'wilcoxon_entropy_p': res['wilcoxon'].get('entropy').pvalue if res['wilcoxon'].get('entropy') else np.nan,
            'ttest_low_freq_p': res['ttest'].get('low_freq').pvalue if res['ttest'].get('low_freq') else np.nan,
            'ttest_entropy_p': res['ttest'].get('entropy').pvalue if res['ttest'].get('entropy') else np.nan,
        }
        
        global_results.append(metrics)
    
    # Convert to DataFrame
    df_global = pd.DataFrame(global_results)
    
    # Aggregate across all diseases
    agg_summary = {
        'total_pairs': df_global['n_pairs'].sum(),
        'median_low_freq_shift': df_global['median_low_freq_shift'].median(),
        'mean_low_freq_shift': df_global['mean_low_freq_shift'].mean(),
        'std_low_freq_shift': df_global['std_low_freq_shift'].mean(),
        'median_frac_reddening': df_global['frac_reddening'].median(),
        'median_frac_entropy_increase': df_global['frac_entropy_increase'].median(),
        'max_shift_overall': df_global['max_low_freq_shift'].max()
    }
    
    return df_global, agg_summary
# -------------------------- RUN ANALYSIS -----------------------------------

print("Starting paired null-forced spectral analysis...")
print(f"Data directory: {DATA_DIR}")

all_results = {}
for disease_key, disease_config in DISEASES.items():
    result = analyze_disease_paired(disease_key, disease_config)
    if result is not None:
        all_results[disease_key] = result

# -------------------------- STATISTICAL ANALYSIS ---------------------------

print("\n" + "="*80)
print("PAIRED SPECTRAL SHIFT ANALYSIS")
print("="*80)

summary_stats = []

for disease_key, data in all_results.items():
    config = data['config']
    pairs = data['pairs']
    
    if len(pairs) == 0:
        continue
    
    df_pairs = pd.DataFrame(pairs)
    
    print(f"\n{config['label']}:")
    print(f"  N pairs: {len(pairs)}")
    print(f"  Null length: {df_pairs['null_length'].mean():.1f} days (mean)")
    print(f"  Forced length: {df_pairs['forced_length'].mean():.1f} days (mean)")
    
    # Low-frequency power analysis
    print(f"\n  LOW-FREQUENCY POWER (<0.1 cycles/day):")
    print(f"    Null:   {df_pairs['null_low_freq'].median():.4f} (median)")
    print(f"    Forced: {df_pairs['forced_low_freq'].median():.4f} (median)")
    print(f"    Shift:  {df_pairs['low_freq_shift'].median():+.4f} ({df_pairs['low_freq_pct_shift'].median():+.1f}%)")
    
    # Wilcoxon signed-rank test (paired)
    if len(pairs) >= 3:
        stat, pval = wilcoxon(df_pairs['null_low_freq'], df_pairs['forced_low_freq'], alternative='less')
        print(f"    Wilcoxon (null < forced): p={pval:.4f}")
        
        if df_pairs['low_freq_shift'].median() > 0 and pval < 0.05:
            interpretation = "✓✓ SPECTRAL REDDENING (p<0.05)"
        elif df_pairs['low_freq_shift'].median() > 0 and pval < 0.10:
            interpretation = "✓ Reddening trend (p<0.10)"
        elif df_pairs['low_freq_shift'].median() < 0 and (1-pval) < 0.05:
            interpretation = "✗✗ BLUE-SHIFTING (p<0.05)"
        else:
            interpretation = "○ No clear pattern"
        
        print(f"Wilcoxon interpretation    {interpretation}")
    
    # Mean frequency analysis
    print(f"\n  MEAN FREQUENCY (spectral centroid):")
    print(f"    Null:   {df_pairs['null_mean_freq'].median():.4f} cycles/day")
    print(f"    Forced: {df_pairs['forced_mean_freq'].median():.4f} cycles/day")
    print(f"    Shift:  {df_pairs['mean_freq_shift'].median():+.4f}")
    
    if len(pairs) >= 3:
        stat, pval = wilcoxon(df_pairs['null_mean_freq'], df_pairs['forced_mean_freq'], alternative='greater')
        print(f"    Wilcoxon (null > forced): p={pval:.4f}")
    
    # High-frequency power
    print(f"\n  HIGH-FREQUENCY POWER (>0.3 cycles/day):")
    print(f"    Null:   {df_pairs['null_high_freq'].median():.4f}")
    print(f"    Forced: {df_pairs['forced_high_freq'].median():.4f}")
    print(f"    Shift:  {df_pairs['high_freq_shift'].median():+.4f}")
    
    # Store summary
    summary_stats.append({
        'Disease': config['label'],
        'N_pairs': len(pairs),
        'Null_low_freq': df_pairs['null_low_freq'].median(),
        'Forced_low_freq': df_pairs['forced_low_freq'].median(),
        'Low_freq_shift_%': df_pairs['low_freq_pct_shift'].median(),
        'Mean_freq_shift': df_pairs['mean_freq_shift'].median(),
        'Pattern': interpretation if len(pairs) >= 3 else 'N/A'
    })
# =============================================================================
# GLOBAL ANALYSIS: ALL DISEASES COMBINED
# =============================================================================

print("\n" + "="*80)
print("GLOBAL (ALL DISEASES) REDDENING ANALYSIS")
print("="*80)

# Collect all pairs across diseases
# Proper aggregation
all_pairs_list = []

for disease_key, data in all_results.items():
    df_pairs = data['pairs']
    if df_pairs is not None and not df_pairs.empty:
        all_pairs_list.append(df_pairs)

# Concatenate properly
if all_pairs_list:
    df_all = pd.concat(all_pairs_list, ignore_index=True)
else:
    df_all = pd.DataFrame()  # empty fallback

print(f"\nTotal pairs across all diseases: {len(df_all)}")

# --- Core statistics ---
metrics_to_check = [
    ('low_freq_shift', 'Low Freq Shift'),
    ('mid_freq_shift', 'Mid Freq Shift'),
    ('high_freq_shift', 'High Freq Shift'),
    ('mean_freq_shift', 'Mean Freq Shift'),
    ('period_shift', 'Dominant Period Shift'),
    ('entropy_shift', 'Entropy Shift'),
    ('peak_sharpness_shift', 'Peak Sharpness Shift'),
    ('reddening_index', 'Reddening Index'),
    ('multiscale_index', 'Multiscale Index'),
]

for key, label in metrics_to_check:
    if key in df_all.columns:
        vals = df_all[key].dropna()
        if len(vals) > 0:
            print(f"\n{label}:")
            print(f"  Median: {np.median(vals):+.4f}")
            print(f"  Mean:   {np.mean(vals):+.4f}")
            print(f"  Std:    {np.std(vals):.4f}")

# --- Fraction of reddening cases ---
frac_reddening = np.mean(df_all['low_freq_shift'] > 0)
frac_entropy_increase = np.mean(df_all['entropy_shift'] > 0)

print(f"\nFraction with low-freq increase (reddening): {frac_reddening:.3f}")
print(f"Fraction with entropy increase (multiscale): {frac_entropy_increase:.3f}")

# --- Paired statistical tests ---
print("\nPaired tests (ALL diseases):")

if len(df_all) >= 5:
    try:
        stat, p_low = wilcoxon(df_all['null_low_freq'], df_all['forced_low_freq'], alternative='less')
        stat, p_mean = wilcoxon(df_all['null_mean_freq'], df_all['forced_mean_freq'], alternative='greater')
        stat, p_entropy = wilcoxon(df_all['null_spectral_entropy'], df_all['forced_spectral_entropy'], alternative='less')

        print(f"  Low-freq increase (null < forced): p={p_low:.4e}")
        print(f"  Mean-freq decrease (null > forced): p={p_mean:.4e}")
        print(f"  Entropy increase (null < forced): p={p_entropy:.4e}")
    except Exception as e:
        print(f"  Statistical test error: {e}")

# --- Wilcoxon interpretations for global metrics ---
print("\nGlobal Wilcoxon Interpretation:")

if len(df_all) >= 5:
    # Low-frequency power
    stat, p_low = wilcoxon(df_all['null_low_freq'], df_all['forced_low_freq'], alternative='less')
    if df_all['low_freq_shift'].median() > 0 and p_low < 0.05:
        interpretation_low = "✓✓ SPECTRAL REDDENING (p<0.05)"
    elif df_all['low_freq_shift'].median() > 0 and p_low < 0.10:
        interpretation_low = "✓ Reddening trend (p<0.10)"
    elif df_all['low_freq_shift'].median() < 0 and (1-p_low) < 0.05:
        interpretation_low = "✗✗ BLUE-SHIFTING (p<0.05)"
    else:
        interpretation_low = "○ No clear pattern"
    print(f"  Low-frequency power: median shift {df_all['low_freq_shift'].median():+.4f}, {interpretation_low}")

    # Mean frequency
    stat, p_mean = wilcoxon(df_all['null_mean_freq'], df_all['forced_mean_freq'], alternative='greater')
    if df_all['mean_freq_shift'].median() < 0 and p_mean < 0.05:
        interpretation_mean = "✓✓ MEAN FREQ DECREASE (p<0.05)"
    elif df_all['mean_freq_shift'].median() < 0 and p_mean < 0.10:
        interpretation_mean = "✓ Mean freq decreasing trend (p<0.10)"
    elif df_all['mean_freq_shift'].median() > 0 and (1-p_mean) < 0.05:
        interpretation_mean = "✗✗ MEAN FREQ INCREASE (p<0.05)"
    else:
        interpretation_mean = "○ No clear pattern"
    print(f"  Mean frequency: median shift {df_all['mean_freq_shift'].median():+.4f}, {interpretation_mean}")

    # Spectral entropy
    stat, p_entropy = wilcoxon(df_all['null_spectral_entropy'], df_all['forced_spectral_entropy'], alternative='less')
    if df_all['entropy_shift'].median() > 0 and p_entropy < 0.05:
        interpretation_entropy = "✓✓ ENTROPY INCREASE (p<0.05)"
    elif df_all['entropy_shift'].median() > 0 and p_entropy < 0.10:
        interpretation_entropy = "✓ Entropy increasing trend (p<0.10)"
    elif df_all['entropy_shift'].median() < 0 and (1-p_entropy) < 0.05:
        interpretation_entropy = "✗✗ ENTROPY DECREASE (p<0.05)"
    else:
        interpretation_entropy = "○ No clear pattern"
    print(f"  Spectral entropy: median shift {df_all['entropy_shift'].median():+.4f}, {interpretation_entropy}")

# --- Correlations ---
print("\nCorrelation analysis:")

try:
    corr_lf_entropy = np.corrcoef(df_all['low_freq_shift'], df_all['entropy_shift'])[0, 1]
    corr_lf_sharp = np.corrcoef(df_all['low_freq_shift'], df_all['peak_sharpness_shift'])[0, 1]

    print(f"  Corr(low_freq_shift, entropy_shift): {corr_lf_entropy:.3f}")
    print(f"  Corr(low_freq_shift, sharpness_shift): {corr_lf_sharp:.3f}")
except:
    pass

# --- Variance increase ---
print("\nVariance comparison:")

print(f"  Mean freq variance (null):   {np.var(df_all['null_mean_freq']):.6f}")
print(f"  Mean freq variance (forced): {np.var(df_all['forced_mean_freq']):.6f}")
# Save summary
df_summary = pd.DataFrame(summary_stats)
df_summary.to_csv(f"{OUTPUT_DIR}/paired_comparison_summary.csv", index=False)
df_all.to_csv(f"{OUTPUT_DIR}/global_reddening_metrics.csv", index=False)
print(f"\nSummary saved to {OUTPUT_DIR}/paired_comparison_summary.csv")

df_all_diseases, global_summary = aggregate_global_metrics(DISEASES)
print(global_summary)
# -------------------------- VISUALIZATIONS ---------------------------------

print("\nGenerating visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')
pdf_path = f"{OUTPUT_DIR}/paired_spectral_analysis.pdf"
pdf = PdfPages(pdf_path)

# =========Figure 1: Paired comparison - Low-frequency power =======

fig = plt.figure(figsize=(14, 4.2))

# leave room for 2 metric rows above plots
plt.subplots_adjust(top=0.8, bottom=0.10, wspace=0.35)

disease_keys = list(all_results.keys())
axes = []

# -------------------- PLOT LOOP --------------------
for idx, disease_key in enumerate(disease_keys):
    data = all_results[disease_key]
    config = data['config']
    pairs = data['pairs']
    
    if len(pairs) == 0:
        continue
    
    df_pairs = pd.DataFrame(pairs)
    
    ax = plt.subplot(1, 4, idx + 1)
    axes.append(ax)
    
    # paired lines
    for _, row in df_pairs.iterrows():
        ax.plot([0, 1],
                [row['null_low_freq'], row['forced_low_freq']],
                'o-', color=config['color'], alpha=0.4, linewidth=1.5)
    
    # stats
    null_mean = df_pairs['null_low_freq'].mean()
    forced_mean = df_pairs['forced_low_freq'].mean()
    null_std = df_pairs['null_low_freq'].std()
    forced_std = df_pairs['forced_low_freq'].std()
    
    mean_shift = forced_mean - null_mean
    pct_shift = (mean_shift / null_mean * 100) if null_mean > 0 else 0
    
    ax.errorbar([0, 1], [null_mean, forced_mean],
                yerr=[null_std, forced_std],
                fmt='ko-', linewidth=3, markersize=10,
                capsize=5, capthick=2, zorder=10)
    
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Null',
                        'Forced'], fontsize=14)
    
    ax.set_ylabel('Low-Freq Power (<0.1 c/d)', fontsize=16)
    
    ax.set_title(f'{config["label"]} (N={len(pairs)})',
                 fontsize=16, fontweight='bold', loc='left')
    
    ax.grid(True, alpha=0.3, axis='y')

metric_row1_y = 0.92

for i, ax in enumerate(axes):
    data = all_results[disease_keys[i]]
    df_pairs = pd.DataFrame(data['pairs'])
    
    if len(df_pairs) == 0:
        continue
    
    # per-panel anchor
    x_left = ax.get_position().x0
    
    null_mean = df_pairs['null_low_freq'].mean()
    forced_mean = df_pairs['forced_low_freq'].mean()
    mean_shift = forced_mean - null_mean
    pct_shift = (mean_shift / null_mean * 100) if null_mean > 0 else 0

    null_std = df_pairs['null_low_freq'].std()

    # calculate % of outbreaks with CSD signal
    df_pairs['low_freq_z_shift'] = (
        df_pairs['low_freq_shift'] / null_std
    ) if null_std > 0 else np.nan
    df_pairs['is_redshift'] = df_pairs['low_freq_z_shift'] > 1.5

    df_pairs['pct_low_freq_increase'] = (
    (df_pairs['forced_low_freq'] - df_pairs['null_low_freq'])
    / df_pairs['null_low_freq']
    ) * 100

    df_pairs['is_redshift'] = df_pairs['pct_low_freq_increase'] >= 100
    redshift_pct = df_pairs['is_redshift'].mean() * 100
    # redshift_pct = (df_pairs['low_freq_shift'] > 0).mean() * 100
    # redshift_pct = df_pairs['is_redshift'].mean() * 100


    n_red = df_pairs['is_redshift'].sum()
    n_total = len(df_pairs)

    binom_res = binomtest(
        n_red,
        n_total,
        p=0.5,
        alternative='greater'
    )

    redshift_p = binom_res.pvalue
    p_text = "p < 0.0001" if redshift_p < 1e-4 else f"p = {redshift_p:.4f}"

    # ROW 1: main metric
    fig.text(
        x_left, metric_row1_y,
        f"Low frequency shift: \n "
        f"{mean_shift:+.3f} ({pct_shift:+.1f}%)\n",
        # f"% outbreaks slowing: {redshift_pct:.1f}% ",
        linespacing=1.2,
        ha='left',
        va='center',
        fontsize=16
    )


print(f"Saved: {OUTPUT_DIR}/paired_low_freq_comparison.png")

pdf.savefig(fig, dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/paired_low_freq_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# ======== Figure 2: Distribution of shifts
fig = plt.figure(figsize=(12, 5))

for idx, disease_key in enumerate(disease_keys):
    data = all_results[disease_key]
    config = data['config']
    pairs = data['pairs']
    
    if len(pairs) == 0:
        continue
    
    df_pairs = pd.DataFrame(pairs)
    
    ax = plt.subplot(1, 4, idx + 1)
    
    # Histogram of shifts
    ax.hist(df_pairs['low_freq_pct_shift'], bins=15, color=config['color'], 
           alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No change')
    ax.axvline(df_pairs['low_freq_pct_shift'].median(), color='black', 
              linestyle='-', linewidth=2, label=f'Median: {df_pairs["low_freq_pct_shift"].median():+.1f}%')
    
    ax.set_xlabel('Low-Freq Power Shift (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{config["label"]}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation text
    median_shift = df_pairs['low_freq_pct_shift'].median()
    if median_shift > 10:
        text = 'Reddening →'
    elif median_shift < -10:
        text = '← Blue-shift'
    else:
        text = 'Weak/No shift'
    ax.text(0.98, 0.98, text, transform=ax.transAxes, fontsize=11, 
           va='top', ha='right', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
pdf.savefig(fig, dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/shift_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Summary bar chart
fig, ax = plt.subplots(figsize=(10, 6))

diseases = []
shifts = []
colors_list = []

for disease_key in disease_keys:
    data = all_results[disease_key]
    config = data['config']
    pairs = data['pairs']
    
    if len(pairs) > 0:
        df_pairs = pd.DataFrame(pairs)
        diseases.append(config['label'])
        shifts.append(df_pairs['low_freq_pct_shift'].median())
        colors_list.append('#e74c3c' if shifts[-1] > 0 else '#3498db')

x_pos = np.arange(len(diseases))
bars = ax.bar(x_pos, shifts, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.axhline(0, color='black', linewidth=1, linestyle='-')
ax.set_ylabel('Low-Freq Power Shift (%)\nForced vs Null', fontsize=13, labelpad=12)
ax.set_title('Spectral Shift: Last 4 Weeks Approaching Bifurcation', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(diseases, fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Add text labels on bars
for i, (bar, shift) in enumerate(zip(bars, shifts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{shift:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
           fontsize=11, fontweight='bold')

# Add interpretation
ax.text(0.02, 0.98, 'Positive = Reddening (Slowing)\nNegative = Blue-shift (Acceleration)', 
       transform=ax.transAxes, fontsize=14, va='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
pdf.savefig(fig, dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/summary_bar_chart.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------- GENERATE COMPARISON TABLE --------------------------
print("\n" + "="*80)
print("GENERATING DISEASE COMPARISON TABLE")
print("="*80)

def create_disease_comparison_table(all_results):
    """
    Create a comprehensive comparison table for all diseases
    Combines null and forced data into overall disease statistics
    """
    from scipy.stats import mannwhitneyu, ks_2samp
    
    # Define metrics to include 
    metrics = [
        # --- Core spectral structure ---
        ('null_dominant_period', 'Dominant Period (Null)'),
        ('forced_dominant_period', 'Dominant Period (Forced)'),
        ('period_shift', 'Dominant Period Shift'),

        ('null_mean_freq', 'Mean Frequency (Null)'),
        ('forced_mean_freq', 'Mean Frequency (Forced)'),
        ('mean_freq_shift', 'Mean Frequency Shift'),

        # --- Power distribution ---
        ('null_low_freq', 'Low Freq Power (Null)'),
        ('forced_low_freq', 'Low Freq Power (Forced)'),
        ('low_freq_shift', 'Low Freq Shift'),

        ('null_mid_freq', 'Mid Freq Power (Null)'),
        ('forced_mid_freq', 'Mid Freq Power (Forced)'),
        ('mid_freq_shift', 'Mid Freq Shift'),

        ('null_high_freq', 'High Freq Power (Null)'),
        ('forced_high_freq', 'High Freq Power (Forced)'),
        ('high_freq_shift', 'High Freq Shift'),

        # --- Multi-scale / structure ---
        ('null_spectral_entropy', 'Spectral Entropy (Null)'),
        ('forced_spectral_entropy', 'Spectral Entropy (Forced)'),
        ('entropy_shift', 'Entropy Shift'),

        ('null_peak_sharpness', 'Peak Sharpness (Null)'),
        ('forced_peak_sharpness', 'Peak Sharpness (Forced)'),
        ('peak_sharpness_shift', 'Peak Sharpness Shift'),

        # --- Composite indices ---
        ('reddening_index', 'Reddening Index'),
        ('multiscale_index', 'Multiscale Index'),
    ]
    
    # Prepare figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.4)
    
    # ========== TABLE 1: Summary Statistics ==========
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis('tight')
    ax_summary.axis('off')
    
    table_data = []
    for metric_key, metric_label in metrics:
        row = [metric_label]
        
        for disease_key in all_results.keys():
            data = all_results[disease_key]
            pairs = data['pairs']
            
            if len(pairs) > 0:
                df_pairs = pd.DataFrame(pairs)
                
                # Check if metric exists in dataframe
                if metric_key in df_pairs.columns:
                    values = df_pairs[metric_key].values
                    values = values[np.isfinite(values)]
                    
                    if len(values) > 0:
                        median_val = np.median(values)
                        std_val = np.std(values)
                        row.append(f"{median_val:.3f} ± {std_val:.3f}")
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        
        table_data.append(row)
    
    # Create column labels
    col_labels = ['Metric'] + [all_results[k]['config']['label'] for k in all_results.keys()]
    col_widths = [0.3] + [0.7 / len(all_results)] * len(all_results)
    
    table = ax_summary.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            # Highlight shift rows
            if 'shift' in table_data[i-1][0].lower():
                table[(i, 0)].set_facecolor('#f39c12')
                table[(i, 0)].set_text_props(weight='bold')
    
    

# Generate the table BEFORE pdf.close()
try:
    fig_table = create_disease_comparison_table(all_results)
    pdf.savefig(fig_table, dpi=300, bbox_inches='tight')
    
    table_path = f"{OUTPUT_DIR}/disease_comparison_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison table: {table_path}")
    plt.close(fig_table)
    
except Exception as e:
    print(f"Error generating comparison table: {e}")
    import traceback
    traceback.print_exc()

print("\nGenerating disease comparison table...")

try:
    fig_table = create_disease_comparison_table(all_results)
    pdf.savefig(fig_table, dpi=300, bbox_inches='tight')
    
    table_path = f"{OUTPUT_DIR}/disease_comparison_table.png"
    fig_table.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison table: {table_path}")
    plt.close(fig_table)
    
    # ========== SAVE TABLE DATA TO CSV  ==========
    print("\nSaving table data to CSV files...")
    save_table_data_to_csv(all_results, OUTPUT_DIR)
    
except Exception as e:
    print(f"Error generating comparison table: {e}")
    import traceback
    traceback.print_exc()


pdf.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Results saved to: {OUTPUT_DIR}/")
print(f"\nGenerated files:")
print(f"  - paired_comparison_summary.csv")
print(f"  - paired_spectral_analysis.pdf")
print(f"  - paired_low_freq_comparison.png")
print(f"  - shift_distributions.png")
print(f"  - summary_bar_chart.png")
print(f"\nPDF report: {pdf_path}")

print("\n" + "="*80)
print("INTERPRETATION:")
print("  Positive shift = Spectral REDDENING (last 4 weeks show slowing)")
print("  Negative shift = Spectral BLUE-SHIFTING (last 4 weeks show acceleration)")
print("  Near zero = No clear spectral change")
print("="*80)
