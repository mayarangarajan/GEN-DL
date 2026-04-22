"""
Spectral Analysis of Disease Dynamics: Comparative Study of COVID-19, Influenza, 
and Mpox Against SIR and SEIPR Model Signatures

FUNCTIONS OF THIS CODE:
Calculate and plot all of the following in raw space:
1. OOD visualization of disease data vs SEIPR training data (samples COVID US and SEIPR)
2. Various spectral metrics for disease data
3. Silhouette separation between diseases 

Maya Rangarajan
November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist

from matplotlib.patches import Ellipse
import warnings
from scipy.stats import mannwhitneyu, ks_2samp

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from itertools import combinations
from collections import defaultdict



warnings.filterwarnings('ignore')

# Set plotting style to match the PDFs
sns.set_style("darkgrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.facecolor'] = '#E8E8F0'
plt.rcParams['figure.facecolor'] = 'white'

# ==================== CONFIGURATION ====================
class Config:
    """Configuration for analysis parameters and toggles"""
    
    # Disease toggles (set to True/False to include/exclude)
    # Setting toggle off does not load data
    INCLUDE_COVID = True
    INCLUDE_FLU = True
    INCLUDE_MPOX = True
    INCLUDE_COVID_COUNTY = True
    COVID_COUNTY_PLOT_SAMPLE = 200
    
    # Model toggles
    INCLUDE_SIR = False
    INCLUDE_SEIPR = True
    
    # Sampling parameters
    SEIPR_SAMPLE_SIZE = 5000  # Sample size for SEIPR (out of 100K)
    SIR_SAMPLE_SIZE = None    # None = use all, or set a number to sample
    RANDOM_SEED = 42          # For reproducibility
    
    # Data paths
    BASE_PATH = Path(".")
    SEIPR_PATH = Path("../../training_data/resids")
    
    # Frequency domain parameters
    LOW_FREQ_THRESHOLD = 0.1   # cycles/day (10 day periods)
    HIGH_FREQ_THRESHOLD = 0.3  # cycles/day (3.3 day periods)
    SAMPLING_RATE = 1.0        # Daily data
    
    # PSD computation parameters
    NPERSEG = 256              # Segment length for Welch's method
    
    # Disease characteristics (biological periods in days)
    DISEASE_PERIODS = {
        'COVID': 5,    # Expected at 5 days
        'Flu': 3,      # Expected at 3 days  
        'Mpox': 12     # Expected at 12 days
    }
    
    # Visual styling - matching the PDFs
    COLORS = {
        'COVID': '#D32F2F',      # Red
        'Flu': '#388E3C',        # Green
        'Mpox': '#1976D2',       # Blue
        "COVID_county": "#7B1FA2",  # Dark maroon
        'SIR': '#E57373',        # Light red
        'SEIPR': '#64B5F6',      # Light blue
    }

    disease_display_labels = {
        'COVID': 'COVID Edmonton',
        'Flu': 'Influenza',
        'Mpox': 'Mpox',
        'COVID_county': 'COVID U.S.'
    }
    
    # Output configuration
    OUTPUT_DIR = Path(".,/../output_charts/raw_spectral_analysis")
    FIGURE_FORMAT = 'png'
    FIGURE_DPI = 300

def run_silhouette_analysis(df_all):
    """
    Returns all regime/disease separation metrics in a structured dict.
    """

    feature_cols = ['low_freq_power', 'mid_freq_power', 'high_freq_power']

    # -------------------------
    # PREPROCESS (single source of truth)
    # -------------------------
    df_valid = df_all.dropna(subset=feature_cols + ['regime', 'dataset'])

    X = df_valid[feature_cols].values
    y_regime = df_valid['regime'].values
    disease = df_valid['dataset'].values

    # remove SEIPR once
    mask = disease != 'SEIPR'
    X = X[mask]
    y_regime = y_regime[mask]
    disease = disease[mask]

    # scale once
    X = StandardScaler().fit_transform(X)

    # encode diseases once
    le = LabelEncoder()
    disease_num = le.fit_transform(disease)

    mask_tc = y_regime == 1

    # -------------------------
    # SAFE HELPERS
    # -------------------------
    def safe_mean(d):
        return np.mean(list(d.values())) if len(d) > 0 else np.nan

    def pairwise_sil(X, labels, mask=None):
        out = {}
        if mask is not None:
            X = X[mask]
            labels = labels[mask]

        for d1, d2 in combinations(np.unique(labels), 2):
            m = (labels == d1) | (labels == d2)
            Xp = X[m]
            yp = labels[m]

            if len(np.unique(yp)) < 2:
                continue

            out[(d1, d2)] = silhouette_score(Xp, le.transform(yp))

        return out

    def regime_by_disease(X, y_regime, disease, mask=None):
        out = {}
        if mask is not None:
            X = X[mask]
            y_regime = y_regime[mask]
            disease = disease[mask]

        for d in np.unique(disease):
            m = disease == d
            if np.sum(m) < 5:
                continue

            Xd = X[m]
            yd = y_regime[m]

            if len(np.unique(yd)) > 1:
                out[d] = silhouette_score(Xd, yd)

        return out

    # -------------------------
    # METRICS
    # -------------------------

    # 1. regime global
    regime_global = silhouette_score(X, y_regime)

    # 2. regime per disease (ALL)
    regime_all = regime_by_disease(X, y_regime, disease)
    mean_regime_all = safe_mean(regime_all)

    # 3. regime per disease (TC)
    regime_tc = regime_by_disease(X, y_regime, disease, mask=mask_tc)
    mean_regime_tc = safe_mean(regime_tc)

    # 4. disease global
    disease_global = silhouette_score(X, disease_num)

    # 5. pairwise disease (ALL)
    pairwise_all = pairwise_sil(X, disease)
    mean_pairwise_all = safe_mean(pairwise_all)
    ratio_all = mean_regime_all / mean_pairwise_all if mean_pairwise_all else np.nan


    # 6. pairwise disease (TC)
    pairwise_tc = pairwise_sil(X, disease, mask=mask_tc)
    mean_pairwise_tc = safe_mean(pairwise_tc)
    ratio_tc = mean_regime_all / mean_pairwise_tc if mean_pairwise_tc else np.nan


    # -------------------------
    # OUTPUT STRUCTURE
    # -------------------------
    return {
        "regime_global": regime_global,

        "regime_all": regime_all,
        "mean_regime_all": mean_regime_all,

        "regime_tc": regime_tc,
        "mean_regime_tc": mean_regime_tc,

        "disease_global": disease_global,

        "pairwise_all": pairwise_all,
        "mean_pairwise_all": mean_pairwise_all,

        "pairwise_tc": pairwise_tc,
        "mean_pairwise_tc": mean_pairwise_tc,

        "ratio_all": ratio_all,
        "ratio_tc": ratio_tc
    }

def compute_pairwise_statistics(df_all, datasets):
    """
    Compute pairwise statistical comparisons between datasets
    
    Parameters:
    -----------
    df_all : pd.DataFrame
        Combined dataframe with all datasets
    datasets : list
        List of dataset names to compare
    
    Returns:
    --------
    pd.DataFrame : Statistical comparison results
    """
    metrics = ['low_freq_power', 'high_freq_power', 'dominant_period', 
               'peak_sharpness', 'spectral_entropy']
    
    stats_results = []
    
    for metric in metrics:
        row_data = {'Metric': metric}
        
        # Get data for each dataset
        for dataset in datasets:
            df_subset = df_all[df_all['dataset'] == dataset]
            values = df_subset[metric].values
            values = values[np.isfinite(values)]
            
            if len(values) > 0:
                row_data[f'{dataset}_Median'] = np.median(values)
                row_data[f'{dataset}_Mean'] = np.mean(values)
                row_data[f'{dataset}_Std'] = np.std(values)
                row_data[f'{dataset}_N'] = len(values)
            else:
                row_data[f'{dataset}_Median'] = np.nan
                row_data[f'{dataset}_Mean'] = np.nan
                row_data[f'{dataset}_Std'] = np.nan
                row_data[f'{dataset}_N'] = 0
        
        stats_results.append(row_data)
    
    return pd.DataFrame(stats_results)


def create_disease_comparison_table(df_all, output_path=None):
    """
    Create a statistical comparison table for diseases
    Similar to the SIR vs SEIPR table
    """
    # diseases = ['COVID', 'Flu', 'Mpox', 'COVID_county']
    diseases = ['COVID', 'COVID_county','Mpox', 'Flu']

    # Compute statistics
    df_stats = compute_pairwise_statistics(df_all, diseases)
    
    # Perform pairwise statistical tests
    metrics = ['low_freq_power', 'high_freq_power', 'dominant_period', 
               'peak_sharpness', 'spectral_entropy']
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.8], hspace=0.3)
    
    # Main table
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table_data = []
    for _, row in df_stats.iterrows():
        table_data.append([
            row['Metric'],

            f"{row['COVID_Median']:.4f} ± {row['COVID_Std']:.4f}" if not np.isnan(row['COVID_Median']) else 'N/A',

            f"{row['COVID_county_Median']:.4f} ± {row['COVID_county_Std']:.4f}" if not np.isnan(row['COVID_county_Median']) else 'N/A',

            f"{row['Mpox_Median']:.4f} ± {row['Mpox_Std']:.4f}" if not np.isnan(row['Mpox_Median']) else 'N/A',

            f"{row['Flu_Median']:.4f} ± {row['Flu_Std']:.4f}" if not np.isnan(row['Flu_Median']) else 'N/A'
        ])
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=[
            'Metric',
            'COVID (median±std)',
            'COVID county (median±std)',
            'Mpox (median±std)',
            'Flu (median±std)'
        ],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.19, 0.19, 0.19, 0.19]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code rows
    n_cols = len(table_data[0])  # automatically detect column count

    for i in range(1, len(table_data) + 1):
        for j in range(n_cols):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
        
    plt.suptitle('Disease Spectral Feature Comparison: COVID-19, Influenza, and Mpox', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, df_stats

def load_residuals_from_csv(filepath):
    """
    Load residuals from a CSV file
    
    Parameters:
    -----------
    filepath : Path
        Path to CSV file
    
    Returns:
    --------
    ndarray or None : Residual values
    """
    try:
        df = pd.read_csv(filepath)
        
        # Try different column names
        for col_name in ['residuals', 'Residuals', 'resid', 'value', 'values']:
            if col_name in df.columns:
                resid = df[col_name].values
                resid = resid[~np.isnan(resid)]
                return resid if len(resid) > 0 else None
        
        # If no named column, take first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            resid = df[numeric_cols[0]].values
            resid = resid[~np.isnan(resid)]
            return resid if len(resid) > 0 else None
            
    except Exception as e:
        print(f"  Error loading {filepath.name}: {e}")
    
    return None


def compute_psd(residuals, fs=1.0, nperseg=256):
    """
    Compute Power Spectral Density using Welch's method
    
    Parameters:
    -----------
    residuals : array-like
        Time series residuals
    fs : float
        Sampling frequency (default: 1.0 for daily data)
    nperseg : int
        Length of each segment for Welch's method
    
    Returns:
    --------
    freqs : ndarray
        Frequency bins
    psd : ndarray
        Power spectral density (normalized)
    """
    if len(residuals) < 4:
        return None, None
    
    # Adjust nperseg to data length
    nperseg_actual = min(len(residuals), nperseg)
    
    freqs, psd = signal.welch(residuals, fs=fs, nperseg=nperseg_actual,
                              scaling='density', detrend='constant')
    
    # Normalize PSD
    if np.sum(psd) > 0:
        psd = psd / np.sum(psd)
    
    return freqs, psd


def compute_spectral_features(freqs, psd, low_thresh=0.1, high_thresh=0.3):
    """
    Compute spectral features from PSD
    
    Parameters:
    -----------
    freqs : ndarray
        Frequency array
    psd : ndarray
        Power spectral density
    low_thresh : float
        Low frequency threshold (cycles/day)
    high_thresh : float
        High frequency threshold (cycles/day)
    
    Returns:
    --------
    dict : Dictionary of spectral features
    """
    if freqs is None or psd is None or len(freqs) < 2:
        return None
    
    # Compute power in frequency bands
    low_mask = freqs < low_thresh
    med_mask = (freqs >= low_thresh) & (freqs < high_thresh)
    high_mask = freqs >= high_thresh
    
    low_power = np.sum(psd[low_mask]) if np.any(low_mask) else 0
    med_power = np.sum(psd[med_mask]) if np.any(med_mask) else 0
    high_power = np.sum(psd[high_mask]) if np.any(high_mask) else 0
    
    # Find dominant frequency (excluding DC component)
    if len(psd) > 1:
        peak_idx = np.argmax(psd[1:]) + 1
        dominant_freq = freqs[peak_idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else np.inf
        peak_power = psd[peak_idx]
    else:
        dominant_freq = 0
        dominant_period = np.inf
        peak_power = 0
    
    # Compute spectral entropy
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm)) if len(psd_norm) > 0 else 0
    
    # Peak sharpness (ratio of peak to mean)
    mean_power = np.mean(psd[1:]) if len(psd) > 1 else 0
    peak_sharpness = peak_power / mean_power if mean_power > 0 else 0
    
    return {
        'low_freq_power': low_power,
        'mid_freq_power': med_power,
        'high_freq_power': high_power,
        'dominant_freq': dominant_freq,
        'dominant_period': dominant_period,
        'spectral_entropy': spectral_entropy,
        'peak_sharpness': peak_sharpness
    }

# ==================== OOD QUANTIFICATION ====================

def compute_ood_metrics(df_all):
    """
    Quantify out-of-distribution generalization metrics
    
    Method 1: Centroid Distance
    Method 3: Percentage Outside Convex Hull
    
    Only computed for SEIPR training data vs disease test data
    """
    from scipy.spatial import ConvexHull, Delaunay
    from scipy.spatial.distance import euclidean
    
    print("\n" + "="*80)
    print("OUT-OF-DISTRIBUTION GENERALIZATION METRICS")
    print("="*80)
    
    # Get SEIPR training distribution
    df_seipr = df_all[df_all['dataset'] == 'SEIPR']
    if len(df_seipr) == 0:
        print("Warning: No SEIPR data available for OOD analysis")
        return None
    
    # seipr_freq = df_seipr[['low_freq_power', 'mid_freq_power', 'high_freq_power']].values
    seipr_freq = df_seipr[['low_freq_power', 'high_freq_power']].values
    seipr_freq = seipr_freq[np.all(np.isfinite(seipr_freq), axis=1)]
    
    if len(seipr_freq) < 3:
        print("Warning: Insufficient SEIPR data for OOD analysis")
        return None
    
    seipr_centroid = seipr_freq.mean(axis=0)
    
    print(f"\nSEIPR training distribution:")
    print(f"  Centroid: (low={seipr_centroid[0]:.4f}, high={seipr_centroid[1]:.4f})")
    print(f"  N sequences: {len(seipr_freq)}")
    
    # Compute convex hull for SEIPR
    try:
        hull = ConvexHull(seipr_freq)
        delaunay = Delaunay(seipr_freq[hull.vertices])
    except Exception as e:
        print(f"Warning: Could not compute convex hull: {e}")
        delaunay = None
    
    # Initialize results
    ood_results = []
    
    # Analyze each disease
    diseases = []
    if Config.INCLUDE_COVID:
        diseases.append('COVID')
    if Config.INCLUDE_COVID_COUNTY:
        diseases.append('COVID_county')
    if Config.INCLUDE_FLU:
        diseases.append('Flu')
    if Config.INCLUDE_MPOX:
        diseases.append('Mpox')
    
    
    for disease in diseases:
        # Get disease data (both forced and null)
        df_disease = df_all[df_all['dataset'] == disease]
        
        if len(df_disease) == 0:
            continue
        
        # disease_freq = df_disease[['low_freq_power', 'mid_freq_power', 'high_freq_power']].values
        disease_freq = df_disease[['low_freq_power', 'high_freq_power']].values
        disease_freq = disease_freq[np.all(np.isfinite(disease_freq), axis=1)]
        
        if len(disease_freq) == 0:
            continue
        
        disease_centroid = disease_freq.mean(axis=0)
        diffs = disease_freq - disease_centroid  # Nx2 array of differences
        distances = np.linalg.norm(diffs, axis=1)  # Euclidean distance for each point
        dispersion = distances.mean()  # mean distance as dispersion

        
        # Method 1: Centroid Distance
        centroid_dist = euclidean(seipr_centroid, disease_centroid)
        
        # Method 3: Percentage Outside Hull
        if delaunay is not None:
            inside = delaunay.find_simplex(disease_freq) >= 0
            pct_outside = 100 * (1 - inside.mean())
        else:
            pct_outside = np.nan
        
        ood_results.append({
            'Disease': disease,
            'N_sequences': len(disease_freq),
            'Centroid_Low': disease_centroid[0],
            'Centroid_High': disease_centroid[1],
            'Centroid_Distance': centroid_dist,
            'Percent_Outside_Hull': pct_outside,
            'Dispersion_From_Centroid': dispersion
        })
        
        print(f"\n{disease}:")
        print(f"  Centroid: (low={disease_centroid[0]:.4f}, high={disease_centroid[1]:.4f})")
        print(f"  Distance from SEIPR: {centroid_dist:.4f}")
        print(f"  Dispersion_From_Centroid from SEIPR: {dispersion:.4f}")
        print(f"  Sequences outside SEIPR hull: {pct_outside:.1f}%")
        print(f"  N sequences: {len(disease_freq)} (forced + null)")
    
    # Create comparison
    if ood_results:
        df_ood = pd.DataFrame(ood_results)
        
        # Find relative distances
        min_dist = df_ood['Centroid_Distance'].min()
        df_ood['Relative_Distance'] = df_ood['Centroid_Distance'] / min_dist
        
        print("\n" + "="*80)
        print("SUMMARY:")
        print("="*80)
        print("\nCentroid Distances from SEIPR:")
        for _, row in df_ood.iterrows():
            print(f"  {row['Disease']:8s}: {row['Centroid_Distance']:.4f} "
                  f"({row['Relative_Distance']:.2f}× baseline)")
        
        print("\nPercentage Outside SEIPR Training Distribution:")
        for _, row in df_ood.iterrows():
            print(f"  {row['Disease']:8s}: {row['Percent_Outside_Hull']:.1f}%")
        
        # Save results
        ood_csv_path = Config.OUTPUT_DIR / 'ood_metrics.csv'
        df_ood.to_csv(ood_csv_path, index=False)
        print(f"\nOOD metrics saved to: {ood_csv_path}")
        
        return df_ood
    
    return None


def create_ood_visualization(df_all, df_ood, disease_display_labels):
    """
    Create visualization showing OOD metrics on the frequency distribution plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    # Get SEIPR data
    df_seipr = df_all[df_all['dataset'] == 'SEIPR']
    # seipr_freq = df_seipr[['low_freq_power', 'mid_freq_power','high_freq_power']].values
    seipr_freq = df_seipr[['low_freq_power', 'high_freq_power']].values
    seipr_freq = seipr_freq[np.all(np.isfinite(seipr_freq), axis=1)]
    
    # Plot SEIPR cloud (no hull boundary)
    sample_idx = np.random.choice(len(seipr_freq), 
                                   min(500, len(seipr_freq)), 
                                   replace=False)
    ax.scatter(seipr_freq[sample_idx, 0], seipr_freq[sample_idx, 1],
              c=Config.COLORS['SEIPR'], alpha=0.3, s=30,
              label='GEN-DL data', zorder=1)

    
    # Plot diseases
    diseases = [d for d in df_all['dataset'].unique() if d not in ['SEIPR']]
    
    for disease in diseases:
        df_disease = df_all[df_all['dataset'] == disease]
        
        display_name = disease_display_labels[disease]
        all_disease_freq = []
        
        # Plot forced and null separately
        for forcing_type in ['forced', 'null']:
        # for forcing_type in ['null']:
            df_subset = df_disease[df_disease['forcing_type'] == forcing_type]
            
            # disease_freq = df_subset[['low_freq_power', 'mid_freq_power', 'high_freq_power']].values
            disease_freq = df_subset[['low_freq_power', 'high_freq_power']].values
            disease_freq = disease_freq[np.all(np.isfinite(disease_freq), axis=1)]
            
            if len(disease_freq) > 0:
                all_disease_freq.append(disease_freq)

        if len(all_disease_freq) == 0:
            continue
        all_disease_freq = np.vstack(all_disease_freq)

        if disease == 'COVID_county' and len(disease_freq) > Config.COVID_COUNTY_PLOT_SAMPLE // 2:
            np.random.seed(Config.RANDOM_SEED)
            plot_idx = np.random.choice(
                len(disease_freq),
                Config.COVID_COUNTY_PLOT_SAMPLE,
                replace=False
            )
            disease_freq_plot = all_disease_freq[plot_idx]
        else:
            disease_freq_plot = all_disease_freq
        
        # Choose marker based on forcing type
        marker = 's' if forcing_type == 'forced' else 'o'
        marker_size = 100 if forcing_type == 'forced' else 100
        label_suffix = ' (forced)' if forcing_type == 'forced' else ' (null)'
            
            # Plot disease points
        ax.scatter(disease_freq_plot[:, 0], disease_freq_plot[:, 1],
                c=Config.COLORS[disease], alpha=0.7, s=marker_size,
                marker=marker, label=f'{display_name}', 
                edgecolors='white', linewidth=0.5, zorder=3)
        
        # Use all data (both forced and null) for centroid
        # disease_freq = df_disease[['low_freq_power', 'mid_freq_power', 'high_freq_power']].values
        disease_freq = df_disease[['low_freq_power', 'high_freq_power']].values
        disease_freq = disease_freq[np.all(np.isfinite(disease_freq), axis=1)]
        
        if len(disease_freq) == 0:
            continue
        '''
        # Plot disease centroid (smaller)
        # disease_centroid = all_disease_freq.mean(axis=0)
        # ax.scatter(disease_centroid[0], disease_centroid[1],
        #         c=Config.COLORS[disease], marker='*', s=250,
        #          edgecolors='black', linewidth=1.5,
        #          label=f'{display_name} Centroid', zorder=4)
        
        # Draw line from SEIPR to disease centroid
        
        ax.plot([seipr_centroid[0], disease_centroid[0]],
               [seipr_centroid[1], disease_centroid[1]],
               color=Config.COLORS[disease], linestyle='--',
               linewidth=2, alpha=0.6, zorder=2)
        '''
    ax.set_xlabel('Low Frequency Power (<0.1 cycles/day, >10 day periods)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('High Frequency Power (>0.3 cycles/day, <3.3 day periods)', 
                  fontsize=12, fontweight='bold')
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.8)
    
    # Legend in upper right
    legend = ax.legend(loc='upper right', framealpha=0.95, fontsize=12, ncol=2)
    
    # Add OOD statistics below legend (upper right) - no box, larger bold text
    if df_ood is not None:
        stats_lines = []
        stats_lines.append("Out-of-Distribution Metrics:")
        
        for _, row in df_ood.iterrows():
            disease = row['Disease']
            display_name = disease_display_labels.get(disease, disease)
            dist = row['Centroid_Distance']
            pct = row['Percent_Outside_Hull']
            dispersion = row.get('Dispersion_From_Centroid', np.nan)
            # stats_lines.append(f"{display_name}: d={dist:.4f}, {pct:.0f}% OOD")
            stats_lines.append(
                f"{display_name}: {pct:.0f}% OOD, Dispersion={dispersion:.4f}"
            )
        stats_text = '\n'.join(stats_lines)
        
        ax.text(0.98, 0.9, stats_text,
               transform=ax.transAxes,
               fontsize=12,  # Larger
               fontweight='bold',
               verticalalignment='top',
               horizontalalignment='right',
               linespacing=2.0)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def load_disease_residuals(disease_name, base_path):
    """
    Load all residuals for a disease
    
    Parameters:
    -----------
    disease_name : str
        Disease name (COVID, Flu, Mpox)
    base_path : Path
        Base path to residuals
    
    Returns:
    --------
    list : List of residual arrays
    """
    print(f"\nLoading {disease_name} residuals...")
    
    disease_path = base_path / disease_name / "data" / "resids"
    if not disease_path.exists():
        print(f"  Warning: Path {disease_path} not found")
        return []
    
    residuals = []
    filenames = []
    csv_files = list(disease_path.glob("resids*.csv"))
    
    print(f"  Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        resid = load_residuals_from_csv(csv_file)
        if resid is not None:
            residuals.append(resid)
            filenames.append(csv_file)

    
    print(f"  Loaded {len(residuals)} valid time series")
    return residuals, filenames


def load_sir_residuals(sir_path, sample_size=None):
    """
    Load SIR residuals from all noise types
    
    Parameters:
    -----------
    sir_path : Path
        Path to SIR data directory
    sample_size : int or None
        Number of files to sample (None = all)
    
    Returns:
    --------
    list : List of residual arrays
    """
    print(f"\nLoading SIR residuals...")
    
    noise_types = ['AddWhiteNoise', 'DemNoise', 'MultEnvNoise']
    all_files = []
    
    for noise_type in noise_types:
        noise_path = sir_path / noise_type
        if noise_path.exists():
            files = list(noise_path.glob("*.csv"))
            all_files.extend(files)
            print(f"  Found {len(files)} files in {noise_type}")
    
    # Sample if requested
    if sample_size is not None and len(all_files) > sample_size:
        np.random.seed(Config.RANDOM_SEED)
        all_files = np.random.choice(all_files, sample_size, replace=False).tolist()
        print(f"  Sampled {sample_size} files")
    
    residuals = []
    for csv_file in all_files:
        resid = load_residuals_from_csv(csv_file)
        if resid is not None:
            residuals.append(resid)
    
    print(f"  Loaded {len(residuals)} valid time series")
    return residuals


def load_seipr_residuals(seipr_path, sample_size=3000):
    """
    Load and sample SEIPR residuals
    
    Parameters:
    -----------
    seipr_path : Path
        Path to SEIPR residuals
    sample_size : int
        Number of files to sample
    
    Returns:
    --------
    list : List of residual arrays
    """
    print(f"\nLoading SEIPR residuals...")
    
    if not seipr_path.exists():
        print(f"  Warning: Path {seipr_path} not found")
        return []
    
    all_files = list(seipr_path.glob("**/resids*.csv"))
    print(f"  Found {len(all_files)} CSV files")
    
    # Sample files
    if len(all_files) > sample_size:
        np.random.seed(Config.RANDOM_SEED)
        sampled_files = np.random.choice(all_files, sample_size, replace=False)
    else:
        sampled_files = all_files
        print(f"  Using {len(all_files)} files")
    
    residuals = []
    for csv_file in sampled_files:
        resid = load_residuals_from_csv(csv_file)
        if resid is not None:
            residuals.append(resid)
    
    print(f"  Loaded {len(residuals)} valid time series")
    return residuals


def analyze_dataset(residuals, dataset_name, filenames=None):
    """
    Analyze a dataset and compute spectral features
    
    Parameters:
    -----------
    residuals : list
        List of residual time series
    dataset_name : str
        Name of the dataset
    
    Returns:
    --------
    pd.DataFrame : DataFrame with spectral features
    """
    print(f"\nAnalyzing {dataset_name}...")
    
    features_list = []
    
    for i, resid in enumerate(residuals):
        freqs, psd = compute_psd(resid, fs=Config.SAMPLING_RATE, 
                                 nperseg=Config.NPERSEG)
        
        if freqs is not None:
            features = compute_spectral_features(freqs, psd,
                                                Config.LOW_FREQ_THRESHOLD,
                                                Config.HIGH_FREQ_THRESHOLD)
            
            if features is not None:
                features['dataset'] = dataset_name
                features['series_id'] = i

                # Determine forcing type
                if filenames is not None and i < len(filenames):
                    filename = str(filenames[i]).lower()
                    if 'forced' in filename:
                        features['forcing_type'] = 'forced'
                    elif 'null' in filename:
                        features['forcing_type'] = 'null'
                    else:
                        features['forcing_type'] = 'unknown'
                else:
                    features['forcing_type'] = 'unknown'

                features['regime'] = 1 if features['forcing_type'] == 'forced' else 0
                features_list.append(features)
    
    df = pd.DataFrame(features_list)
    print(f"  Computed features for {len(df)} time series")
    
    return df


# ==================== FIGURE 1: SPECTRAL SIGNATURES ====================

def create_figure1(df_all):
    """
    Create Figure 1: Disease Spectral Signatures in 2D Frequency Space
    Matches the style from Page 4 of disease analysis PDF
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    datasets_to_plot = []
    if Config.INCLUDE_COVID:
        datasets_to_plot.append('COVID')
    if Config.INCLUDE_FLU:
        datasets_to_plot.append('Flu')
    if Config.INCLUDE_MPOX:
        datasets_to_plot.append('Mpox')
    if Config.INCLUDE_COVID_COUNTY:
        datasets_to_plot.append('COVID_county')
    
    # Plot disease data points
    for disease in datasets_to_plot:
        df_disease = df_all[df_all['dataset'] == disease]
        if len(df_disease) > 0:
            # --- START SAMPLING CODE ---
            plot_df = df_disease
            
            # Apply sampling only to COVID_county if it exceeds our limit
            if disease == 'COVID_county' and len(df_disease) > Config.COVID_COUNTY_PLOT_SAMPLE:
                plot_df = df_disease.sample(
                    n=Config.COVID_COUNTY_PLOT_SAMPLE, 
                    random_state=Config.RANDOM_SEED
                )
            # --- END SAMPLING CODE ---
            ax.scatter(plot_df['low_freq_power'], 
                      plot_df['high_freq_power'],
                      c=Config.COLORS[disease], 
                      alpha=0.7, 
                      s=100, 
                      label=f'{disease}',
                      edgecolors='white', 
                      linewidth=0.5,
                      zorder=3)
    
    # Add SEIPR cloud if available
    if Config.INCLUDE_SEIPR:
        df_seipr = df_all[df_all['dataset'] == 'SEIPR']
        if len(df_seipr) > 0:
            # Plot sample points
            sample_idx = np.random.choice(len(df_seipr), 
                                         min(500, len(df_seipr)), 
                                         replace=False)
            ax.scatter(df_seipr.iloc[sample_idx]['low_freq_power'],
                      df_seipr.iloc[sample_idx]['high_freq_power'],
                      c=Config.COLORS['SEIPR'],
                      alpha=0.3,
                      s=30,
                      label='SEIPR',
                      edgecolors='none',
                      zorder=1)
            
            # Add ellipse
            center_x = df_seipr['low_freq_power'].median()
            center_y = df_seipr['high_freq_power'].median()
            width = 2 * df_seipr['low_freq_power'].std()
            height = 2 * df_seipr['high_freq_power'].std()
            
            ellipse = Ellipse((center_x, center_y), width, height,
                            facecolor=Config.COLORS['SEIPR'], 
                            alpha=0.15,
                            edgecolor=Config.COLORS['SEIPR'], 
                            linewidth=2.5,
                            linestyle='--',
                            zorder=2)
            ax.add_patch(ellipse)
    
    # Add SIR constraint line if available
    if Config.INCLUDE_SIR:
        df_sir = df_all[df_all['dataset'] == 'SIR']
        if len(df_sir) > 0:
            # Plot sample points
            sample_idx = np.random.choice(len(df_sir), 
                                         min(500, len(df_sir)), 
                                         replace=False)
            ax.scatter(df_sir.iloc[sample_idx]['low_freq_power'],
                      df_sir.iloc[sample_idx]['high_freq_power'],
                      c=Config.COLORS['SIR'],
                      alpha=0.3,
                      s=30,
                      label='SIR (Simple)',
                      edgecolors='none',
                      marker='s',
                      zorder=1)
            
            # Draw diagonal constraint
            x_vals = np.array([0.2, 0.7])
            y_vals = np.array([0.5, 0.1])
            ax.plot(x_vals, y_vals, 'r--', linewidth=2.5, 
                   label='SIR constraint', alpha=0.8, zorder=2)
    
    ax.set_xlabel('Low Frequency Power (<0.1 cycles/day, >10 day periods)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('High Frequency Power (>0.3 cycles/day, <3.3 day periods)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Frequency Distribution Comparison', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.8)
    ax.legend(loc='upper right', 
            framealpha=0.95, 
            fontsize=10,
            labelspacing=1.2,      # Vertical space between entries (default 0.5)
            borderpad=0.8,          # Padding inside legend border (default 0.4)
            handletextpad=0.6,
            label=disease_labels[disease])      # Space between marker and text (default 0.8)    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ==================== FIGURE 2: DOMINANT PERIOD DISTRIBUTIONS ====================

def create_figure2(df_all):
    """
    Create Figure 2: Dominant Period Distributions
    Three panel figure matching the style from the PDFs
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Disease Dominant Periods (Boxplots)
    ax = axes[0]
    
    disease_data = []
    disease_labels = []
    positions = []
    
    datasets = []
    if Config.INCLUDE_COVID:
        datasets.append('COVID')
    if Config.INCLUDE_FLU:
        datasets.append('Flu')
    if Config.INCLUDE_MPOX:
        datasets.append('Mpox')
    if Config.INCLUDE_COVID_COUNTY:
        datasets.append('COVID_county')
    
    for i, disease in enumerate(datasets):
        df_disease = df_all[df_all['dataset'] == disease]
        if len(df_disease) > 0:
            periods = df_disease['dominant_period'].values
            periods = periods[np.isfinite(periods) & (periods > 0) & (periods < 30)]
            
            if len(periods) > 0:
                disease_data.append(periods)
                disease_labels.append(disease)
                positions.append(i)
    
    if disease_data:
        bp = ax.boxplot(disease_data, positions=positions, 
                       labels=disease_labels,
                       widths=0.5, 
                       patch_artist=True,
                       showfliers=False,
                       medianprops=dict(color='black', linewidth=2),
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        for patch, label in zip(bp['boxes'], disease_labels):
            patch.set_facecolor(Config.COLORS[label])
            patch.set_alpha(0.7)
        
        # Add expected period annotations
        '''
        for i, (disease, pos) in enumerate(zip(disease_labels, positions)):
            if disease in Config.DISEASE_PERIODS:
                expected = Config.DISEASE_PERIODS[disease]
                ax.axhline(y=expected, xmin=(pos)/len(positions)/1.5, 
                          xmax=(pos+1)/len(positions)/1.5,
                          color=Config.COLORS[disease], 
                          linestyle=':', linewidth=2, alpha=0.6)
                ax.text(pos, expected + 0.5, f'Expected ({expected}d)', 
                       fontsize=8, ha='center')
        '''
    
    ax.set_ylabel('Dominant Period (days)', fontsize=11, fontweight='bold')
    ax.set_title('A: Disease Dominant Periods', fontsize=12, fontweight='bold')
    ax.autoscale(enable=True, axis='y')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: SIR Training Distribution
    ax = axes[1]
    
    if Config.INCLUDE_SIR:
        df_sir = df_all[df_all['dataset'] == 'SIR']
        if len(df_sir) > 0:
            periods = df_sir['dominant_period'].values
            periods = periods[np.isfinite(periods) & (periods > 0) & (periods < 30)]
            
            ax.hist(periods, bins=25, alpha=0.7, color=Config.COLORS['SIR'],
                   edgecolor='black', linewidth=0.5, density=True)
            
            # Overlay disease expected periods
            '''
            for disease in datasets:
                if disease in Config.DISEASE_PERIODS:
                    expected = Config.DISEASE_PERIODS[disease]
                    ax.axvline(expected, color=Config.COLORS[disease],
                             linestyle='--', linewidth=2, alpha=0.7,
                             label=f'{disease} ({expected}d)')
            '''
    
    ax.set_xlabel('Period (days)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title('B: SIR Training Distribution', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 30)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Panel C: SEIPR Training Distribution
    ax = axes[2]
    
    if Config.INCLUDE_SEIPR:
        df_seipr = df_all[df_all['dataset'] == 'SEIPR']
        if len(df_seipr) > 0:
            periods = df_seipr['dominant_period'].values
            periods = periods[np.isfinite(periods) & (periods > 0) & (periods < 30)]
            
            ax.hist(periods, bins=25, alpha=0.7, color=Config.COLORS['SEIPR'],
                   edgecolor='black', linewidth=0.5, density=True)
            
            # Overlay disease expected periods
            '''
            for disease in datasets:
                if disease in Config.DISEASE_PERIODS:
                    expected = Config.DISEASE_PERIODS[disease]
                    ax.axvline(expected, color=Config.COLORS[disease],
                             linestyle='--', linewidth=2, alpha=0.7,
                             label=f'{disease} ({expected}d)')
            '''
    ax.set_xlabel('Period (days)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title('C: SEIPR Training Distribution', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 30)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ==================== FIGURE 3: PEAK SHARPNESS COMPARISON ====================

def create_figure3(df_all):
    """
    Create Figure 3: Peak Sharpness Comparison
    Matching the distribution comparison style from the PDFs
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect data
    datasets_to_plot = []
    labels = []
    colors = []
    
    disease_list = []
    if Config.INCLUDE_COVID:
        disease_list.append('COVID')
    if Config.INCLUDE_FLU:
        disease_list.append('Flu')
    if Config.INCLUDE_MPOX:
        disease_list.append('Mpox')
    if Config.INCLUDE_COVID_COUNTY:
        disease_list.append('COVID_county')
    
    # Add diseases
    for disease in disease_list:
        df_disease = df_all[df_all['dataset'] == disease]
        if len(df_disease) > 0:
            sharpness = df_disease['peak_sharpness'].values
            sharpness = sharpness[np.isfinite(sharpness) & (sharpness > 0) & (sharpness < 50)]
            
            if len(sharpness) > 0:
                datasets_to_plot.append(sharpness)
                labels.append(disease)
                colors.append(Config.COLORS[disease])
    
    # Add models
    if Config.INCLUDE_SIR:
        df_sir = df_all[df_all['dataset'] == 'SIR']
        if len(df_sir) > 0:
            sharpness = df_sir['peak_sharpness'].values
            sharpness = sharpness[np.isfinite(sharpness) & (sharpness > 0) & (sharpness < 50)]
            
            if len(sharpness) > 0:
                datasets_to_plot.append(sharpness)
                labels.append('SIR (Simple)')
                colors.append(Config.COLORS['SIR'])
    
    if Config.INCLUDE_SEIPR:
        df_seipr = df_all[df_all['dataset'] == 'SEIPR']
        if len(df_seipr) > 0:
            sharpness = df_seipr['peak_sharpness'].values
            sharpness = sharpness[np.isfinite(sharpness) & (sharpness > 0) & (sharpness < 50)]
            
            if len(sharpness) > 0:
                datasets_to_plot.append(sharpness)
                labels.append('SEIPR (Complex)')
                colors.append(Config.COLORS['SEIPR'])
    
    # Create violin plots
    if datasets_to_plot:
        positions = np.arange(len(datasets_to_plot))
        
        parts = ax.violinplot(datasets_to_plot, positions=positions,
                             showmeans=False, showmedians=True, 
                             widths=0.7)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Style the other elements
        parts['cmedians'].set_edgecolor('black')
        parts['cmedians'].set_linewidth(2)
        parts['cbars'].set_edgecolor('black')
        parts['cbars'].set_linewidth(1.5)
        parts['cmins'].set_edgecolor('black')
        parts['cmins'].set_linewidth(1.5)
        parts['cmaxes'].set_edgecolor('black')
        parts['cmaxes'].set_linewidth(1.5)
        
        # Add median value labels
        for i, (data, label) in enumerate(zip(datasets_to_plot, labels)):
            median_val = np.median(data)
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Add text annotation
            ax.text(i, max(data) * 1.05, 
                   f'Median: {median_val:.1f}\nMean: {mean_val:.1f}±{std_val:.1f}',
                   ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=15, ha='right')
    
    ax.set_ylabel('Peak Sharpness', fontsize=12, fontweight='bold')
    ax.set_title('Peak Sharpness Comparison: Diseases vs Models', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation
    ax.text(0.02, 0.98, 
           'Higher sharpness = More periodic signal\nLower sharpness = Less periodic signal',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("SPECTRAL ANALYSIS OF DISEASE DYNAMICS")
    print("Comparative Study: COVID-19, Influenza, Mpox vs SIR/SEIPR Models")
    print("=" * 80)
    
    # Create output directory
    Config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


    
    # Load all datasets
    all_dataframes = []
    
    # Load diseases
    if Config.INCLUDE_COVID:
        covid_resids, covid_files = load_disease_residuals('COVID', Config.BASE_PATH)
        if covid_resids:
            df_covid = analyze_dataset(covid_resids, 'COVID', covid_files)
            all_dataframes.append(df_covid)
    
    if Config.INCLUDE_FLU:
        flu_resids, flu_files = load_disease_residuals('Flu', Config.BASE_PATH)
        if flu_resids:
            df_flu = analyze_dataset(flu_resids, 'Flu', flu_files)
            all_dataframes.append(df_flu)
    
    if Config.INCLUDE_MPOX:
        mpox_resids, mpox_files = load_disease_residuals('mpox', Config.BASE_PATH)
        if mpox_resids:
            df_mpox = analyze_dataset(mpox_resids, 'Mpox', mpox_files)
            all_dataframes.append(df_mpox)

    if Config.INCLUDE_COVID_COUNTY:
        covid_county_resids, covid_county_files = load_disease_residuals('COVID_county', Config.BASE_PATH)
        if covid_county_resids:
            df_covid_county = analyze_dataset(covid_county_resids, 'COVID_county', covid_county_files)
            all_dataframes.append(df_covid_county)
    
    # Load models
    if Config.INCLUDE_SIR:
        sir_resids = load_sir_residuals(Config.SIR_PATH, Config.SIR_SAMPLE_SIZE)
        if sir_resids:
            df_sir = analyze_dataset(sir_resids, 'SIR')
            all_dataframes.append(df_sir)
    
    if Config.INCLUDE_SEIPR:
        seipr_resids = load_seipr_residuals(Config.SEIPR_PATH, 
                                           Config.SEIPR_SAMPLE_SIZE)
        if seipr_resids:
            df_seipr = analyze_dataset(seipr_resids, 'SEIPR')
            all_dataframes.append(df_seipr)
    
    disease_display_labels = Config.disease_display_labels


    # Check if we have data
    if not all_dataframes:
        print("\n" + "=" * 80)
        print("ERROR: No data was loaded successfully!")
        print("=" * 80)
        print("\nPlease check:")
        print("1. File paths in Config class are correct")
        print("2. CSV files exist in specified directories")
        print("3. CSV files contain valid residual data")
        return
    
    # Combine all data
    df_all = pd.concat(all_dataframes, ignore_index=True)

    # ==================== SILHOUETTE ANALYSIS ====================

    
    results = run_silhouette_analysis(df_all)

    print("\n" + "="*80)
    print("SILHOUETTE RESULTS")
    print("="*80)

    print(f"Global regime separation: {results['regime_global']:.4f}")
    print(f"Mean regime (ALL): {results['mean_regime_all']:.4f}")
    print(f"Mean regime (TC): {results['mean_regime_tc']:.4f}")

    print(f"\nGlobal disease separation: {results['disease_global']:.4f}")

    print(f"\nMean pairwise disease (ALL): {results['mean_pairwise_all']:.4f}")
    print(f"Mean pairwise disease (TC): {results['mean_pairwise_tc']:.4f}")

    print("\nRATIOS")
    print(f"ALL: {results['ratio_all']:.4f}")
    print(f"TC : {results['ratio_tc']:.4f}")

    sil_path = Config.OUTPUT_DIR / "silhouette_raw_results.txt"

    with open(sil_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SILHOUETTE RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Global regime separation: {results['regime_global']:.4f}\n")
        f.write(f"Mean regime (ALL): {results['mean_regime_all']:.4f}\n")

        f.write(f"Global disease separation: {results['disease_global']:.4f}\n\n")
        f.write(f"Mean pairwise disease (ALL): {results['mean_pairwise_all']:.4f}\n")
        f.write(f"Mean pairwise disease (TC): {results['mean_pairwise_tc']:.4f}\n\n")

        f.write("RATIOS\n")
        f.write(f"ALL: {results['ratio_all']:.4f}\n")
        f.write(f"TC : {results['ratio_tc']:.4f}\n")

    # -------------------------------
    # Aggregate dominant period by dataset (ignoring forcing types)
    # -------------------------------
    dominant_period_summary = (
        df_all
        .groupby("dataset", as_index=False)
        .agg(
            dominant_period_mean=("dominant_period", "mean"),
            dominant_period_median=("dominant_period", "median"),
            dominant_period_std=("dominant_period", "std"),
            dominant_period_q25=("dominant_period", lambda x: x.quantile(0.25)),
            dominant_period_q75=("dominant_period", lambda x: x.quantile(0.75)),
            n_series=("dominant_period", "count")
        )
    )

    # Save to CSV
    output_path = Config.OUTPUT_DIR / "dominant_period_aggregated_by_dataset.csv"
    dominant_period_summary.to_csv(output_path, index=False)
    print(f"Saved dominant period summary by dataset to: {output_path}")

    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"\nTotal time series analyzed: {len(df_all)}")
    print("\nBreakdown by dataset:")
    print(df_all.groupby('dataset').size().to_string())
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    summary = df_all.groupby('dataset').agg({
        'low_freq_power': ['mean', 'std', 'median'],
        # 'mid_freq_power': ['mean', 'std', 'median'],
        'high_freq_power': ['mean', 'std', 'median'],
        'dominant_period': ['mean', 'std', 'median'],
        'peak_sharpness': ['mean', 'std', 'median']
    }).round(4)
    
    # Flatten MultiIndex columns
    summary.columns = [
        f"{feature}_{stat}" for feature, stat in summary.columns
    ]

    summary = summary.reset_index()

    # Pretty print to terminal
    print("\n=== SPECTRAL SUMMARY ===")
    print(summary.to_string(index=False))

    # Save clean CSV
    summary_path = Config.OUTPUT_DIR / 'summary_spectral_statistics.csv'
    summary.to_csv(summary_path, index=False)

    print(f"\nSummary statistics saved to: {summary_path}")
    
    # Compute OOD metrics (add after line ~900, before generating figures)
    df_ood = None
    if Config.INCLUDE_SEIPR:
        try:
            df_ood = compute_ood_metrics(df_all)
        except Exception as e:
            print(f"Error computing OOD metrics: {e}")
            import traceback
            traceback.print_exc()

    # Generate figures
    
    try:
        print("\nCreating Figure 1: Spectral Signatures...")
        fig1 = create_figure1(df_all)
        fig1_path = Config.OUTPUT_DIR / f'spectral_analysis_disease_SEIPR.{Config.FIGURE_FORMAT}'
        fig1.savefig(fig1_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {fig1_path}")
        plt.close(fig1)
    except Exception as e:
        print(f"  Error creating Figure 1: {e}")
    
    try:
        print("\nCreating Figure 2: Dominant Period Distributions...")
        fig2 = create_figure2(df_all)
        fig2_path = Config.OUTPUT_DIR / f'figure2_period_distributions.{Config.FIGURE_FORMAT}'
        fig2.savefig(fig2_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {fig2_path}")
        plt.close(fig2)
    except Exception as e:
        print(f"  Error creating Figure 2: {e}")
    
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {Config.OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print(f"  - figure1_spectral_signatures.{Config.FIGURE_FORMAT}")
    print(f"  - figure2_period_distributions.{Config.FIGURE_FORMAT}")
    print(f"  - figure3_peak_sharpness.{Config.FIGURE_FORMAT}")
    print(f"  - summary_statistics.csv")
    print()

    # Generate disease comparison table
    print("\n" + "=" * 80)
    print("CREATING DISEASE COMPARISON TABLE")
    print("=" * 80)

    # Add after Figure 1 generation
    if df_ood is not None and Config.INCLUDE_SEIPR:
        try:
            print("\nCreating OOD Visualization...")
            fig_ood = create_ood_visualization(df_all, df_ood, disease_display_labels)
            ood_path = Config.OUTPUT_DIR / f'figure_ood_quantification.{Config.FIGURE_FORMAT}'
            fig_ood.savefig(ood_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
            print(f"  Saved: {ood_path}")
            plt.close(fig_ood)
        except Exception as e:
            print(f"  Error creating OOD visualization: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        table_path = Config.OUTPUT_DIR / f'disease_comparison_table.{Config.FIGURE_FORMAT}'
        fig_table, df_stats = create_disease_comparison_table(df_all, table_path)
        plt.close(fig_table)
        
        # Also save the statistics as CSV
        stats_csv_path = Config.OUTPUT_DIR / 'disease_statistics.csv'
        df_stats.to_csv(stats_csv_path, index=False)
        print(f"Statistics saved to: {stats_csv_path}")
        
    except Exception as e:
        print(f"  Error creating comparison table: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()