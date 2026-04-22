#!/usr/bin/env python
# coding: utf-8

"""
Visualize learned feature representations to prove model learns 
universal dynamics, not disease-specific patterns
Developed by Maya Rangarajan 2026
Based on research: Hinton et al

Updated to include peak diversity marker, difference based heatmap

"""

import os
import io, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy.spatial.distance import cdist
# import tensorflow as tf
# from tensorflow.keras.models import load_model, Model

import tf_keras
from tf_keras.models import load_model, Model
from tensorflow.keras.layers import Concatenate, Lambda
from glob import glob

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

MODE = "report"   # "poster" or "report"

# ===== CONFIGURATION =====
classifier_length = 100
ts_len = 100

# Diseases to analyze
diseases = {
    'COVID Edmonton': {'file': 'COVID', 'color': '#C63A3C'},
    'Mpox': {'file': 'mpox', 'color': '#2B80D4'},
    'Influenza': {'file': 'flu', 'color': '#388E3C'},
    'COVID U.S.': {'file': 'COVID_county', 'color': '#9B5BB9'},
    # 'SEIPR_MultNoise': {'file': 'SEIPR_MultNoise', 'color': '#9B5BB9'},
    # 'SEIPR_AddNoise': {'file': 'SEIPR_AddNoise', 'color': '#9B5BB9'}
}

# List of diseases for calculations
disease_list_all = list(diseases.keys())
disease_list_no_flu = [d for d in diseases.keys() if d != 'Influenza']


# Load trained model
model_path = f'./trained_GEN-DL_models/best_model_1_1_length{ts_len}.keras'
full_model = load_model(model_path, compile=False)

# ===== EXTRACT FEATURE EXTRACTOR =====
# Get features from layer 25 (concatenate layer - before final classification)
#

# Print all layers so you can verify
# for smaller model
for i, layer in enumerate(full_model.layers):
    print(f"Layer {i}: {layer.name} — {layer.output.shape}")


feature_extractor = Model(
    inputs=full_model.input,
    outputs=full_model.layers[7].output  # lstm_2 — 32-dim features
)


print(f"Feature extractor output shape: {feature_extractor.output.shape}")

# ===== LOAD AND PROCESS DATA =====


def load_and_process_sequence(filepath):
    """Load residual time series and pad"""
    df = pd.read_csv(filepath).dropna()
    
    resids = df['residuals'].values
    seq_len = len(resids)
    
    # Create padded array
    temp_ts = np.zeros((1, ts_len, 1))
    
    if seq_len >= ts_len:
        # Take last 100 points
        temp_ts[0, :, 0] = resids[-ts_len:]
    else:
        # Pad at beginning with zeros, put data at end
        ts_gap = ts_len - seq_len
        temp_ts[0, ts_gap:, 0] = resids
    
    # Normalize (same as training and testing)
    values_avg = 0.0
    count_avg = 0
    for j in range(ts_len):
        if temp_ts[0, j, 0] != 0:
            values_avg += abs(temp_ts[0, j, 0])
            count_avg += 1
    
    if count_avg > 0:
        values_avg = values_avg / count_avg
        for j in range(ts_len):
            if temp_ts[0, j, 0] != 0:
                temp_ts[0, j, 0] = temp_ts[0, j, 0] / values_avg
    
    return temp_ts

# Collect all sequences and metadata
all_features = []
all_labels = []
all_diseases = []
all_types = []
all_raw_data = []

for disease_name, disease_info in diseases.items():
    print(f"\nProcessing {disease_name}...")
    # Dynamically pick all residual CSVs for this disease
    resids_files = glob(f'./{disease_info["file"]}/data/resids/resids*.csv')

    if len(resids_files) == 0:
        print(f"  No residuals files found")
        continue

    for filepath in resids_files:
        try:
            # Load and process sequence
            sequence = load_and_process_sequence(filepath)

            # Extract features using the model
            features = feature_extractor.predict(sequence, verbose=0)

            all_features.append(features[0])
            all_raw_data.append(sequence.flatten())
            
            label = 1 if 'forced' in os.path.basename(filepath).lower() else 0
            all_labels.append(label)
            all_diseases.append(disease_name)
        except: continue

# Convert to arrays
all_features = np.array(all_features)
all_labels = np.array(all_labels)
all_diseases = np.array(all_diseases)
all_types = np.array(all_types)
all_raw_data = np.array(all_raw_data)

X_feat = StandardScaler().fit_transform(all_features)
X_raw = StandardScaler().fit_transform(all_raw_data)

### Only Transcritical sequences for disease-invariant analysis ===
tc_mask = all_labels == 1
features_tc = all_features[tc_mask]
diseases_tc = all_diseases[tc_mask]
labels_tc = all_labels[tc_mask]

log_buffer = io.StringIO()
sys.stdout = log_buffer

report_lines = []

report_lines.append(f"\nTotal sequences collected: {len(all_features)}")
report_lines.append(f"Feature dimensionality: {all_features.shape[1]}")
report_lines.append(f"Transcritical: {np.sum(all_labels == 1)}, Nulls: {np.sum(all_labels == 0)}")


# Convert to arrays
X_feat = StandardScaler().fit_transform(np.array(all_features))
X_raw = StandardScaler().fit_transform(np.array(all_raw_data)) # For Point 4
y_label = np.array(all_labels)
y_disease = np.array(all_diseases)

# ======== CALCULATE SILHOUETTE SCORES =============

# null vs tc
score = silhouette_score(X_feat, all_labels)

# by disease
disease_score = silhouette_score(X_feat, all_diseases)

# not printing as it is overwhelmed by many more COVID U.S. points
# print(f"Silhouette  (Feature, Null vs TC): {score:.4f}")
# print(f"Silhouette (Feature, Disease Clusters): {disease_score:.4f}")

# ===== MEAN PER DISEASE NULL VS TC FEATURES SILHOUETTE =====
null_tc_per_disease = {}

os.makedirs("../../output_charts/latent_space_analysis/", exist_ok=True)

report_lines.append("\n" + "-"*60 + "\n")
report_lines.append("FEATURE SPACE REGIME SEPARATION")

for d_name in disease_list_all:
    mask = (all_diseases == d_name)
    X_subset = X_feat[mask]  # or X_raw for raw series
    y_subset = all_labels[mask]  # 0 = Null, 1 = TC
    
    # Only compute if both classes are present
    if len(np.unique(y_subset)) > 1:
        s = silhouette_score(X_subset, y_subset)
        null_tc_per_disease[d_name] = s
        report_lines.append(f"{d_name:15} Silhouette (Feature, per disease mean, Null vs TC): {s:.4f}")
    else:
        report_lines.append(f"{d_name:15} Skip (Only one class present)")

if len(null_tc_per_disease) > 0:
    mean_null_tc = np.mean(list(null_tc_per_disease.values()))
    report_lines.append(f"Mean per-disease Silhouette (Null vs TC): {mean_null_tc:.4f}")
else:
    mean_null_tc = np.nan
    report_lines.append("No valid diseases to compute mean silhouette")

mean_null_tc_no_flu = np.mean([v for k, v in null_tc_per_disease.items() if k != 'Influenza'])
report_lines.append(f"Mean per-disease Silhouette (Null vs TC, no Influenza): {mean_null_tc_no_flu:.4f}")
report_lines.append("\n" + "-"*60 + "\n")

# ====== AGGREGATE SILHOUETTE IN RAW SERIES ========

s_raw_null_tc = silhouette_score(X_raw, all_labels)  # all_labels is 0=Null, 1=TC
report_lines.append(f"Silhouette (Raw series, Null vs TC): {s_raw_null_tc:.4f}")

s_raw_disease = silhouette_score(X_raw, all_diseases)
report_lines.append(f"Silhouette (Raw series, Disease clusters): {s_raw_disease:.4f}")

# ===== MEAN PER DISEASE SILHOUETTE NULL VS TC (RAW SERIES) =====
null_tc_raw_per_disease = {}
report_lines.append("\n" + "-"*60 + "\n")
report_lines.append("RAW SPACE REGIME SEPARATION")

for d_name in disease_list_all:
    mask = (all_diseases == d_name)
    X_subset = X_raw[mask]       # Raw sequences
    y_subset = all_labels[mask]  # 0=Null, 1=TC

    if len(np.unique(y_subset)) > 1:
        s = silhouette_score(X_subset, y_subset)
        null_tc_raw_per_disease[d_name] = s
        report_lines.append(f"{d_name:15} Silhouette (Null vs TC, Raw): {s:.4f}")
    else:
        report_lines.append(f"{d_name:15} Skip (Only one class present)")

if len(null_tc_raw_per_disease) > 0:
    mean_null_tc_raw = np.mean(list(null_tc_raw_per_disease.values()))
    report_lines.append(f"Mean per-disease Silhouette (Null vs TC, Raw series): {mean_null_tc_raw:.4f}")
else:
    mean_null_tc_raw = np.nan
    report_lines.append("No valid diseases to compute mean silhouette")

mean_null_tc_raw_no_flu = np.mean([v for k, v in null_tc_raw_per_disease.items() if k != 'Influenza'])
report_lines.append(f"Mean per-disease Silhouette (Null vs TC, Raw series, no Influenza): {mean_null_tc_raw_no_flu:.4f}")
report_lines.append("\n" + "-"*60 + "\n")

from itertools import combinations

# List of diseases to include (optionally exclude Influenza later)
disease_list_all = list(diseases.keys())

# ===== PAIRWISE DISEASE SILHOUETTE (FEATURE SPACE) =====
pairwise_feat = {}

report_lines.append("FEATURE SPACE DISEASE SEPARATION")
for d1, d2 in combinations(disease_list_all, 2):
    mask = (all_diseases == d1) | (all_diseases == d2)
    X_subset = X_feat[mask]
    y_subset = all_diseases[mask]  # labels restricted to the two diseases

    # Compute silhouette only if both diseases have sequences
    if len(np.unique(y_subset)) == 2:
        s = silhouette_score(X_subset, y_subset)
        pairwise_feat[(d1, d2)] = s
        print(f"{d1:12} vs {d2:12} Silhouette (Feature): {s:.4f}")
    else:
        print(f"{d1:12} vs {d2:12} Skip (not enough sequences)")

# Mean across all pairs
mean_pairwise_feat = np.mean(list(pairwise_feat.values()))
report_lines.append(f"\nMean pairwise Silhouette (Feature space): {mean_pairwise_feat:.4f}")

# Mean excluding any pair with Influenza
mean_pairwise_feat_no_flu = np.mean([v for k, v in pairwise_feat.items() if 'Influenza' not in k])
report_lines.append(f"Mean pairwise Silhouette (Feature space, no Influenza): {mean_pairwise_feat_no_flu:.4f}")


# ===== PAIRWISE DISEASE SILHOUETTE (RAW SPACE) =====
report_lines.append("\n" + "-"*60 + "\n")
report_lines.append("RAW SPACE DISEASE SEPARATION")

pairwise_raw = {}

for d1, d2 in combinations(disease_list_all, 2):
    mask = (all_diseases == d1) | (all_diseases == d2)
    X_subset = X_raw[mask]
    y_subset = all_diseases[mask]  # labels restricted to the two diseases

    if len(np.unique(y_subset)) == 2:
        s = silhouette_score(X_subset, y_subset)
        pairwise_raw[(d1, d2)] = s
        report_lines.append(f"{d1:12} vs {d2:12} Silhouette (Raw): {s:.4f}")
    else:
        report_lines.append(f"{d1:12} vs {d2:12} Skip (not enough sequences)")

mean_pairwise_raw = np.mean(list(pairwise_raw.values()))
report_lines.append(f"\nMean pairwise Silhouette (Raw space): {mean_pairwise_raw:.4f}")

mean_pairwise_raw_no_flu = np.mean([v for k, v in pairwise_raw.items() if 'Influenza' not in k])
report_lines.append(f"Mean pairwise Silhouette (Raw space, no Influenza): {mean_pairwise_raw_no_flu:.4f}")

# Use existing variables (no influenza versions)
regime_raw = mean_null_tc_raw_no_flu
disease_raw = mean_pairwise_raw_no_flu

regime_feat = mean_null_tc_no_flu
disease_feat = mean_pairwise_feat_no_flu

# Compute ratios
ratio_raw = regime_raw / disease_raw
ratio_feat = regime_feat / disease_feat

# Compute shift
shift = ratio_feat / ratio_raw

# ===== PRINT =====
report_lines.append("\n" + "="*50)
report_lines.append("SHIFT IN LEARNED REPRESENTATION -- ALL")
report_lines.append("="*50)
report_lines.append("REGIME TO DISEASE SEPARATION RATIO")
report_lines.append(f"RAW SPACE:     {ratio_raw:.2f}")
report_lines.append(f"FEATURE SPACE: {ratio_feat:.2f}")

report_lines.append(f"\nShift in dominance toward regime: {shift:.2f}x")
report_lines.append("="*50)



from itertools import combinations

# ===== PAIRWISE DISEASE SILHOUETTE (FEATURE SPACE, TC only) =====
pairwise_feat = {}

report_lines.append("\n" + "-"*60 + "\n")
report_lines.append("FEATURE SPACE DISEASE SEPARATION TC ONLY")


for d1, d2 in combinations(disease_list_all, 2):
    mask = ((all_diseases == d1) | (all_diseases == d2)) & (all_labels == 1)  # only TC
    X_subset = X_feat[mask]
    y_subset = all_diseases[mask]

    if len(np.unique(y_subset)) == 2:
        s = silhouette_score(X_subset, y_subset)
        pairwise_feat[(d1, d2)] = s
        report_lines.append(f"{d1:12} vs {d2:12} Silhouette (Feature, TC only): {s:.4f}")
    else:
        report_lines.append(f"{d1:12} vs {d2:12} Skip (not enough sequences)")

mean_pairwise_feat = np.mean(list(pairwise_feat.values()))
report_lines.append(f"\nMean pairwise Silhouette (Feature space, TC only): {mean_pairwise_feat:.4f}")

mean_pairwise_feat_no_flu = np.mean([v for k, v in pairwise_feat.items() if 'Influenza' not in k])
report_lines.append(f"Mean pairwise Silhouette (Feature space, TC only, no Influenza): {mean_pairwise_feat_no_flu:.4f}")

report_lines.append("\n" + "-"*60 + "\n")

# ===== PAIRWISE DISEASE SILHOUETTE (RAW SPACE, TC only) =====
pairwise_raw = {}
report_lines.append("RAW SPACE DISEASE SEPARATION TC ONLY")

for d1, d2 in combinations(disease_list_all, 2):
    mask = ((all_diseases == d1) | (all_diseases == d2)) & (all_labels == 1)  # only TC
    X_subset = X_raw[mask]
    y_subset = all_diseases[mask]

    if len(np.unique(y_subset)) == 2:
        s = silhouette_score(X_subset, y_subset)
        pairwise_raw[(d1, d2)] = s
        report_lines.append(f"{d1:12} vs {d2:12} Silhouette (Raw, TC only): {s:.4f}")
    else:
        report_lines.append(f"{d1:12} vs {d2:12} Skip (not enough sequences)")

mean_pairwise_raw = np.mean(list(pairwise_raw.values()))
report_lines.append(f"\nMean pairwise Silhouette (Raw space, TC only): {mean_pairwise_raw:.4f}")

mean_pairwise_raw_no_flu = np.mean([v for k, v in pairwise_raw.items() if 'Influenza' not in k])
report_lines.append(f"Mean pairwise Silhouette (Raw space, TC only, no Influenza): {mean_pairwise_raw_no_flu:.4f}")

# ===== SIGNAL DOMINANCE (REGIME / DISEASE) =====

# Use existing variables (no influenza versions)
regime_raw = mean_null_tc_raw_no_flu
disease_raw = mean_pairwise_raw_no_flu

regime_feat = mean_null_tc_no_flu
disease_feat = mean_pairwise_feat_no_flu

# Compute ratios
ratio_raw = regime_raw / disease_raw
ratio_feat = regime_feat / disease_feat

# Compute shift
shift = ratio_feat / ratio_raw

# ===== PRINT =====
report_lines.append("\n" + "="*50)
report_lines.append("SHIFT IN LEARNED REPRESENTATION -- TC ONLY")
report_lines.append("="*50)
report_lines.append("REGIME TO DISEASE SEPARATION RATIO")
report_lines.append(f"RAW SPACE:     {ratio_raw:.2f}")
report_lines.append(f"FEATURE SPACE: {ratio_feat:.2f}")

report_lines.append(f"\nShift in dominance toward regime: {shift:.2f}x")
report_lines.append("="*50)

# ===== OUTPUT AND SAVE SILHOUETTE RESULTS =====
for line in report_lines:
    print(line)

os.makedirs("../../output_charts/latent_space_analysis/", exist_ok=True)
output_path = "../../output_charts/latent_space_analysis/tsne_analysis.csv"

# Save each line as a row
with open(output_path, "w") as f:
    for line in report_lines:
        f.write(line + "\n")

print(f"✓ Silhouette report saved to {output_path}")


# ========================== t-SNE CHART + HEAT MAP =========================

# ======== RUN T-SNE ========

tsne = TSNE(
    n_components=2,
    perplexity=100,
    learning_rate='auto',
    max_iter=5000,
    random_state=42,
    verbose=1
)

embedded = tsne.fit_transform(X_feat)
print(f"t-SNE embedding complete. Shape: {embedded.shape}")

theta = np.radians(0) 
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])
embedded = embedded @ rotation_matrix.T

# Normalize to [-1, 1]
from sklearn.preprocessing import MinMaxScaler

scaler_embed = MinMaxScaler(feature_range=(-1, 1))
embedded = scaler_embed.fit_transform(embedded)

# ========================== t-SNE SCATTER + HEAT MAP (FIXED) =========================

# --- Configuration ---
BINS = 75
SATURATION_LIMIT = 5
MODE = "report"  # "poster" or "report"

# --- Define plot labels and diseases ---
plot_labels = all_labels      # Null=0, TC=1
plot_diseases = all_diseases  # Disease strings

# --- Split points ---
null_points = embedded[plot_labels == 0]
tc_points   = embedded[plot_labels == 1]

# After embedding, create covid_plot_mask for scatter only
covid_mask = plot_diseases == 'COVID U.S.'
covid_null_indices = np.where(covid_mask & (plot_labels == 0))[0]
covid_tc_indices   = np.where(covid_mask & (plot_labels == 1))[0]

num_pairs = min(500, len(covid_null_indices), len(covid_tc_indices))
pair_indices = np.random.choice(num_pairs, size=num_pairs, replace=False)

sample_null_indices = covid_null_indices[pair_indices]
sample_tc_indices   = covid_tc_indices[pair_indices]

covid_plot_mask = np.zeros(len(plot_labels), dtype=bool)
covid_plot_mask[sample_null_indices] = True
covid_plot_mask[sample_tc_indices] = True


# ======== 2DENSITY HEATMAP ========
if MODE == "poster":
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
else:
    fig2, ax2 = plt.subplots(figsize=(12, 10))

ax2.set_facecolor('#E8E8F0')

''' 
# Null density
ax2.hist2d(
    null_points[:, 0], null_points[:, 1],
    bins=BINS,
    cmap="Blues",
    alpha=0.75,
    cmin=1,
    vmax=SATURATION_LIMIT
)

# TC density
ax2.hist2d(
    tc_points[:, 0], tc_points[:, 1],
    bins=BINS,
    cmap="Reds",
    alpha=0.75,
    cmin=1,
    vmax=SATURATION_LIMIT
)
'''
# ======== ADDED: DIVERGING DIFFERENCE MAP (replaces hist2d calls) ========
from scipy.ndimage import gaussian_filter

x_range_lat = [embedded[:, 0].min(), embedded[:, 0].max()]
y_range_lat = [embedded[:, 1].min(), embedded[:, 1].max()]

null_hist_lat, xedges_lat, yedges_lat = np.histogram2d(
    null_points[:, 0], null_points[:, 1],
    bins=BINS, range=[x_range_lat, y_range_lat]
)
tc_hist_lat, _, _ = np.histogram2d(
    tc_points[:, 0], tc_points[:, 1],
    bins=BINS, range=[x_range_lat, y_range_lat]
)

# Normalize
null_hist_lat = null_hist_lat / null_hist_lat.sum()
tc_hist_lat   = tc_hist_lat   / tc_hist_lat.sum()

# Difference map
diff_lat = tc_hist_lat - null_hist_lat

# Mask bins with no data so they show as background color
no_data_mask_lat = (null_hist_lat == 0) & (tc_hist_lat == 0)
diff_lat_masked = np.ma.masked_where(no_data_mask_lat, diff_lat)

im2 = ax2.pcolormesh(
    xedges_lat, yedges_lat, diff_lat_masked.T,
    cmap='RdBu_r',
    vmin=-np.abs(diff_lat).max(),
    vmax=np.abs(diff_lat).max()
)
# ======== END DIVERGING DIFFERENCE MAP ========

# ======== ADDED: PEAK DENSITY MARKERS ========
null_smooth_lat = gaussian_filter(null_hist_lat, sigma=2.0)
tc_smooth_lat   = gaussian_filter(tc_hist_lat,   sigma=2.0)

x_centers_lat = (xedges_lat[:-1] + xedges_lat[1:]) / 2
y_centers_lat = (yedges_lat[:-1] + yedges_lat[1:]) / 2

null_peak_idx_lat = np.unravel_index(null_smooth_lat.argmax(), null_smooth_lat.shape)
tc_peak_idx_lat   = np.unravel_index(tc_smooth_lat.argmax(),   tc_smooth_lat.shape)

null_peak_x_lat = x_centers_lat[null_peak_idx_lat[0]]
null_peak_y_lat = y_centers_lat[null_peak_idx_lat[1]]
tc_peak_x_lat   = x_centers_lat[tc_peak_idx_lat[0]]
tc_peak_y_lat   = y_centers_lat[tc_peak_idx_lat[1]]

ax2.scatter(null_peak_x_lat, null_peak_y_lat,
            color='#1a5fa8', marker='*', s=1000,
            zorder=20, edgecolors='white', linewidth=1.5,
            label='Null peak density')
ax2.scatter(tc_peak_x_lat, tc_peak_y_lat,
            color='#8b0000', marker='*', s=1000,
            zorder=20, edgecolors='white', linewidth=1.5,
            label='TC peak density')

# ax2.legend(fontsize=10, framealpha=0.9)
# ======== END PEAK DENSITY MARKERS ========

# styling
# ax2.set_xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
# ax2.set_ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
# ax2.set_title('Density of Null vs Transcritical Sequences', fontsize=16)
ax2.tick_params(axis='both', labelsize=12)
ax2.grid(alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# for normalized scale
ax2.set_xlim(-1.05, 1.05)
ax2.set_ylim(-1.05, 1.05)

plt.tight_layout()
plt.savefig('../../output_charts/latent_space_analysis/latent_space_tsne.png', dpi=600, bbox_inches='tight')
plt.savefig('../../output_charts/latent_space_analysis/latent_space_tsne.pdf', bbox_inches='tight')
print("✓ Latent space tsne chart saved")
plt.show()

# ======== RAW SPACE T-SNE ========

tsne_raw = TSNE(
    n_components=2,
    perplexity=100,
    learning_rate='auto',
    max_iter=5000,
    random_state=42,
    verbose=1
)

embedded_raw = tsne_raw.fit_transform(X_raw)
print(f"Raw space t-SNE embedding complete. Shape: {embedded_raw.shape}")

theta = np.radians(0)
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])
embedded_raw = embedded_raw @ rotation_matrix.T

# Normalize to [-1, 1] using fresh scaler
scaler_embed_raw = MinMaxScaler(feature_range=(-1, 1))
embedded_raw = scaler_embed_raw.fit_transform(embedded_raw)

# --- Split points ---
raw_null_points = embedded_raw[plot_labels == 0]
raw_tc_points   = embedded_raw[plot_labels == 1]

# ======== RAW SPACE DENSITY HEATMAP ========
if MODE == "poster":
    fig3, ax3 = plt.subplots(figsize=(5.5, 5.5))
else:
    fig3, ax3 = plt.subplots(figsize=(12, 10))

ax3.set_facecolor('#E8E8F0')

# Null density
ax3.hist2d(
    raw_null_points[:, 0], raw_null_points[:, 1],
    bins=BINS,
    cmap="Blues",
    alpha=0.75,
    cmin=1,
    vmax=SATURATION_LIMIT
)

# TC density
ax3.hist2d(
    raw_tc_points[:, 0], raw_tc_points[:, 1],
    bins=BINS,
    cmap="Reds",
    alpha=0.75,
    cmin=1,
    vmax=SATURATION_LIMIT
)

# --- peak density marker
from scipy.ndimage import gaussian_filter

x_range = [embedded_raw[:, 0].min(), embedded_raw[:, 0].max()]
y_range = [embedded_raw[:, 1].min(), embedded_raw[:, 1].max()]

null_hist, xedges, yedges = np.histogram2d(
    raw_null_points[:, 0], raw_null_points[:, 1],
    bins=BINS, range=[x_range, y_range]
)
tc_hist, _, _ = np.histogram2d(
    raw_tc_points[:, 0], raw_tc_points[:, 1],
    bins=BINS, range=[x_range, y_range]
)

# Normalize -- difference calculation
null_hist = null_hist / null_hist.sum()
tc_hist   = tc_hist   / tc_hist.sum()

# Difference map
diff = tc_hist - null_hist

# Mask bins with no data so they show as background color
no_data_mask_raw = (null_hist == 0) & (tc_hist == 0)
diff_masked = np.ma.masked_where(no_data_mask_raw, diff)

im = ax3.pcolormesh(
    xedges, yedges, diff_masked.T,
    cmap='RdBu_r',
    vmin=-np.abs(diff).max(),
    vmax=np.abs(diff).max()
)
# ---- end difference calculation

null_smooth = gaussian_filter(null_hist, sigma=2.0)
tc_smooth   = gaussian_filter(tc_hist,   sigma=2.0)

x_centers = (xedges[:-1] + xedges[1:]) / 2
y_centers = (yedges[:-1] + yedges[1:]) / 2

null_peak_idx = np.unravel_index(null_smooth.argmax(), null_smooth.shape)
tc_peak_idx   = np.unravel_index(tc_smooth.argmax(),   tc_smooth.shape)

null_peak_x = x_centers[null_peak_idx[0]]
null_peak_y = y_centers[null_peak_idx[1]]
tc_peak_x   = x_centers[tc_peak_idx[0]]
tc_peak_y   = y_centers[tc_peak_idx[1]]

ax3.scatter(null_peak_x, null_peak_y,
            color='#1a5fa8', marker='*', s=1000,
            zorder=20, edgecolors='white', linewidth=1.5,
            label='Null peak density')
ax3.scatter(tc_peak_x, tc_peak_y,
            color='#8b0000', marker='*', s=1000,
            zorder=20, edgecolors='white', linewidth=1.5,
            label='TC peak density')

# --- end peak density marker

ax3.tick_params(axis='both', labelsize=12)
ax3.grid(alpha=0.3, linestyle='--')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# for normalized scale
ax3.set_xlim(-1.05, 1.05)
ax3.set_ylim(-1.05, 1.05)

plt.tight_layout()
plt.savefig('../../output_charts/latent_space_analysis/raw_space_tsne.png', dpi=600, bbox_inches='tight')
plt.savefig('../../output_charts/latent_space_analysis/raw_space_tsne.pdf', bbox_inches='tight')
print("✓ Raw space tsne chart saved")
plt.show()