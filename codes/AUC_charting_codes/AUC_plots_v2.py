"""
Code developed by Maya Rangarajan, 2026

FUNCTION: PRINT BAR GRAPHS FOR REPORT
HOW TO USE: 
RUN FROM CODES/TRAINING_DATA_AND_TESTING FOLDER
1. UPDATE AUC VALUES MANUALLY
2. SET MODE = 'report'

Consolidated AUC pdf saved in ./output_charts

"""

import pandas as pd
import plotly.express as px
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import wrap

# ============= POSTER STYLING =============
import matplotlib as mpl

MODE = "report" # size fonts appropriately

# Global font scaling for 23.5" wide poster
mpl.rcParams.update({
    'font.size':          18,      # base font size
    'axes.titlesize':     22,      # chart title / header
    'axes.labelsize':     20,      # x and y axis labels
    'xtick.labelsize':    16,      # x tick labels
    'ytick.labelsize':    16,      # y tick labels
    'legend.fontsize':    12,      # legend text
    'figure.titlesize':   24,      # suptitle
    'font.weight':        'normal',
    'axes.titleweight':   'bold',
    'axes.labelweight':   'bold',
})

def apply_poster_fonts(ax, title_size=26, label_size=24, tick_size=17, annot_size=16):
    ax.title.set_fontsize(title_size)
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(label_size)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontweight('bold')
    ax.tick_params(axis='both', labelsize=tick_size)
    
    # Override all text objects in the axes (bar labels, annotations etc.)
    for text in ax.texts:
        text.set_fontsize(annot_size)
    
    # Override legend if present
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(label_size)

POSTER_ACCENT = '#0E0E0E'   # warm gold — complements the blue poster header
POSTER_GRAY   = '#F5F5F5'   # panel background
POSTER_BORDER = '#CCCCCC'   # spine/border color

def add_dataset_divider(ax, n_synthetic, y_min, y_max):
    """Vertical divider line with callout labels inside the chart."""
    divider_x = n_synthetic - 0.5

    # Vertical dashed line
    ax.axvline(x=divider_x, color='#888888', linestyle='--',
               linewidth=1.2, alpha=0.7, zorder=5)

    # Labels inside chart, near the top
    label_y = y_max * 0.99

    '''
    ax.text(divider_x - 0.15, label_y, 'Synthetic',
            ha='right', va='top',
            fontsize=14,
            color='#0E0E0E', clip_on=False)

    ax.text(divider_x + 0.15, label_y, 'Real-world',
            ha='left', va='top',
            fontsize=14,
            color='#0E0E0E', clip_on=False)
    '''
    

def apply_poster_style(ax, title=None):
    """Apply consistent poster styling to any axis."""
    ax.set_facecolor(POSTER_GRAY)
    for spine in ax.spines.values():
        spine.set_edgecolor(POSTER_BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(colors='#333333', labelsize=14)
    ax.yaxis.label.set_color('#333333')
    ax.xaxis.label.set_color('black')
    ax.grid(axis='y', color='white', linestyle='-', linewidth=1.2, alpha=1.0, zorder=0)
    if title:
        ax.set_title(title, fontsize=18, fontweight='bold', color=POSTER_ACCENT, pad=30)

# ============= AUC ROC Data =============


data = {
    "Model": ["LAG1-AC", "VARIANCE", "SIDATR", "GEN-DL"],
    "SEIPR White Noise": [0.2754,0.2786,0.6104,0.9303],
    "SEIPR Environmental Noise": [0.5735,0.1818,0.4930,0.9403],
    "COVID Edmonton": [0.3425, 0.2675, 0.7065, 0.8234],
    "Influenza": [0.4161, 0.4208, 0.4800, 0.7412],
    "mpox": [0.6833, 0.4654, 0.7910, 0.9532],
    "COVID U.S.": [0.5312,0.2357,0.6880,0.6468]
}

DATASET_DISPLAY_NAMES = {
    'SEIPR White Noise': 'SEIPR White Noise',
    'SEIPR Environmental Noise': 'SEIPR Env Noise',
    'COVID Edmonton': 'COVID Edmonton',
    'Influenza': 'Influenza', 
    'mpox': 'mpox',
    'COVID U.S.': 'COVID U.S.'
}

# Create full dataframe for bar graph (before filtering)
df_full = pd.DataFrame(data).set_index("Model")
df_full = df_full.apply(pd.to_numeric, errors='coerce')
desired_order = ["LAG1-AC", "VARIANCE", "SIDATR","GEN-DL"]
df = df_full.loc[desired_order]

output_folder = "../../output_charts"
os.makedirs(output_folder, exist_ok=True)

# Prepare data for plotting
df_reset = df_full.loc[desired_order].reset_index()
df_melted = df_reset.melt(id_vars='Model', var_name='Dataset', value_name='AUC')
df_melted = df_melted.dropna()
df_melted['AUC_delta'] = df_melted['AUC'] - 0.5

# Create PDF with chart
pdf_file = os.path.join(output_folder, "AUC_bargraphs_combined.pdf")

# ============= NATURE PUBLISHING COLOR PALETTE =============
# Dataset colors - used consistently across ALL charts
dataset_colors = {
    'SEIPR White Noise':          '#6B4C8B',   # muted purple — neutral enough
    'SEIPR Environmental Noise':  '#8B6B4C',   # warm gray-brown — neutral
    'COVID Edmonton':             '#891312',   # burgundy
    'COVID U.S.':            '#C0562A',   # burnt orange — acceptable warm accent
    'mpox':                       '#4C6B8B',   # slate blue-gray — cool, poster-compatible
    'Influenza':                  '#7A6B8B',   # dusty violet — muted, neutral
}

# Model colors - used consistently across ALL charts
model_colors = {
    'LAG1-AC': '#A8C4D8',   # muted amber
    'VARIANCE': '#5B8DB8',   # scientific blue
    'SIDATR': '#1E3F6B',     # deep teal
    'GEN-DL': '#891312'      # maroon (dominant)
}

if MODE == "report":
    mpl.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
    })

with PdfPages(pdf_file) as pdf:
    # ========== CHART 1: All Data Clustered by Model ==========

    fig1, ax = plt.subplots(figsize=(8.5, 5))  # report-friendly aspect ratio

    datasets_all = [
        'SEIPR White Noise',
        'SEIPR Environmental Noise',
        'COVID Edmonton',
        'COVID U.S.',
        'mpox'
    ]

    models = ['LAG1-AC', 'VARIANCE', 'SIDATR', 'GEN-DL']

    x = np.arange(len(models))
    width = 0.16  # wider bars = better readability in report

    for j, dataset in enumerate(datasets_all):
        dataset_data = df_melted[df_melted['Dataset'] == dataset]

        values = []
        for model in models:
            row = dataset_data[dataset_data['Model'] == model]
            values.append(row['AUC'].values[0] if not row.empty else np.nan)

        ax.bar(
            x + j * width,
            values,
            width,
            color=dataset_colors[dataset],
            alpha=0.9,
            label=dataset
        )

    # Axes styling (report clean)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_xticks(x + width * (len(datasets_all) - 1) / 2)
    ax.set_xticklabels(models)

    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)

    # 🔥 FIXED LEGEND (no squashing)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),  # BELOW plot
        ncol=3,
        frameon=False
    )

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    pdf.savefig(fig1, bbox_inches='tight')
    plt.close(fig1)

    # ========== CHART 2: Disease Dataset Performance (AUC - 0.5) ==========
    fig2 = plt.figure(figsize=(8, 6))
    
    # datasets_disease = ['COVID', 'Influenza', 'mpox', 'COVID-county']
    datasets_disease = ['COVID Edmonton', 'COVID U.S.', 'mpox', 'Influenza']
    models_selected = ['LAG1-AC', 'VARIANCE', 'SIDATR','GEN-DL']
    
    # Filter data for selected models and datasets
    df_melted_filtered = df_melted[
        (df_melted['Model'].isin(models_selected)) & 
        (df_melted['Dataset'].isin(datasets_disease))
    ]
    
    x = np.arange(len(datasets_disease))
    width = 0.20
    
    for i, model in enumerate(models_selected):
        model_data = df_melted_filtered[df_melted_filtered['Model'] == model]
        values = []
        positions = []
        for j, dataset in enumerate(datasets_disease):
            row = model_data[model_data['Dataset'] == dataset]
            if not row.empty:
                values.append(row['AUC_delta'].values[0])
                positions.append(j)
        
        plt.bar(np.array(positions) + i*width, values, width, label=model, 
                color=model_colors[model], alpha=0.9)
    
    plt.xlabel('Dataset', fontsize=13, fontweight='bold')
    plt.ylabel('Improvement over Random (AUC - 0.5)', fontsize=13, fontweight='bold')
    x_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in datasets_disease]
    plt.xticks(x + width * (len(models_selected) - 1) / 2, datasets_disease, fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    ax2 = fig1.gca()
    apply_poster_fonts(ax2)
    
    pdf.savefig(fig2, bbox_inches='tight')
    plt.close(fig2)    

#### FIG 12 FOR QUAD CHART ####################

    fig12, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)

    ROBUSTNESS_MODELS = ["LAG1-AC", "VARIANCE", "SIDATR", "GEN-DL"]
    DISPLAY_NAMES = {
        "LAG1-AC":  "LAG1-AC",
        "VARIANCE": "VARIANCE",
        "SIDATR":   "SIDATR",
        "GEN-DL":   "GEN-DL"
    }

    models             = ROBUSTNESS_MODELS
    disease_datasets   = ['COVID Edmonton', 'mpox', 'COVID U.S.','Influenza']
    synthetic_datasets = ['SEIPR White Noise', 'SEIPR Environmental Noise']

    spacing = 0.03
    x       = np.arange(len(models)) * spacing

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')

    WHISKER_COLOR = '#000000'

    for i, model in enumerate(models):
        model_idx = data['Model'].index(model)
        mcolor    = model_colors[model]

        syn_aucs = [data[d][model_idx] for d in synthetic_datasets]
        dis_aucs = [data[d][model_idx] for d in disease_datasets]
        all_aucs = syn_aucs + dis_aucs

        auc_min  = np.min(all_aucs)
        auc_max  = np.max(all_aucs)
        mean_auc = np.mean(all_aucs)
        xi       = i * spacing

        # GEN-DL column highlight
        if model == 'GEN-DL':
            half_width = spacing * 0.45
            ax.axvspan(
                xi - half_width - 0.001,
                xi + half_width + 0.001,
                color='#e8e8e8', alpha=0.6, zorder=1
            )

        # Range bar
        ax.vlines(xi, ymin=auc_min, ymax=auc_max,
                linewidth=1.5, color=WHISKER_COLOR, alpha=0.5, zorder=4)

        # Cap ticks — narrower at small size
        cap_width = spacing * 0.08
        for cap_y in [auc_min, auc_max]:
            ax.hlines(cap_y, xi - cap_width, xi + cap_width,
                    linewidth=1.5, color=WHISKER_COLOR, alpha=1.0, zorder=4)

        # Individual points
        ax.scatter(np.full(len(syn_aucs), xi), syn_aucs,
                color=mcolor, alpha=0.45, s=25, zorder=3, linewidths=0)
        ax.scatter(np.full(len(dis_aucs), xi), dis_aucs,
                color=mcolor, alpha=0.85, s=25, zorder=3, linewidths=0)

        # Mean dot
        ax.scatter(xi, mean_auc, color='#891312', s=150, zorder=5, linewidths=0)

        # Mean label — tighter offset at small scale
        if model == "LAG1-AC":
            ax.text(xi + 0.003, mean_auc, f"{mean_auc:.2f}",
                    ha='left', va='center', fontsize=20, fontweight='bold',
                    color='black', zorder=5, clip_on=False)
        else:
            ax.text(xi - 0.003, mean_auc, f"{mean_auc:.2f}",
                    ha='right', va='center', fontsize=20, fontweight='bold',
                    color='black', zorder=5, clip_on=False)

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(m, m) for m in models],
        rotation=0, ha='center', fontsize=14, fontweight='bold'
    )
    ax.tick_params(axis='x', which='both', length=0, pad=2)
    ax.tick_params(axis='y', labelsize=7, pad=2)

    ax.set_ylabel('AUC', fontsize=20, fontweight='bold',
                color='#333333', labelpad=6)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.7], minor=True)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_yticklabels(['0.7'], minor=True, fontsize=20, fontweight='bold')

    for label in ax.get_yticklabels(minor=True):
        label.set_fontweight('bold')

    ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.5, color='#999999')
    ax.axhline(0.7, color=WHISKER_COLOR, linestyle=':', linewidth=0.9, alpha=1.0)

    left  = x[0]  - spacing * 0.5
    right = x[-1] + spacing * 0.5
    ax.set_xlim(left, right)

    plt.tight_layout(pad=0.4)
    pdf.savefig(fig12, bbox_inches='tight')
    plt.close(fig12)

print(f"Combined PDF saved to: {pdf_file}")