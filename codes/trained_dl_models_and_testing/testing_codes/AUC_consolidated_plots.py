#!/usr/bin/env python
# coding: utf-8
"""
Combined ROC panel for empirical data (PNAS figure).

One panel per empirical test_model. Each panel shows:
  - 5 GEN-DL curves, one per training regime (fast, base, COVID, mpox, influenza)
  - Variance, Lag-1 AC, Chakraborty  (computed ONCE per test_model)

Assumed folder layout (per empirical disease):
  {test_model}/data/{regime}/df_ml_forced.csv , df_ml_null.csv     # GEN-DL regimes
  {test_model}/data/ml_pred_Chakraborty/df_ml_forced.csv , ...     # Chakraborty
  {test_model}/data/ews/df_ews_forced.csv , df_ews_null.csv        # timing
  {test_model}/data/ews/df_ktau_forced.csv , df_ktau_null.csv      # Variance / Lag-1 AC

Edited from Bury et al. (2021) PNAS and Chakraborty et al. (2025).
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TEST_MODELS = ['COVID', 'COVID_state', 'flu', 'mpox']   # panel order (2x2)

TITLES = {
    'flu': 'Influenza',
    'COVID': 'COVID Edmonton',
    'mpox': 'mpox',
    'COVID_state': 'COVID U.S.',
}

# GEN-DL training regimes  ->  subfolder name under {test_model}/data/
GENDL_REGIMES = ['fast', 'base', 'COVID', 'mpox', 'influenza']
CHAK_FOLDER = 'ml_pred_Chakraborty'

N_PREDICTIONS = 5
PRED_INTERVAL_REL = np.array([0.5, 1.0])
VAR = 'I'

# Style grammar:  solid = GEN-DL variants (our method) ; dashed = baselines.
# Okabe-Ito colourblind-safe palette.
GENDL_STYLE = {
    'fast':   dict(color='#0072B2'),
    'base':      dict(color='#E69F00'),
    'COVID':     dict(color='#009E73'),
    'mpox':      dict(color='#CC79A7'),
    'influenza': dict(color='#D55E00'),
}
BASELINE_STYLE = {
    'Variance':    dict(color='#1f77b4', linestyle="--"),   # matplotlib defaults, all solid
    'Lag-1 AC':    dict(color='#ff7f0e', linestyle="--"),
    'Chakraborty': dict(color='#2ca02c', linestyle="--"),
}
GENDL_LABEL = {r: f'GEN-DL (regime: {r})' for r in GENDL_REGIMES}

OUTDIR = Path('../../output_charts/ROC')
OUTDIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Data extraction / ROC helpers
# ----------------------------------------------------------------------
def roc_from(truth_vals, indicator_vals):
    fpr, tpr, _ = metrics.roc_curve(truth_vals, indicator_vals)
    return fpr, tpr, metrics.auc(fpr, tpr)


def get_ml_preds(test_model, folder, n=N_PREDICTIONS):
    """branch_prob predictions for one ML folder (a GEN-DL regime or Chakraborty)."""
    f = pd.read_csv(f'{test_model}/data/{folder}/df_ml_forced.csv')
    z = pd.read_csv(f'{test_model}/data/{folder}/df_ml_null.csv')
    f['truth value'] = 1
    z['truth value'] = 0

    parts = []
    for tsid in f['tsid'].unique():
        parts.append(f[f['tsid'] == tsid].tail(n))
    for tsid in z['tsid'].unique():
        d = z[z['tsid'] == tsid].tail(n)
        # drop all-zero null rows
        d = d.loc[~(d.drop(columns=['tsid', 'Time'], errors='ignore') == 0).all(axis=1)]
        parts.append(d)

    preds = pd.concat(parts)
    # drop rows where both probabilities are zero
    if 'null_prob' in preds.columns:
        preds = preds[~((preds['branch_prob'] == 0.0) & (preds['null_prob'] == 0.0))]
    return preds.reset_index(drop=True)


def get_ktau_preds(test_model, n=N_PREDICTIONS, rel=PRED_INTERVAL_REL):
    """Kendall-tau EWS predictions (Variance, Lag-1 AC). Regime-independent -> computed once."""
    ews_f = pd.read_csv(f'{test_model}/data/ews/df_ews_forced.csv')
    ews_z = pd.read_csv(f'{test_model}/data/ews/df_ews_null.csv')
    kt_f = pd.read_csv(f'{test_model}/data/ews/df_ktau_forced.csv')
    kt_z = pd.read_csv(f'{test_model}/data/ews/df_ktau_null.csv')
    kt_f['truth value'] = 1
    kt_z['truth value'] = 0

    def extract(ews, kt):
        out = []
        for tsid in kt['tsid'].unique():
            e = ews[(ews['tsid'] == tsid) & (ews['Variable'] == VAR)]
            t0 = e['Time'].iloc[0]
            t1 = e[['Time', 'residuals']].dropna()['Time'].iloc[-1]  # end of residuals
            ps, pe = t0 + (t1 - t0) * rel[0], t0 + (t1 - t0) * rel[1]
            out.append(kt[(kt['tsid'] == tsid) &
                          (kt['Time'] >= ps) & (kt['Time'] <= pe)].tail(n))
        return out

    return pd.concat(extract(ews_f, kt_f) + extract(ews_z, kt_z))


def compute_curves(test_model):
    """Return list of (label, style, fpr, tpr, auc) for one panel."""
    curves = []
    # GEN-DL: one curve per training regime
    for r in GENDL_REGIMES:
        p = get_ml_preds(test_model, f'ml_pred_GEN-DL/{r}')
        fpr, tpr, auc = roc_from(p['truth value'], p['branch_prob'])
        curves.append((GENDL_LABEL[r], GENDL_STYLE[r], fpr, tpr, auc))
    # Baselines computed once
    kt = get_ktau_preds(test_model)
    for name, col in [('Variance', 'ktau_variance'), ('Lag-1 AC', 'ktau_ac')]:
        fpr, tpr, auc = roc_from(kt['truth value'], kt[col])
        curves.append((name, BASELINE_STYLE[name], fpr, tpr, auc))
    ch = get_ml_preds(test_model, CHAK_FOLDER)
    fpr, tpr, auc = roc_from(ch['truth value'], ch['branch_prob'])
    curves.append(('Chakraborty', BASELINE_STYLE['Chakraborty'], fpr, tpr, auc))
    return curves


# ----------------------------------------------------------------------
# Build the 2x2 panel figure with ONE shared legend
# ----------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'pdf.fonttype': 42,   # editable text in Illustrator / PNAS
    'ps.fonttype': 42,
})

fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.8))
axes = axes.ravel()
panel_labels = ['A', 'B', 'C', 'D']
auc_records = []
handles = labels = None

for i, tm in enumerate(TEST_MODELS):
    ax = axes[i]
    for label, style, fpr, tpr, auc in compute_curves(tm):
        lw = 2.0 if label.startswith('GEN-DL') else 1.0
        ax.plot(fpr, tpr, lw=lw, label=label, **style)
        auc_records.append({'test_model': tm, 'method': label, 'AUC': auc})

    ax.plot([0, 1], [0, 1], color='0.6', lw=0.8, ls=(0, (1, 1)))   # chance
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.set_aspect('equal')
    ax.set_title(TITLES.get(tm, tm), fontsize=14, fontweight='bold')
    # ax.text(-0.05, 1.05, panel_labels[i], transform=ax.transAxes,
    #        fontsize=13, fontweight='bold', va='bottom')
    if i in (2, 3):
        ax.set_xlabel('False positive rate', fontsize = 14)
    if i in (0, 2):
        ax.set_ylabel('True positive rate', fontsize = 14)

    if handles is None:                      # legend entries identical across panels
        handles, labels = ax.get_legend_handles_labels()

# Shared legend: row 1 = 5 GEN-DL regimes, row 2 = 3 baselines
leg = fig.legend(handles, labels, loc='center left', ncol=1,
                 frameon=False, bbox_to_anchor=(1.0,0.5),
                 fontsize=14, handlelength=2.4)

fig.subplots_adjust(left=0.07, right=0.99, top=0.95,
                    bottom=0.14, wspace=0.32, hspace=0.28)

png = OUTDIR / 'empirical_ROC_panel.png'
pdf = OUTDIR / 'empirical_ROC_panel.pdf'
fig.savefig(png, dpi=400, bbox_inches='tight', bbox_extra_artists=(leg,))
fig.savefig(pdf, bbox_extra_artists=(leg,))  # vector for PNAS
plt.close(fig)
print(f'Saved: {png}\nSaved: {pdf}')

# ----------------------------------------------------------------------
# AUC summary table (AUC differs per panel -> reported separately, not in legend)
# ----------------------------------------------------------------------
auc_df = pd.DataFrame(auc_records)
row_order = [GENDL_LABEL[r] for r in GENDL_REGIMES] + ['Variance', 'Lag-1 AC', 'Chakraborty']
auc_wide = (auc_df.pivot(index='method', columns='test_model', values='AUC')
                  .reindex(row_order)[TEST_MODELS]
                  .round(4))
auc_wide.to_csv(OUTDIR / 'empirical_AUC_summary.csv')
print('\n=== AUC summary ===')
print(auc_wide.to_string())