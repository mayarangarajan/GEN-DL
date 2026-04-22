#!/usr/bin/env python
# coding: utf-8

# In[7]:


"""
plot ROC curves for empirical data by all the DL models and statistical indicators
Edited from published codes by Bury et al. (2021), 
Deep learning for early warning signals of tipping points, PNAS. 
Modified by Maya Rangarajan to drop zero predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import scipy.stats as stats
import plotly.io as pio
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import plotly.graph_objects as go

test_model='COVID_county'                          # [flu,  COVID, mpox, COVID_county]

# -----------------------------
# Add this function after roc_compute()
# -----------------------------
def compute_metrics_at_sensitivity(truth_vals, indicator_vals, target_sensitivity=0.75):
    """
    Computes threshold at given sensitivity and complementary metrics:
    specificity, precision, F1 score, and AUC.
    Returns a dict and a pandas DataFrame table.
    """
    fpr, tpr, thresholds = metrics.roc_curve(truth_vals, indicator_vals)
    auc = metrics.auc(fpr, tpr)

    # Find threshold closest to target sensitivity
    idx = np.argmin(np.abs(tpr - target_sensitivity))
    threshold = thresholds[idx]

    # Get predicted labels at this threshold
    preds = (indicator_vals >= threshold).astype(int)

    TP = np.sum((preds == 1) & (truth_vals == 1))
    TN = np.sum((preds == 0) & (truth_vals == 0))
    FP = np.sum((preds == 1) & (truth_vals == 0))
    FN = np.sum((preds == 0) & (truth_vals == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else np.nan

    metrics_dict = {
        'Threshold': threshold,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1 score': f1,
        'AUC': auc
    }

    df_table = pd.DataFrame([metrics_dict])
    return metrics_dict, df_table

def ROC_test_model_empirical(test_model, model):

    thresholds_dict = {}

    # Import ML prediction data
    df_ml_forced = pd.read_csv(f'{test_model}/data/{model}/df_ml_forced.csv')
    df_ml_null = pd.read_csv(f'{test_model}/data/{model}/df_ml_null.csv')

    if test_model=='flu' or test_model=='COVID' or test_model=='mpox' or test_model=='COVID_county':
        # Import EWS data
        df_ews_forced = pd.read_csv(f'{test_model}/data/ews/df_ews_forced.csv')
        df_ews_null = pd.read_csv(f'{test_model}/data/ews/df_ews_null.csv')
    else:
        # Import EWS data
        df_ews_forced = pd.read_csv(f'{test_model}/data/ews/df_ews_forced.csv')
        df_ews_null = pd.read_csv(f'{test_model}/data/ews/df_ews_null.csv')

    # Import kendall tau data
    df_ktau_forced = pd.read_csv(f'{test_model}/data/ews/df_ktau_forced.csv')
    df_ktau_null = pd.read_csv(f'{test_model}/data/ews/df_ktau_null.csv')

    # Add column for truth values (1 for forced, 0 for null)
    df_ktau_forced['truth value'] = 1
    df_ktau_null['truth value'] = 0

    df_ml_forced['truth value'] = 1
    df_ml_null['truth value'] = 0

    pred_interval_rel = np.array([0.5, 1.0])

    list_df_ktau_preds = []
    list_df_ml_preds = []


    # Get predictions from forced trajectories
    tsid_vals = df_ml_forced['tsid'].unique()
    for tsid in tsid_vals:

        for var in ['I']:

            # Get EWS data to find start and transition time (for forced data)
            if test_model=='flu' or test_model=='COVID' or test_model=='mpox' or test_model=='COVID_county':
                df = df_ews_forced[(df_ews_forced['tsid']==tsid)&\
                (df_ews_forced['Variable']==var)
                ]
            else:
                df = df_ews_forced[(df_ews_forced['tsid']==tsid)&\
                                (df_ews_forced['Variable']==var)
                                ]
            # Get time n% of way through time series

            t_start = df['Time'].iloc[0]
            t_transition = df[['Time','residuals']].dropna()['Time'].iloc[-1] # where the residuals end

            # Get prediction interval in time
            t_pred_start = t_start + (t_transition-t_start)*pred_interval_rel[0]
            t_pred_end = t_start + (t_transition-t_start)*pred_interval_rel[1]

            # Extract 10 evenly spaced predictions for each transition
            n_predictions = 10

            df_ktau_forced_final = df_ktau_forced[
                (df_ktau_forced['tsid']==tsid)&\
                (df_ktau_forced['Time'] >= t_pred_start)&\
                (df_ktau_forced['Time'] <= t_pred_end)
                ].tail(n_predictions)
            df_ml_forced_final = df_ml_forced[
                (df_ml_forced['tsid']==tsid)&\
                (df_ml_forced['Time'] >= t_pred_start)&\
                (df_ml_forced['Time'] <= t_pred_end)                
                ].tail(n_predictions)            

            # idx = np.round(np.linspace(0, len(df_ktau_forced_final) - 1, n_predictions)).astype(int)
            list_df_ktau_preds.append(df_ktau_forced_final)

            # ML forced trajectories
            # idx = np.round(np.linspace(0, len(df_ml_forced_final) - 1, n_predictions)).astype(int)
            list_df_ml_preds.append(df_ml_forced_final)


            # df_ml_forced_final = df_ml_forced[(df_ml_forced['tsid']==tsid)].tail(n_predictions)
            # list_df_ml_preds.append(df_ml_forced_final)


    # Get predictions from null trajectories
    tsid_vals = df_ml_null['tsid'].unique()
    for tsid in tsid_vals:
        for var in ['I']:

            # Get EWS data to find start and transition time (for forced data)
            if test_model=='flu' or test_model=='COVID' or test_model=='mpox' or test_model=='COVID_county':
                df = df_ews_null[(df_ews_null['tsid']==tsid)&\
                                (df_ews_null['Variable']==var)                               ##var vs variable
                                ]
            else:
                df = df_ews_null[(df_ews_null['tsid']==tsid)&\
                                (df_ews_null['Variable']==var)                               ##var vs variable
                                ]
            # Get time n% of way through time series
            t_start = df['Time'].iloc[0]
            t_transition = df[['Time','residuals']].dropna()['Time'].iloc[-1] # where the residuals end

            # Get prediction interval in time
            t_pred_start = t_start + (t_transition-t_start)*pred_interval_rel[0]
            t_pred_end = t_start + (t_transition-t_start)*pred_interval_rel[1]

            # print(df, t_start, t_transition)

            # Extract 10 evenly spaced predictions for each transitpython /Users/maya/Research/codes/trained_dl_models_and_testing/testing_codes/plots-empirical.pyion
            n_predictions = 10

            df_ktau_null_final = df_ktau_null[
                (df_ktau_null['tsid']==tsid)&\
                (df_ktau_null['Time'] >= t_pred_start)&\
                (df_ktau_null['Time'] <= t_pred_end)
                ].tail(n_predictions)
            df_ml_null_final = df_ml_null[
                (df_ml_null['tsid']==tsid)&\
                (df_ml_null['Time'] >= t_pred_start)&\
                (df_ml_null['Time'] <= t_pred_end)                
                ].tail(n_predictions)

            # Drop rows where all values are zero (excluding 'Time' and 'tsid' columns if needed)
            df_ml_null_final = df_ml_null_final.loc[~(df_ml_null_final.drop(columns=['tsid', 'Time'], errors='ignore') == 0).all(axis=1)]

            # # Ktau null trajectories
            # idx = np.round(np.linspace(0, len(df_ktau_null_final) - 1, n_predictions)).astype(int)
            list_df_ktau_preds.append(df_ktau_null_final)

            # ML null trajectories
            # idx = np.round(np.linspace(0, len(df_ml_null_final) - 1, n_predictions)).astype(int)
            list_df_ml_preds.append(df_ml_null_final)


    df_ktau_preds = pd.concat(list_df_ktau_preds)
    df_ml_preds = pd.concat(list_df_ml_preds)

    # print(df_ml_preds)
    # Drop rows where both branch_prob and null_prob are zero
    ## added this to drop zero predictions
    if 'null_prob' in df_ml_preds.columns:
        df_ml_preds = df_ml_preds[
            ~((df_ml_preds['branch_prob'] == 0.0) & (df_ml_preds['null_prob'] == 0.0))
        ].reset_index(drop=True)



    # Function to compute ROC data from truth and indicator vals
    # and return a df.
    def roc_compute(truth_vals, indicator_vals):

        # Compute ROC curve and threhsolds using sklearn
        fpr, tpr, thresholds = metrics.roc_curve(truth_vals,indicator_vals)

        # Compute AUC (area under curve)
        auc = metrics.auc(fpr, tpr)

        # Threshold for specificity 0.75
        specificity = 1 - fpr
        target_spec = 0.75
        idx = np.argmin(np.abs(specificity - target_spec))
        threshold_for_spec = thresholds[idx]

        # Put into a DF
        dic_roc = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds, 'auc':auc}
        df_roc = pd.DataFrame(dic_roc)

        return df_roc, threshold_for_spec




    #---------------------
    # Compute ROC data
    #–--------------------

    # Initiliase list for ROC dataframes of each EWS
    list_roc = []
    var = 'I'

    # Assign indicator and truth values for ML prediction
    indicator_vals = df_ml_preds['branch_prob']
    truth_vals = df_ml_preds['truth value']
    df_roc, th_ml = roc_compute(truth_vals,indicator_vals)
    df_roc['ews'] = 'ML bif'
    list_roc.append(df_roc)
    thresholds_dict['ML bif'] = th_ml

    # Assign indicator and truth values for variance
    indicator_vals = df_ktau_preds['ktau_variance']
    truth_vals = df_ktau_preds['truth value']
    df_roc, th_var = roc_compute(truth_vals,indicator_vals)
    df_roc['ews'] = 'Variance'
    list_roc.append(df_roc)
    thresholds_dict['Variance'] = th_var

    # Assign indicator and truth values for lag-1 AC
    indicator_vals = df_ktau_preds['ktau_ac']
    truth_vals = df_ktau_preds['truth value']
    df_roc,th_ac = roc_compute(truth_vals,indicator_vals)
    df_roc['ews'] = 'Lag-1 AC'
    list_roc.append(df_roc)
    thresholds_dict['Lag-1 AC'] = th_ac

    # Concatenate roc dataframes
    df_roc_full = pd.concat(list_roc, ignore_index=True)

    # filepath = 'reza_df_roc_seir_I_late.csv'

    # df_roc_full.to_csv(filepath,
    #                    index=False,)


    return df_roc_full, thresholds_dict, df_ml_preds



# In[ ]:



df_ROC_GEN_DL, thresholds_GEN_DL, df_ml_GEN_DL = ROC_test_model_empirical(test_model, 'ml_pred_GEN-DL')
# df_ROC_Bury = ROC_test_model_empirical(test_model, 'ml_pred_Bury') 
df_ROC_Chakraborty, thresholds_Chakraborty, df_ml_Chakraborty = ROC_test_model_empirical(test_model, 'ml_pred_Chakraborty') 

# Compute complementary metrics for ML predictions at 0.75 sensitivity
# Extract raw ML predictions
metrics_GEN_DL, df_table_GEN_DL = compute_metrics_at_sensitivity(
    truth_vals=df_ml_GEN_DL['truth value'],
    indicator_vals=df_ml_GEN_DL['branch_prob'],
    target_sensitivity=0.75
)

metrics_Chakraborty, df_table_Chakraborty = compute_metrics_at_sensitivity(
    truth_vals=df_ml_Chakraborty['truth value'],
    indicator_vals=df_ml_Chakraborty['branch_prob'],
    target_sensitivity=0.75
)

# Combine GEN-DL and Chakraborty metrics for PDF
metrics_df = pd.concat([
    df_table_GEN_DL.assign(Model='GEN-DL'),
    df_table_Chakraborty.assign(Model='Chakraborty')
], ignore_index=True)

# Optional: reorder columns so Model is first
metrics_df = metrics_df[['Model', 'Threshold', 'Sensitivity', 'Specificity', 'Precision', 'F1 score', 'AUC']]

# Print nicely
print("=== GEN-DL Metrics at 0.75 Sensitivity ===")
print(df_table_GEN_DL.round(4).to_string(index=False))

print("\n=== Chakraborty Metrics at 0.75 Sensitivity ===")
print(df_table_Chakraborty.round(4).to_string(index=False))

# In[9]:



fig = go.Figure()
df_roc = df_ROC_GEN_DL

# reza bif plot
df_trace = df_roc[df_roc['ews']=='ML bif']
fig.add_trace(
    go.Scatter(x=df_trace['fpr'],
                y=df_trace['tpr'],
                mode='lines',
                name='GEN-DL (AUC={:.4f})'.format(df_trace['auc'].iloc[0])
                )
    )

# Chakraborty bif plot
df_roc = df_ROC_Chakraborty
df_trace = df_roc[df_roc['ews']=='ML bif']
fig.add_trace(
    go.Scatter(x=df_trace['fpr'],
                y=df_trace['tpr'],
                mode='lines',
                name='Chakraborty (AUC={:.4f})'.format(df_trace['auc'].iloc[0])
    )
)


# Line y=x
fig.add_trace(
    go.Scatter(x=np.linspace(0,1,100),
                y=np.linspace(0,1,100),
                showlegend=False,
                line={'color':'black',
                      'dash':'dash'
                      }
                )
)

# Variance plot
df_trace = df_roc[df_roc['ews']=='Variance']
fig.add_trace(
    go.Scatter(x=df_trace['fpr'],
                y=df_trace['tpr'],
                name='Variance (AUC={:.4f})'.format(df_trace['auc'].iloc[0])
                )
    )

# Lag-1  AC plot
df_trace = df_roc[df_roc['ews']=='Lag-1 AC']
fig.add_trace(
    go.Scatter(x=df_trace['fpr'],
                y=df_trace['tpr'],
                name='Lag-1 AC (AUC={:.4f})'.format(df_trace['auc'].iloc[0])
                )
    )

fig.update_xaxes(
    title=dict(text='False positive rate', font=dict(size=18)),  # Font size for x-axis title
    range=[-0.01, 1],
    tickfont=dict(size=14)  # Font size for x-axis tick labels
)
fig.update_yaxes(
    title=dict(text='True positive rate', font=dict(size=18)),  # Font size for y-axis title
    tickfont=dict(size=14)  # Font size for y-axis tick labels
)

if test_model=='COVID_county':
    title = 'COVID U.S.'
elif test_model=='seir_1500':
    title = 'SEIR'
elif test_model=='flu':
    title = 'FLU'
elif test_model=='COVID':
    title = 'COVID Edmonton'
elif test_model=='mpox':
    title = 'mpox'


fig.update_layout(
    legend=dict(
        x=0.3,
        y=0,
        font=dict(size=16)
    ),
    width=600,
    height=600,  # ⬅️ increase overall canvas height
    title=dict(
        text=f'{title}',
        font=dict(size=20),
        x=0.5,  # Center the title
        xanchor='center',
        y=0.9
    )
)
'''
# Add thresholds annotation for GEN-DL
threshold_text = "Thresholds (specificity = 0.75)<br><br>"

threshold_text += "<b>GEN-DL</b><br>"
for key, val in thresholds_GEN_DL.items():
    threshold_text += f"{key}: {val:.3f}<br>"

threshold_text += "<br><b>Chakraborty</b><br>"
for key, val in thresholds_Amit.items():
    threshold_text += f"{key}: {val:.3f}<br>"
'''
fig.write_image(f'../../output_charts/ROC/{test_model}-ROC.png')
output_path = f'../../output_charts/ROC/{test_model}-ROC.png'
fig.write_image(output_path, scale=2)

print(f"ROC plot saved at: {output_path}")

with PdfPages('../../output_charts/ROC/{}_ROC_and_metrics.pdf'.format(test_model)) as pdf:
    # Page 1: Insert the PNG image
    fig, ax = plt.subplots(figsize=(9,9))
    img = mpimg.imread(f'../../output_charts/ROC/{test_model}-ROC.png')
    ax.imshow(img)
    ax.axis('off')  # Hide axes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Page 2: Table
    metrics_df = pd.concat([df_table_GEN_DL, df_table_Chakraborty], keys=['GEN-DL','Chakraborty'])
    metrics_df.reset_index(level=0, inplace=True)
    metrics_df.rename(columns={'level_0':'Model'}, inplace=True)

    # Round all numeric columns to 4 decimals
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    metrics_df[numeric_cols] = metrics_df[numeric_cols].round(4)    

    metrics_display = metrics_df.copy()

    # Format all float columns to 4 decimal points
    for col in metrics_display.select_dtypes(include=['float']):
        metrics_display[col] = metrics_display[col].map('{:.4f}'.format)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    tbl = ax.table(cellText=metrics_df.values,
                   colLabels=metrics_display.columns,
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(metrics_df.columns)))) 
    tbl.scale(1, 2)
    pdf.savefig(fig)
    plt.close()

