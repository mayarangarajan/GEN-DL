"""
Compute rolling ktau values from EWS files
Modified from Chakraborty (2024), 
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
import warnings

#============================================================================
# CHANGE THESE VARIABLES
#============================================================================
ts_type = 'COVID_county'  # Options: 'SEIR', 'SIR', 'COVID', 'COVID_county', 'mpox', 'flu'
forced_or_null = 'null'  # Options: 'forced', 'null'

#============================================================================
# FIXED SETTINGS
#============================================================================
data_dir = './data/ews'
output_dir = './data/ews'
ml_spacing = 10  # spacing between ML predictions
t_res = ml_spacing

warnings.filterwarnings(
    "ignore",
    message="One or more sample arguments is too small"
)


#============================================================================
# FUNCTIONS
#============================================================================

# Function to compute kendall tau for time series data up to point t_fin
def ktau_compute(series, t_fin):
    # selected data in series where from point where measured variable
    # is defined, up to t_fin
    t_start = series[pd.notnull(series)].index[1]
    series_reduced = series.loc[t_start:t_fin]
    x1 = series_reduced.index.values
    x2 = series_reduced.values
    ktau, pval = stats.kendalltau(x1, x2)
    return ktau

# Compute kendall tau values at equally spaced time points
def ktau_series(series):
    tVals = series.index[::t_res]
    ktauVals = []
    for t in tVals:
        ktau = ktau_compute(series, t)
        ktauVals.append(ktau)
    # Return series
    ktauSeries = pd.Series(ktauVals, index=tVals)
    return ktauSeries

#============================================================================
# MAIN PROCESSING
#============================================================================

print(f"Processing: {ts_type} - {forced_or_null}")
print(f"Data directory: {data_dir}")
print("-" * 60)

data_dir = Path(data_dir)

# Read the EWS file
ews_filepath = data_dir / f'df_ews_{forced_or_null}_{ts_type}.csv'

try:
    df_ews_all = pd.read_csv(ews_filepath)
    print(f"Loaded {ews_filepath}")
    print(f"Shape: {df_ews_all.shape}")
    print(f"Columns: {df_ews_all.columns.tolist()}")
    
    # Set index
    df_ews_all = df_ews_all.set_index(['tsid', 'Time'])
    
    # Store list of dfs with kendall tau values from each simulation
    list_df = []
    
    # Loop through each tsid
    tsid_vals = df_ews_all.index.unique(level='tsid')
    
    for tsid in tsid_vals:
        # Get pre-computed variance and ac1 for this simulation
        series_var = df_ews_all.loc[tsid]['variance']
        series_ac = df_ews_all.loc[tsid]['ac1']
        
        # Check number of valid points
        n_var = series_var[pd.notnull(series_var)].shape[0]
        n_ac = series_ac[pd.notnull(series_ac)].shape[0]
        
        if n_var < 2 or n_ac < 2:
            print(f"Skipping tsid {tsid}: not enough valid points (variance={n_var}, ac1={n_ac})")
            continue  # Skip this tsid

        # Compute kendall tau series
        series_ktau_var = ktau_series(series_var)
        series_ktau_var.name = 'ktau_variance'
        
        series_ktau_ac = ktau_series(series_ac)
        series_ktau_ac.name = 'ktau_ac'
        
        # Put into temporary dataframe
        df_temp = pd.concat([series_ktau_var, series_ktau_ac], axis=1).reset_index()
        df_temp['tsid'] = tsid
        list_df.append(df_temp)
        
        print(f'Ktau computed for tsid {tsid}')
    
    if len(list_df) == 0:
        print("No data processed successfully")
    else:
        # Concatenate kendall tau dataframes
        df_ktau = pd.concat(list_df).set_index(['tsid', 'Time'])
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export dataframe
        output_file = output_dir / f'df_ktau_{forced_or_null}_{ts_type}.csv'
        df_ktau.to_csv(output_file)
        
        print(f'\nResults saved to {output_file}')
        print(f'\nSummary statistics:')
        print(df_ktau.describe())

except FileNotFoundError:
    print(f"File not found: {ews_filepath}")
except Exception as e:
    print(f"Error processing file: {e}")
    raise