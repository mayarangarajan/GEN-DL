#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Organise ML data output of empirical datasets by Chakraborty et al's model into a single dataframe. 
Edited from published codes by Bury et al. (2021), 
Deep learning for early warning signals of tipping points, PNAS. 
Codes: Deep Learning for Disease Outbreak Prediction: A parallel LSTM-CNN model (2025)
https://zenodo.org/records/15377482

FUNCTION
Aggregates all predictions into dl_ml_forced.csv and dl_ml_null.csv
INPUT
1. predictions for each outbreak in dataset
2. df_ews_forced and df_ews_null
"""

import numpy as np
import pandas as pd
import os



# length of classifier
classifier_length=100
pad_samples = 10
# Spacing between ML data points
ml_spacing = int(classifier_length/pad_samples)

#Change your test model
#datasets are: 'flu', 'COVID', 'mpox', 'COVID_county'
dataset = 'COVID_county'
model = 'GEN-DL'      # 'GEN-DL', 'Chakraborty'

# Import EWS data for variable I (required for time values of original data)
df_ews = pd.read_csv('{}/data/ews/df_ews_forced.csv'.format(dataset))
df_ews = df_ews[df_ews['Variable']=='I']

# Get all file names of ML predictions 
all_files = os.listdir('{}/data/ml_pred_{}/'.format(dataset,model))
# all_files = [f for f in all_files if f.split('_')[0]=='ensemble']

# Get null and forced filenames
all_files_null = [s for s in all_files if s.find('null')!=-1]
all_files_forced = [s for s in all_files if s.find('forced')!=-1]



#----------------
# Organize data for forced trajectories
#-----------------

list_df_ml = []
for filename in all_files_forced:
    df = pd.read_csv('{}/data/ml_pred_{}/{}'.format(dataset, model, filename), 
                     header = None,
                     names = ['null_prob','branch_prob','bif_prob'])         #label 0 for null; 1 for branch

    # Remove ".csv" from filename
    filename = filename.replace(".csv", "")
    print(filename)

    # Extract the numeric part from filename
    filename_split = filename.split('_')
    numeric_part = "".join(filter(lambda x: x.isdigit(), filename_split[-1]))

    # Skip files that do not contain a number
    if not numeric_part:
        # print(f"Skipping file: {filename} {numeric_part} (no numeric tsid found)")
        continue

    # Convert extracted number to integer
    tsid = int(numeric_part)

    df['tsid'] = tsid

    # Get time values up to the transition point
    tVals = df_ews[(df_ews['tsid']==tsid)][['Time','residuals']].dropna()['Time'].values

    # Take last 'classifier_length' time points of data
    tValsLast = tVals[-classifier_length:]
    # If shorter than classifier_length points, pad with Nan (this is done prior to using ML)
    if len(tValsLast)<classifier_length:
        tValsLast = np.pad(tValsLast, (classifier_length-len(tValsLast),0), constant_values=0)
    # ML time points spacing
    ml_time_vals = tValsLast[::ml_spacing]

    # Assign to df
    df['Time']=ml_time_vals      

    # Append dataframe to list
    list_df_ml.append(df)


# Concatenate dfs
df_ml = pd.concat(list_df_ml)
# sort by type, then latitude
df_ml.sort_values(['tsid','Time'],inplace=True)

# Export ML dataframe
df_ml.to_csv('{}/data/ml_pred_{}/df_ml_forced.csv'.format(dataset, model), index=False)




#----------------
# Organize data for null trajectories
#-----------------

list_df_ml = []
for filename in all_files_null:
    df = pd.read_csv('{}/data/ml_pred_{}/{}'.format(dataset, model, filename), 
                     header = None,
                     names = ['null_prob','branch_prob','bif_prob'])

    # Remove ".csv" from filename
    filename = filename.replace(".csv", "")
    print(filename)

    # Extract the numeric part from filename
    filename_split = filename.split('_')
    numeric_part = "".join(filter(lambda x: x.isdigit(), filename_split[-1]))

    # Skip files that do not contain a number
    if not numeric_part:
        # print(f"Skipping file: {filename} (no numeric tsid found)")
        continue

    # Convert extracted number to integer
    tsid = int(numeric_part)

    df['tsid'] = tsid

    # Get time values for this transition
    tVals = df_ews[(df_ews['tsid']==tsid)]['Time'].values

    # Take last classifier_length time points of data
    tValsLast = tVals[-classifier_length:]
    # If shorter than classifier_length points, pad with Nan (this is done prior to using ML)
    if len(tValsLast)<classifier_length:
        tValsLast = np.pad(tValsLast, (classifier_length-len(tValsLast),0), constant_values=0)
    # ML time points spacing
    ml_time_vals = tValsLast[::ml_spacing]

    # Assign to df
    df['Time']=ml_time_vals      

    # Append dataframe to list
    list_df_ml.append(df)


# Concatenate dfs
df_ml = pd.concat(list_df_ml)
# sort by type, then latitude
df_ml.sort_values(['tsid','Time'],inplace=True)

# Export ML dataframe
df_ml.to_csv('{}/data/ml_pred_{}/df_ml_null.csv'.format(dataset, model), index=False)

print('Done')

