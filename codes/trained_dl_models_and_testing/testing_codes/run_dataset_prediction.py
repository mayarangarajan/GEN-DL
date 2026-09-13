'''
Code to generate predictions from the Chakraborty et al.'s DL classifiers
on a give time series of residuals
Edited from published codes by Bury et al. (2021), 
Deep learning for early warning signals of tipping points, PNAS. 
Codes: Deep Learning for Disease Outbreak Prediction: A parallel LSTM-CNN model (2025)
https://zenodo.org/records/15377482
Edited by Maya Rangarajan
1. Added GEN-DL
2. Updated Chakraborty load
3. Loading model only once per run for speedup

INPUT
1. Null/TC resids files for dataset tested

RUN
1. RUN FROM TRAINED_DL_MODELS_AND_TESTING folder

OUTPUT
1. pred_resids_{dataset}.csv in .{dataset}/data/ml_pred_{test_model}

'''

#python libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import random
import sys
import itertools
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
from glob import glob
import tf_keras as k3

tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

random.seed(42)

##################################################################
### ====== SET BEFORE RUNNING CODE ===============================
##################################################################

dataset = 'COVID'                #[COVID, COVID_county, COVID_state, mpox, flu, SEIPR_MultNoise, SEIPR_AddNoise, SIR_MultNoise, SIR_AddNoise]     
test_model = 'GEN-DL'            # [Chakraborty, GEN-DL]  
ts_len=100  

resid_files = glob("./{}/data/resids/resids*.csv".format(dataset))
print("FOUND FILES:", len(resid_files))

# ---- Load all models once before the file loop ----
print("Loading models...")
models = {}
for model_type in [1, 2]:
    for kk in np.arange(1, 2):
        if test_model == 'GEN-DL':
            try:
                import tf_keras as tk
            except ImportError:
                import tensorflow.keras as tk
            model_name = f"./trained_GEN-DL_models/best_model_{kk}_{model_type}_length{ts_len}.keras"
            models[(kk, model_type)] = tk.models.load_model(model_name)

        elif test_model == 'Chakraborty':
            model_name = f"./trained_Chak_models/best_model_{kk}_{model_type}_CK100.pkl"
            models[(kk, model_type)] = k3.models.load_model(model_name)

print(f"Loaded {len(models)} models.")

from glob import glob
for index, filepath in enumerate(resid_files):

    filepath = filepath.replace('\\', '/')

    # Extract the filename after the last '/'
    filename = filepath.split('/')[-1]

    print("{} - doing prediction for {}".format(index, filepath))

    # Construct the output file path
    # filepath_out = f"{dataset}/data/ml_pred_GEN-DL/pred_{filename}"
    os.makedirs(f"{dataset}/data/ml_pred_{test_model}", exist_ok=True)
    filepath_out = f"{dataset}/data/ml_pred_{test_model}/pred_{filename}"                                                                        

    # Steps of datapoints in between each DL prediction
    mult_factor = 10

    # Total number of DL predictions to make
    pad_samples = int(ts_len/mult_factor)                                                          

    df = pd.read_csv(filepath).dropna()
    # Length of inupt time series 
    df_len = len(df)   

    if df_len > ts_len:
        #df of last 100/500 points
        df_last = df[-ts_len:]               
        if 'residuals' in df.columns:
            resids = df_last['residuals'].values.reshape(1,-1,1)
        elif 'Residuals' in df.columns:
            resids = df_last['Residuals'].values.reshape(1,-1,1)
        else:
            print("Column not found.")      
        seq_len = len(df_last)
    else:
        resids = df['residuals'].values.reshape(1,-1,1)          
        # Length of inupt time series 
        seq_len = len(df)               

    def get_dl_predictions(resids, model_type, kk, model):

        '''
        Generate DL prediction time series on resids
        from DL classifier with type 'model_type' and index kk.
        '''

        # Setup file to store DL predictions
        predictions_file_name = './predictions/y_pred_{}_{}.csv'.format(kk,model_type)
        f1 = open(predictions_file_name,'w')

        # Loop through each possible length of padding
        # Start with revelaing the DL algorith only the earliest points
        for pad_count in range(pad_samples-1, -1, -1):

            temp_ts = np.zeros((1,ts_len,1))

            ts_gap = ts_len-seq_len
            pad_length = mult_factor*pad_count

            if pad_length + ts_gap > ts_len:
                zero_range = ts_len
            else:
                zero_range = pad_length + ts_gap

            if zero_range == ts_len:
                # Set all ML predictions to zero
                y_pred = np.zeros(2).reshape(1,2)                                               
            else:
                for j in range(0, zero_range):
                    temp_ts[0,j] = 0
                for j in range(zero_range, ts_len):
                    temp_ts[0,j] = resids[0,j-zero_range]

                # normalizing inputs: take averages, since the models were also trained on averaged data. 
                values_avg = 0.0
                count_avg = 0
                for j in range (0,ts_len):
                    if temp_ts[0,j] != 0:
                        values_avg = values_avg + abs(temp_ts[0,j])
                        count_avg = count_avg + 1
                if count_avg != 0:
                    values_avg = values_avg/count_avg
                for j in range (0,ts_len):
                    if temp_ts[0,j] != 0:
                        temp_ts[0,j] = temp_ts[0,j]/values_avg

                # Compute DL prediction
                y_pred = model.predict(temp_ts)



            # Write predictions to file
            np.savetxt(f1, y_pred, delimiter=',')
            # print('Predictions computed for padding={}'.format(pad_count*mult_factor))
        gc.collect()
        f1.close()

        return 

    # Compute DL predictions from all 2 trained models
    for model_type in [1,2]:                                
        for kk in np.arange(1,2):
            # print('Compute DL predictions for model_type={}, kk={}'.format(
            #     model_type,kk))

            get_dl_predictions(resids, model_type, kk, model=models[(kk, model_type)])

    # Compute average prediction among all 2 DL classifiers
    list_df_preds = []
    for model_type in [1,2]:
        for kk in np.arange(1,2):
            filename = './predictions/y_pred_{}_{}.csv'.format(kk,model_type)
            df_preds = pd.read_csv(filename,header=None)
            df_preds['time_index'] = df_preds.index
            df_preds['model_type'] = model_type
            df_preds['kk'] = kk
            list_df_preds.append(df_preds)


    # Concatenate
    df_preds_all = pd.concat(list_df_preds).reset_index(drop=True)

    # Compute mean over all predictions
    df_preds_mean = df_preds_all.groupby('time_index').mean()
    df_preds_mean = df_preds_mean[[0,1]]
    df_preds_mean = df_preds_mean.assign(b=df_preds_mean.iloc[:,[1]].sum(axis=1))

    # Export predictions
    df_preds_mean.iloc[:,[0,1,2]].to_csv(filepath_out,index=False,header=False)


