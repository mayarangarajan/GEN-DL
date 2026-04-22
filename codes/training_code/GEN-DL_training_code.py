#Python script to train a neural network using Keras library.
#Edited from published codes by Bury et al. (2021), Deep learning for early warning signals of tipping points, PNAS.
#
# Chakraborty et al. (2024), An early warning indicator trained on stochastic disease-spreading models with different noises
# https://zenodo.org/records/12537663
#
# #Modified by Maya Rangarajan

import os
import sys

os.environ["TF_USE_LEGACY_KERAS"] = "1"

SCRIPT_VERSION = "v4.0 optimized version"  # Bury simple architecture
print(f"Running script version: {SCRIPT_VERSION}")


import tensorflow as tf
import pandas as pd
import numpy as np
import random
import sys
import zipfile
import time
import gc
import pickle
from tqdm import trange, tqdm
from tqdm.keras import TqdmCallback
from sklearn.metrics import roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, AveragePooling1D
#try:
#    from tensorflow.keras.optimizers.legacy import Adam
#except ImportError:
#training_code/DL_Bury_training.py    from tensorflow.keras.optimizers import Adam

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import keras_tuner as kt
from datetime import datetime

# ===== LOAD TUNED HYPERPARAMETERS =====
# best_hp_path = f'best_hps_bury.pickle'
'''
best_hp_path = f'best_hps_bury.pickle'
if os.path.exists(best_hp_path):
    print(f"\nLoading tuned hyperparameters from {best_hp_path}")

    with open(best_hp_path, 'rb') as f:
        best_hps = pickle.load(f)

    print("\nBest hyperparameters loaded:")
    for key, val in best_hps.values.items():
        print(f"  {key}: {val}")

else:
    print("\nNo hyperparameter file found — using default parameters.")
    best_hps = None
'''
best_hps = None

random.seed(0)

run = "run12"

# ===== DATA CONFIGURATION =====
(lib_size, ts_len) = (100000, 100)
zip_path = './data_train/resids.zip'
all_data = None

# ===== MODEL CONFIGURATION (Bury Architecture) =====

''' ORIGINAL
CNN_LAYERS        = 2 #switch between 1 and 2
LSTM_LAYERS       = 1
FILTERS           = 50
KERNEL_SIZE       = 12
POOL_SIZE         = 2
MEM_CELLS         = 50
MEM_CELLS2        = 10
DROPOUT           = 0.10
LEARNING_RATE     = 0.0005
BATCH_SIZE        = 1000
MAX_EPOCHS        = 200
EARLY_STOPPING_PATIENCE = 20
INITIALIZER       = 'lecun_normal'
ALSO CHANGE AVEPOOL TO MAXPOOL
'''
### TRIAL ####
CNN_LAYERS        = 2 #switch between 1 and 2
LSTM_LAYERS       = 2
FILTERS           = 64
KERNEL_SIZE       = 7
POOL_SIZE         = 2
MEM_CELLS         = 64
MEM_CELLS2        = 32
DROPOUT           = 0.20
LEARNING_RATE     = 0.0001
BATCH_SIZE        = 256
MAX_EPOCHS        = 200
EARLY_STOPPING_PATIENCE = 20
INITIALIZER       = 'lecun_normal'

# ===== CALLBACKS =====

class ClearMemoryCallback(Callback):
    def on_train_end(self, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()

class AUCCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        self.x_val, self.y_val = val_data

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_val, verbose=0)
        y_true_binary = 1 - self.y_val[:, 0]
        y_pred_binary = 1 - y_pred[:, 0]
        auc = roc_auc_score(y_true_binary, y_pred_binary)
        logs['val_auc'] = auc
        print(f' - val_auc: {auc:.4f}')


# ===== MODEL BUILDER =====

def build_model(hp=None):
    """Bury et al. sequential CNN-LSTM architecture."""

    if hp is None:
        # Hardcoded parameters
        filters        = FILTERS
        kernel_size    = KERNEL_SIZE
        pool_size      = POOL_SIZE
        mem_cells      = MEM_CELLS
        mem_cells2     = MEM_CELLS2
        dropout        = DROPOUT
        learning_rate  = LEARNING_RATE
        cnn_layers     = CNN_LAYERS
        lstm_layers    = LSTM_LAYERS
    else:
        filters       = hp.get('filters')
        kernel_size   = hp.get('kernel_size')
        pool_size     = hp.get('pool_size')
        mem_cells     = hp.get('mem_cells')
        mem_cells2    = hp.get('mem_cells2')
        dropout       = hp.get('dropout')
        learning_rate = hp.get('learning_rate')
        cnn_layers    = hp.get('cnn_layers')
        lstm_layers   = hp.get('lstm_layers')

    model = Sequential()

    # --- CNN block ---
    model.add(Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        padding='same',
        input_shape=(ts_len, 1),
        kernel_initializer=INITIALIZER
    ))
    if cnn_layers == 2:
        model.add(Conv1D(
            filters=filters * 2,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            kernel_initializer=INITIALIZER
        ))

    model.add(Dropout(dropout))
    model.add(AveragePooling1D(pool_size=pool_size))

    # --- LSTM block ---
    if lstm_layers == 1:
        model.add(LSTM(mem_cells, return_sequences=True, kernel_initializer=INITIALIZER))
        model.add(Dropout(dropout))
    elif lstm_layers == 2:
        model.add(LSTM(mem_cells, return_sequences=True, kernel_initializer=INITIALIZER))
        model.add(LSTM(mem_cells, return_sequences=True, kernel_initializer=INITIALIZER))
        model.add(Dropout(dropout))

    model.add(LSTM(mem_cells2, kernel_initializer=INITIALIZER))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax', kernel_initializer=INITIALIZER))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate if hp else LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ===== DATA LOADING HELPER =====

def _process_one_tsid(tsid: int):
    df = all_data[tsid]
    if tsid == 1:
        print("Columns:", df.columns.tolist())
        print("First 5 rows:\n", df.head())

    values = df[['residuals']].fillna(0).values
    t_transition = df[['Time', 'residuals']].dropna()['Time'].iloc[-1]

    if (t_transition < 1500 - 1) and (t_transition > ts_len):
        values_100 = values[int(t_transition - ts_len + 1): int(t_transition + 1)]
    elif (t_transition < 1500 - 1) and (t_transition <= ts_len):
        values_100 = values[0:100]
    else:
        values_100 = values[-100:]

    return tsid, values_100.astype(np.float32), int(t_transition)


# ===== MAIN =====

if __name__ == '__main__':

    # --- Load all CSVs into memory ---
    print("Loading all sequences into memory...")
    all_data = {}
    with zipfile.ZipFile(zip_path) as zf:
        for tsid in tqdm(range(1, lib_size + 1), desc="Reading zip"):
            with zf.open(f'resids{tsid}.csv') as f:
                all_data[tsid] = pd.read_csv(f)
    print("Finished loading all data!")

    # --- Check file integrity ---
    df_targets = pd.read_csv('./data_train/labels.csv', index_col='sequence_ID')
    df_groups  = pd.read_csv('./data_train/groups.csv', index_col='sequence_ID')

    for col, name in [({'class_label'}, 'labels.csv'), ({'dataset_ID'}, 'groups.csv')]:
        missing = col - set((df_targets if name == 'labels.csv' else df_groups).columns)
        if missing:
            sys.exit(f"Error: {name} is missing columns: {missing}")

    missing_in_groups = set(df_targets.index) - set(df_groups.index)
    if missing_in_groups:
        sys.exit(f"Error: sequence_IDs in labels.csv but not groups.csv: {missing_in_groups}")
    print("CSV check passed!")

    # --- Command line args ---
    model_type = int(sys.argv[1])
    kk         = int(sys.argv[2])
    RUN_MODE   = "train"
    SEED       = int(sys.argv[3]) if len(sys.argv) > 4 else 42

    print(f'Train a classifier of type {model_type} with index {kk} in {RUN_MODE} mode')

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    f2_name    = f'training_results_{kk}_{model_type}.csv'
    f_results2 = open(f2_name, "w")

    # --- Extract time series ---
    print('Extract time series from zip file')
    tsid_vals = np.arange(1, lib_size + 1)
    results   = [_process_one_tsid(tsid) for tsid in tqdm(tsid_vals, desc="Processing")]

    del all_data
    gc.collect()

    results.sort(key=lambda x: x[0])
    sequences        = np.array([r[1] for r in results])
    transition_point = [r[2] for r in results]

    df_targets = pd.read_csv('./data_train/labels.csv', index_col='sequence_ID')
    df_groups  = pd.read_csv('./data_train/groups.csv', index_col='sequence_ID')

    # --- Normalize by mean absolute value ---
    for i, tsid in enumerate(tsid_vals):
        nonzero = sequences[i][sequences[i] != 0]
        if len(nonzero) > 0:
            avg = np.mean(np.abs(nonzero))
            sequences[i][sequences[i] != 0] /= avg

    final_seq = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))

    # --- Train / val / test split ---
    train      = np.array([final_seq[i] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid] == 1])
    validation = np.array([final_seq[i] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid] == 2])
    test       = np.array([final_seq[i] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid] == 3])

    train_target      = np.array([df_targets['class_label'].loc[tsid] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid] == 1])
    validation_target = np.array([df_targets['class_label'].loc[tsid] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid] == 2])
    test_target       = np.array([df_targets['class_label'].loc[tsid] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid] == 3])

    # AUC callback requires one-hot validation targets
    from tensorflow.keras.utils import to_categorical
    y_val_onehot = to_categorical(validation_target, num_classes=2)
    auc_callback = AUCCallback(val_data=(validation, y_val_onehot))
    
    # --- Build and train ---
    model      = build_model(hp=None)

    if best_hps is not None:
        print("\n===== ACTIVE MODEL PARAMETERS =====")
        for key, val in best_hps.values.items():
            print(f"{key}: {val}")
        print("="*40)

    model_name = f'best_model_{kk}_{model_type}_length{ts_len}.keras'

    early_stop = EarlyStopping(
        monitor='val_auc',
        min_delta=0.0001,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=False,
        verbose=1,
        mode='max'
    )
    chk = ModelCheckpoint(
        filepath=model_name,
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False
    )

    history = model.fit(
        train, train_target,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[auc_callback, early_stop, chk],
        validation_data=(validation, validation_target),
        verbose=1
    )

    # --- Load best checkpoint ---
    model = load_model(model_name, compile=False)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"\nFinal model saved to {model_name}")

    # ===== TEST METRICS =====
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                  recall_score, confusion_matrix,
                                  classification_report, roc_auc_score)

    test_preds       = np.argmax(model.predict(test), axis=1)
    test_probs       = model.predict(test)

    acc       = accuracy_score(test_target, test_preds)
    f1        = f1_score(test_target, test_preds, average='macro')
    precision = precision_score(test_target, test_preds, average='macro')
    recall    = recall_score(test_target, test_preds, average='macro')
    auc       = roc_auc_score((test_target != 0).astype(int), 1 - test_probs[:, 0])
    cm        = confusion_matrix(test_target, test_preds)

    if best_hps is not None:
        print("\n===== FINAL HYPERPARAMETERS USED =====")
        for key, val in best_hps.values.items():
            print(f"{key}: {val}")
            
    print("===== TEST METRICS =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(test_target, test_preds, digits=3))

    param_string = ";".join([f"{k}={v}" for k,v in best_hps.values.items()]) if best_hps else "default"

    f_results2.write(
        f"{kk},"
        f"{f1:.4f},"
        f"{precision:.4f},"
        f"{recall:.4f},"
        f"{ts_len},{lib_size},"
        f"{param_string}\n"
    )
    f_results2.flush()
    f_results2.close()