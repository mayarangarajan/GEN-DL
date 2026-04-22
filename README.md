CONTENTS
All codes and training data for GEN-DL

Codes
1. training_code: code used to train GEN-DL_model
2. data_generation_testing: SEIPR data for testing
3. data_generation_training: GEN-DL training daa
4. analysis_codes: latent space analysis and critical slowing down analysis (on disease data)
5. AUC_charting_codes: plot AUC results
6. Codes
        trained_dl_models_and_testing
            GEN-DL model
            Chakraborty model
            Codes, data for testing
    Output charts (all outputs saved here)


PROCESS FOR TESTING
1. Step 1: Navigate to codes/trained_dl_models_and_testing folder
2. Step 2: run run_data_set_prediction.py for each disease dataset (mpox, flu, COVID, COVID_county)
3. Step 3: run organize_ml_predictions.py for each disease dataset
4. Step 4: run plot_empirical.py for each disease dataset

5. You can skip Step 2 and run just steps 3 and 4 as the folder already contains outputs from Step 2.

All Bury and Chakraborty results from Chakraborty et al. (2025) Codes: Deep Learning for Disease Outbreak Prediction: A parallel LSTM-CNN model
https://zenodo.org/records/15377482

SIDATR results for COVID_county calculated by Maya Rangarajan using testing pipeline
