import numpy as np
import tensorflow as tf
from tensorflow import keras
import shap

def find_exp_id_for_x_missing(df_online, df_od):
    online_exp = df_online['exp_id'].unique()
    od_exp = df_od['exp_id'].unique()
    # predict_exp_id = list(set(online_exp)-list(od_exp))
    predict_exp_id = np.setdiff1d(online_exp, od_exp)
    return predict_exp_id

   

def predict_and_postprocess(model, X_missing):
    """
    Predict OD600 values using the trained LSTM model and ensure the results are physical.
    
    Parameters:
    - model: Trained LSTM model.
    - X_missing: np.array of shape (num_samples, time_steps, num_features), input data for prediction.

    Returns:
    - predictions: np.array of predictions, with negative values rectified to zero.
    """
    # Make predictions on the missing data
    predictions = model.predict(X_missing)

    # Post-process predictions: Ensure OD600 values are physically sensible (e.g., non-negative)
    predictions = np.maximum(predictions, 0)  # Clamp negative predictions to zero

    return predictions

def SHAP_feature_importance(X_train, X_test, model):
    """
    Use SHAP to find feature importance.
    
    Parameters:
    - X_train: np.array, input data for model training
    - X_test: np.array, input data for model training
    - model: Trained LSTM model.

    Returns:
    - shap_values: list of arrays, each array has the shape (10, timestpes, features).
    """
    # Pick 50 samples from X_train as background
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
    # Wrap the model in a function that returns the logits or the outputs
    explainer = shap.GradientExplainer(model, background)
    X_test_sample = X_test[:10]  
    shap_values = explainer.shap_values(X_test_sample)
    return shap_values
