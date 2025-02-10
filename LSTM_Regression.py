import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Function to transform data for LSTM
def create_sequences_for_multiple_series(df, id_col, feature_cols, target_col, time_steps):
    """
    Create sequences from multiple time series data for LSTM input.

    Parameters:
    - df: pd.DataFrame containing time series identifiers, features, and target.
    - id_col: str, name of the identifier column for different time series.
    - feature_cols: list of column names for features.
    - target_col: str, name of the target column.
    - time_steps: int, number of timesteps for each input sequence.

    Returns:
    - X: np.array, Input sequences with shape (samples, time_steps, features).
    - y: np.array, Target values corresponding to each sequence.
    """

    X_total, y_total = [], []

    # Iterate over each unique time series
    for series_id in df[id_col].unique():
        df_series = df[df[id_col] == series_id]

        # Generate sequences for this series
        X_series, y_series = create_sequences_from_timestamps(df_series, feature_cols, target_col, time_steps)
        
        # Append to total lists
        X_total.append(X_series)
        y_total.append(y_series)

    # Concatenate all arrays into single arrays
    X_combined = np.concatenate(X_total, axis=0)
    y_combined = np.concatenate(y_total, axis=0)

    return X_combined, y_combined

def create_sequences_from_timestamps(df, feature_cols, target_col, time_steps):
    """
    Create sequences from timestamped data for LSTM input (as described previously).

    Parameters:
    - df: pd.DataFrame containing time series identifiers, features, and target.
    - feature_cols: list of column names for features.
    - target_col: str, name of the target column.
    - time_steps: int, number of timesteps for each input sequence.

    Returns:
    - X: np.array, Input sequences with shape (samples, time_steps, features).
    - y: np.array, Target values corresponding to each sequence.
    """
    
    X, y = [], []

    for i in range(len(df) - time_steps):
        X.append(df.iloc[i:i + time_steps][feature_cols].values)
        y.append(df.iloc[i + time_steps][target_col])

    return np.array(X), np.array(y)

def LSTM_model(X,y,test_size):
    """
    Create sequences from timestamped data for LSTM input (as described previously).

    Parameters:
    - X: np.array, Input sequences with shape (samples, time_steps, features).
    - y: np.array, Target values corresponding to each sequence.
    - test_size: float number, from 0 to 1, Percentage of test data in the total training data

    Returns:
    - model: the trained model
    - history: the model history contains training data
    - mse: the mean square error of val loss of the last epoch
    - X_train: the feature matrix of training dataset
    - X_val: the feature matrix of test dataset
    - 
    """

    # Split data into training and validation sets
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
    
    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=20, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))  # Optional, helps prevent overfitting
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer=Adam(), loss='mse')

    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)

    return model, history, mse, X_train, X_val

def create_sequences_for_multiple_series_inference(df, id_col, feature_cols, time_steps):
    """
    Create sequences from timestamped data for features of missing OD600.

    Parameters:
    - df: pd.DataFrame containing time series identifiers, features, and target.
    - feature_cols: list of column names for features.
    - target_col: str, name of the target column.
    - time_steps: int, number of timesteps for each input sequence.

    Returns:
    - X_combined: np.array, Input sequences with shape for prediction (samples, time_steps, features).
    - seq_indices_total: list, time index corresponding to each sequence.
    """
    X_total = []
    seq_indices_total = []

    for series_id in df[id_col].unique():
        df_series = df[df[id_col] == series_id].copy()
        
        for i in range(len(df_series) - time_steps + 1):
            seq = df_series.iloc[i : i + time_steps][feature_cols].values
            X_total.append(seq)

            # Store some reference info (could be the last timestamp in that window)
            idx_info = {
                'series_id': series_id,
                'start_index': df_series.index[i],
                'end_index': df_series.index[i + time_steps - 1]
            }
            seq_indices_total.append(idx_info)

    X_combined = np.array(X_total)
    return X_combined, seq_indices_total