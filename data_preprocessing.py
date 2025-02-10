import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_online_params(file_path):
    """
    Load the time-series online parameters CSV file.
    
    Args:
        file_path (str): Path to the Time_series_online_params.csv file.
    
    Returns:
        pd.DataFrame: Loaded online parameters data.
    """
    # Load the data
    df_online = pd.read_csv(file_path)
    # pivot table to better get features for further data process. Here the unit column is removed. 
    pivot_df = df_online.pivot_table(
        index=['exp_id', 'time_hr'],
        columns='parameter',
        values='value',
        aggfunc='first'  # Use 'first' or 'mean' if there are duplicates
    ).reset_index()

    
    return pivot_df

def load_od_data(file_path):
    """
    Load the time-series OD600 data CSV file.
    
    Args:
        file_path (str): Path to the Time_series_OD_data.csv file.
    
    Returns:
        pd.DataFrame: Loaded OD600 measurements.
    """
    # Load the data
    df_od = pd.read_csv(file_path)

        
    return df_od

def OD_interpolation_datasets(df_online, df_od):
    """
    Align the online parameters and OD600 data based on nearest timestamp.
    """

    combined_data = pd.merge(df_online[['time_hr', 'exp_id']],
                         df_od[['time_hr', 'exp_id', 'OD600']],
                         on=['time_hr', 'exp_id'],
                         how='outer')
    
    # Remove time_hr < 0 rows
    combined_data_cleaned = combined_data[combined_data['time_hr'] >= 0]
    # Verify if any such rows remain
    if combined_data_cleaned['time_hr'].min() < 0:
        raise ValueError("There are still unexpected negative-like timestamps remaining.")

    return combined_data_cleaned

    
def interpolate_experiment(group):
    # Set 'time_hr' as the index of the DataFrame
    indexed_group = group.set_index('time_hr')
    
    # Convert object columns to appropriate numeric data types
    numeric_group = indexed_group.infer_objects()
    
    # Interpolate missing values using linear method
    interpolated_group = numeric_group.interpolate(method='linear')
    
    # Reset the index to turn 'time_hr' back into a column
    reset_interpolated_group = interpolated_group.reset_index()
    
    # Return the final dataset
    return reset_interpolated_group

def align_datasets(df_online, df_od):
    """
    Align the online parameters and OD600 data based on nearest timestamp.
    """

    df_aligned = pd.merge(df_online, df_od, on=['time_hr', 'exp_id'])
    
    return df_aligned

def standardize_data(df, feature_cols, target_col):
    """
    Standardizes the specified features in the DataFrame.

    Parameters:
    - df: pd.DataFrame containing the dataset with timestamps.
    - feature_cols: list of column names to be standardized.
    - target_col: string, the column name of the target variable.

    Returns:
    - standardized_df: pd.DataFrame with standardized features and original timestamps and target.
    - scaler: StandardScaler object fitted to the data, useful for inverse transforming or transforming new data.
    """

    # Separate features and target
    features = df[feature_cols]
    target = df[target_col]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform features
    standardized_features = scaler.fit_transform(features)
    # standardized_target = scaler.fit_transform(target)

    # Convert back to a DataFrame for easier handling
    standardized_features_df = pd.DataFrame(standardized_features, columns=feature_cols, index=features.index)
    # standardized_target_df = pd.DataFrame(standardized_target, columns=target_col, index=target.index)

    # Assuming the original DataFrame has a 'timestamp' column you want to keep
    # standardized_df = pd.concat([df['exp_id'], df['time_hr'], standardized_features_df, standardized_target_df], axis=1)
    standardized_df = pd.concat([df['exp_id'], df['time_hr'], standardized_features_df, target], axis=1)
    

    return standardized_df, scaler

def standardize_data_only_features(df, feature_cols):

    # Separate features and target
    features = df[feature_cols]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform features
    standardized_features = scaler.fit_transform(features)

    # Convert back to a DataFrame for easier handling
    standardized_features_df = pd.DataFrame(standardized_features, columns=feature_cols, index=features.index)

    standardized_df = pd.concat([df['exp_id'], df['time_hr'], standardized_features_df], axis=1)
    
    return standardized_df

def standardize_multi_exp(training_dataset, feature_cols, target_col,exp_id):
    dfs = []
    for series_id in exp_id:
        df_series = training_dataset[training_dataset['exp_id'] == series_id]
        standardize_df_series, scalar = standardize_data(df_series, feature_cols, target_col)
        dfs.append(standardize_df_series)
    standardize_training_data = pd.concat(dfs, ignore_index=True)
    return standardize_training_data, scalar


def preprocess_data(df_online, df_od, feature_cols, target_col):
    """
    Load the data, process it, and return merged dataset ready for analysis.
    
    Args:
        df_online: Dataframe of the online parameters CSV file.
        df_od : Dataframe of the OD600 data CSV file.
    
    Returns:
        pd.DataFrame: Fully processed and merged dataset.
    """
    
    # Align dataset before interpolation
    df_merged = OD_interpolation_datasets(df_online, df_od)
    
    # Group by 'experiment_id' and apply the interpolation function
    interpolated_data = df_merged.groupby('exp_id').apply(interpolate_experiment)
    # Reset the index to make it easier to work with. Remove rows with NaN (NaN comes from no OD data for T3-080724 and T6-080724)
    interpolated_data = interpolated_data.reset_index(drop=True).dropna()
    # align dataset to get the training dataset
    training_dataset = align_datasets(df_online, interpolated_data)

    # Standardize the dataset from different experiments
    all_exp_id = training_dataset['exp_id'].unique()
    standardize_training_data, scalar = standardize_multi_exp(training_dataset, feature_cols, target_col, all_exp_id)

    return standardize_training_data, scalar
