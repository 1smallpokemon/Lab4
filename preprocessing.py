import pandas as pd
import numpy as np

def load_data(file_path):
    """Loads data from a CSV file, excluding columns marked with 0 in the first row."""
    data = pd.read_csv(file_path, header=None)
    # Get the binary vector from the first row
    use_columns = data.iloc[0, :].astype(bool).tolist()
    # Exclude columns based on the binary vector
    data = data.loc[:, use_columns]
    # Remove the first row as it is just the binary vector
    data = data.iloc[1:, :]
    return data.reset_index(drop=True)

def preprocess_data(data):
    """Preprocesses the data for clustering."""
    # Convert all columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # Replace '?' with NaN and drop rows with NaN values
    data = data.replace('?', np.nan).dropna()
    # Normalize the data
    return normalize(data)

def normalize(data):
    """Normalizes the numeric columns in the data."""
    for col in data.columns:
        if is_any_real_numeric_dtype(data[col]):
            data[col] = normalize_column(data[col].values)
    return data

def normalize_column(column):
    """Normalizes a single column using min-max scaling."""
    mins = np.min(column)
    maxs = np.max(column)
    return (column - mins) / (maxs - mins) if (maxs - mins) != 0 else column