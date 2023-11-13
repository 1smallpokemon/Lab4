import pandas as pd
import numpy as np
from pandas.api.types import is_any_real_numeric_dtype

def load_data(file_path):
    # Read the CSV file, skipping the first row
    data = pd.read_csv(file_path, header=None)
    use_columns = data.iloc[0, :].astype(bool).tolist()
    
    # Filter columns based on the binary inclusion vector
    data = data.loc[:, use_columns]
    
    # Remove the first row as it is just the binary vector
    data = data.iloc[1:]
    return data.reset_index(drop=True)

def preprocess_data(data):
    # Ensure data is not empty
    if data.empty:
        raise ValueError("No data available to process.")

    # Check if the last column is a class label, indicated by a 0 in the first row.
    has_class_label = not bool(data.iloc[0, -1])  # Using bool() directly for clarity

    class_labels = None
    if has_class_label:
        class_labels = data.iloc[:, -1]  # Extract class labels from the data
        data = data.iloc[:, :-1]  # Remove class labels from the data

    # Convert all columns to numeric, coerce errors, and drop rows with NaN values
    data = data.apply(pd.to_numeric, errors='coerce').dropna()

    # Normalize the data if it's not empty
    if not data.empty:
        data = normalize(data)
    else:
        raise ValueError("All rows contain NaN after conversion to numeric.")

    return data, class_labels  # Return both the features and the class labels (if present)

def normalize(data):
    # Normalizes the numeric columns in the data.
    for col in data.columns:
        if is_any_real_numeric_dtype(data[col]):
            data[col] = normalize_column(data[col].values)
    return data

def normalize_column(column):
    # Normalizes a single column using min-max scaling.
    mins = np.min(column)
    maxs = np.max(column)
    return (column - mins) / (maxs - mins) if (maxs - mins) != 0 else column