import pandas as pd
import numpy as np
from pandas.api.types import is_any_real_numeric_dtype

def load_data(file_path):
    # Read the CSV with the expected number of columns
    data = pd.read_csv(file_path, header=None)
    return data

def preprocess_data(data):
    # Check if the last column is a class label, indicated by a 0 in the first row.
    has_class_label = int(data.iloc[0, 0]) == 0
    
    # If there's a class label, separate it from the features.
    if has_class_label:
        class_labels = data.iloc[1:, 0]  # Extract class labels from the data
        data = data.iloc[1:, 1:]  # Remove class labels from the data
    else:
        class_labels = None
        data = data.iloc[1:, :]

    data.reset_index(drop=True, inplace=True)

    # If class labels exist, add them back as the first column

    # Convert all columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Replace '?' with NaN and drop rows with NaN values
    data = data.replace('?', np.nan).dropna()

    # Normalize the data
    data = normalize(data)

    return data, class_labels


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