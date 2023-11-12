import pandas as pd
import numpy as np
from pandas.api.types import is_any_real_numeric_dtype

def load_data(file_path):
    #Loads data from a CSV file, excluding columns marked with 0 in the first row.
    with open(file_path, 'r') as file:
        first_line = file.readline().strip().split(',')
        # Determine the number of columns to expect based on the first line
        expected_cols = sum([1 for val in first_line if val == '1'])
    
    # Read the CSV with the expected number of columns
    data = pd.read_csv(file_path, header=None, usecols=range(expected_cols))
    
    # Get the binary vector from the first row and exclude columns accordingly
    use_columns = data.iloc[0].astype(bool).tolist()
    data = data.loc[:, use_columns]
    
    # Remove the first row as it is just the binary vector
    data = data.iloc[1:]
    return data.reset_index(drop=True)

def preprocess_data(data):
    # Check if the last column is a class label, indicated by a 0 in the first row.
    has_class_label = not data.iloc[0, -1].astype(bool)
    
    # If there's a class label, separate it from the features.
    if has_class_label:
        class_labels = data.iloc[1:, -1]  # Extract class labels from the data
        data = data.iloc[:, :-1]  # Remove class labels from the data
    else:
        class_labels = None

    # Convert all columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Replace '?' with NaN and drop rows with NaN values
    data = data.replace('?', np.nan).dropna()

    # Normalize the data
    data = normalize(data)

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