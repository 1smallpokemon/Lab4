import pandas as pd
import numpy as np
from io import StringIO
from preprocessing import *

def test_load_data():
    test_csv = StringIO("""0,1,0
                           1,2,3
                           4,5,6
                           7,8,9""")
    expected_output = pd.DataFrame({
        1: [2, 5, 8]
    })
    
    loaded_data = load_data(test_csv)
    assert loaded_data.equals(expected_output), "load_data function failed to exclude columns marked with 0."

def test_preprocess_data():
    test_data = pd.DataFrame({
        'A': [1, '?', 3],
        'B': [4, 5, '?']
    })
    
    # After preprocessing, we should have only the rows without '?'
    expected_output = pd.DataFrame({
        'A': [1.0],
        'B': [4.0]
    })

    preprocessed_data = preprocess_data(test_data)
    assert preprocessed_data.equals(expected_output), "preprocess_data function failed to handle '?' correctly."

def test_normalize():
    test_data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    # After normalization, all values should be between 0 and 1
    normalized_data = normalize(test_data)
    assert normalized_data['A'].min() >= 0 and normalized_data['A'].max() <= 1, "Normalization failed for column 'A'"
    assert normalized_data['B'].min() >= 0 and normalized_data['B'].max() <= 1, "Normalization failed for column 'B'"

def run_tests():
    test_load_data()
    test_preprocess_data()
    test_normalize()
    print("All tests passed.")

# Run the tests
run_tests()
