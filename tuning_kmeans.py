import logging
import numpy as np
import sys
import os
from preprocessing import load_data, preprocess_data
from kmeans import kmeans, calculate_sse, calculate_purity  # Assuming these functions are correctly implemented

def main():
    if len(sys.argv) < 2:
        print("Usage: python tuningScript.py <Filename>")
        sys.exit()

    # Extract the base filename to name the log file
    filename = sys.argv[1]
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    log_filename = f"{base_filename}_kmeans_tuning.log"

    # Initialize logging
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Load and preprocess data
    data, class_labels = preprocess_data(load_data(filename))

    # Check if class labels are present
    is_classification_dataset = class_labels is not None

    # Parameter ranges for tuning
    k_values = range(2, 10)
    threshold_values = [0.001 * (2 ** i) for i in range(int(np.log2(0.01/0.001)) + 1)]

    best_sse = np.inf
    best_k = None
    best_threshold = None
    best_purity = 0 if is_classification_dataset else None

    # Tuning k
    for k in k_values:
        centroids, assignments = kmeans(data, k)
        sse = calculate_sse(data, centroids, assignments)
        logging.info(f"k={k}: SSE={sse}")
        
        purity = None
        if is_classification_dataset:
            purity = calculate_purity(class_labels, assignments)
            logging.info(f"k={k}: Purity={purity}")

        # Update best parameters based on SSE and purity
        if sse < best_sse or (purity is not None and purity > best_purity):
            best_sse = sse
            best_k = k
            best_purity = purity

    # Tuning threshold
    for threshold in threshold_values:
        centroids, assignments = kmeans(data, best_k, threshold)
        sse = calculate_sse(data, centroids, assignments)
        logging.info(f"k={best_k}, threshold={threshold}: SSE={sse}")

        if sse < best_sse:
            best_sse = sse
            best_threshold = threshold

    # Output best parameters
    print(f"Best parameters: k={best_k}, threshold={best_threshold} with SSE: {best_sse}")
    if is_classification_dataset:
        print(f"Best purity: {best_purity}")
    logging.info(f"Best parameters: k={best_k}, threshold={best_threshold} with SSE: {best_sse}")
    if is_classification_dataset:
        logging.info(f"Best purity: {best_purity}")

if __name__ == "__main__":
    main()
