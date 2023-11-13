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

    # Parameter ranges for tuning
    k_values = range(2, 10)
    threshold_values = [0.001 * (2 ** i) for i in range(int(np.log2(0.01/0.001)) + 1)]

    best_sse = np.inf
    best_k = None
    best_threshold = None
    best_purity = 0

    # Tuning k
    for k in k_values:
        centroids, assignments = kmeans(data, k)
        sse = calculate_sse(data, centroids, assignments)
        logging.info(f"k={k}: SSE={sse}")
        
        if sse < best_sse:
            best_sse = sse
            best_k = k

    # If class labels are available, further tune for purity
    if class_labels is not None:
        for k in k_values:
            _, assignments = kmeans(data, k)
            purity, _ = calculate_purity(class_labels, assignments)
            logging.info(f"k={k}: Purity={purity}")

            if purity > best_purity:
                best_purity = purity
                best_k = k

    # Output best parameters for SSE
    print(f"Best SSE parameters: k={best_k} with SSE: {best_sse}")
    logging.info(f"Best SSE parameters: k={best_k} with SSE: {best_sse}")

    # Output best parameters for purity if applicable
    if class_labels is not None:
        print(f"Best Purity parameters: k={best_k} with Purity: {best_purity}")
        logging.info(f"Best Purity parameters: k={best_k} with Purity: {best_purity}")

if __name__ == "__main__":
    main()
