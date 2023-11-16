import logging
import numpy as np
import sys
import os
from preprocessing import load_data, preprocess_data
from kmeans import kmeans, calculate_sse, calculate_purity  # Assuming these functions are correctly implemented

def main():
    # Starting parameters for tuning
    k_values = range(2, 50)  # Example range for k
    threshold_values = [0.001]
    while threshold_values[-1] < 100 / 2:
        threshold_values.append(threshold_values[-1] * 2)

    # Ensure proper command line arguments
    if len(sys.argv) < 2:
        print("Usage: python tuningScript.py <Filename>")
        sys.exit()

    # Extract the base filename to name the log file
    filename = sys.argv[1]
    data, class_labels = preprocess_data(load_data(filename))
    
    # Check if class labels are present
    is_classification_dataset = class_labels is not None
    
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    log_filename = f"{base_filename}_kmeans_tuning.log"

    # Initialize logging
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    # Initialize best parameters
    best_sse = np.inf
    best_k = None
    best_threshold = None
    best_purity = 0 if is_classification_dataset else None

    # Tuning k
    for k in k_values:
        centroids, assignments = kmeans(data, k)
        sse = calculate_sse(data, centroids, assignments)
        logging.info(f"k={k}: SSE={sse}")

        # Update best parameters based on SSE and, if applicable, purity
        update_params = False
        if is_classification_dataset:
            purity, _ = calculate_purity(assignments, class_labels)
            logging.info(f"k={k}: Purity={purity}")
            # Update if purity is better, or if purity is equal and SSE is lower
            if purity > best_purity or (purity == best_purity and sse < best_sse):
                update_params = True
        elif sse < best_sse:
            update_params = True

        # Update best parameters
        if update_params:
            best_sse = sse
            best_k = k
            if is_classification_dataset:
                best_purity = purity

    # Output best parameters for SSE
    print(f"Best SSE parameters: k={best_k} with SSE: {best_sse}")
    logging.info(f"Best SSE parameters: k={best_k} with SSE: {best_sse}")

    # Output best parameters for purity if applicable
    if is_classification_dataset:
        print(f"Best Purity parameters: k={best_k} with Purity: {best_purity}")
        logging.info(f"Best Purity parameters: k={best_k} with Purity: {best_purity}")


if __name__ == "__main__":
    main()
