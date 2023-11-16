import logging
import numpy as np
import pandas as pd
import sys
import os
from preprocessing import load_data, preprocess_data
from hclustering import hclusters, compute_clusters, compute_sse
from kmeans import calculate_purity

def main():
    if len(sys.argv) < 2:
        print("Usage: python tuningScript.py <Filename>")
        sys.exit()

    # Extract the base filename to name the log file
    filename = sys.argv[1]
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    log_filename = f"{base_filename}_hclusters_tuning.log"

    # Initialize logging
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Load and preprocess data
    data, class_labels = preprocess_data(load_data(filename))

    # Check if class labels are present
    is_classification_dataset = class_labels is not None

    # Parameter ranges for tuning
    threshold_values = list()
    threshold_min = 0.1
    threshold_max = 0.9
    step = 0.1
    i = threshold_min
    while i <= threshold_max:
        threshold_values.append(i)
        i += step

    best_sse = np.inf
    best_threshold = None
    best_purity = 0 if is_classification_dataset else None

    for threshold in threshold_values:
        tree = hclusters((data, class_labels))
        clusters = compute_clusters(threshold, tree)
        # calculate sse
        sse = compute_sse(clusters)
        logging.info(f"threshold={threshold}: SSE={sse}")

        if sse < best_sse:
            best_sse = sse
            best_threshold = threshold

        # Update best parameters based on SSE and, if applicable, purity
        update_params = False
        if is_classification_dataset:
            assignments = list()
            cluster_num = 0
            for cluster in clusters:
                num_points = list()
                num_points = cluster.get_data_points(num_points)
                for i in range(len(num_points)):
                    assignments.append(cluster_num)
                cluster_num += 1
            assignments = pd.Series(assignments)
            purity, _ = calculate_purity(assignments, class_labels)
            logging.info(f"threshold={threshold}: Purity={purity}")
            # Update if purity is better, or if purity is equal and SSE is lower
            if purity > best_purity or (purity == best_purity and sse < best_sse):
                update_params = True
        elif sse < best_sse:
            update_params = True

        # Update best parameters
        if update_params:
            best_sse = sse
            best_threshold = threshold
            if is_classification_dataset:
                best_purity = purity

    # Output best parameters for SSE
    print(f"Best SSE parameters: threshold={best_threshold} with SSE: {best_sse}")
    logging.info(f"Best SSE parameters: threshold={best_threshold} with SSE: {best_sse}")

    # Output best parameters for purity if applicable
    if is_classification_dataset:
        print(f"Best Purity parameters: threshold={best_threshold} with Purity: {best_purity}")
        logging.info(f"Best Purity parameters: threshold={best_threshold} with Purity: {best_purity}")

    # Output best parameters for purity if applicable

if __name__ == "__main__":
    main()