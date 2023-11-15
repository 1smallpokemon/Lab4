import logging
import numpy as np
import sys
import os
from preprocessing import load_data, preprocess_data
from hclustering import hclusters, compute_clusters, compute_sse

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

    # Parameter ranges for tuning
    threshold_values = list()
    threshold_min = 0.01
    threshold_max = 0.09
    step = 0.01
    i = threshold_min
    while i <= threshold_max:
        threshold_values.append(i)
        i += step

    best_sse = np.inf
    best_threshold = None
    best_purity = 0

    for threshold in threshold_values:
        tree = hclusters((data, class_labels))
        clusters = compute_clusters(threshold, tree)
        # calculate sse
        sse = compute_sse(clusters)
        logging.info(f"threshold={threshold}: SSE={sse}")

        if sse < best_sse:
            best_sse = sse
            best_threshold = threshold

    # TODO - If class labels are available, further tune for purity

    # Output best parameters for SSE
    print(f"Best SSE parameters: threshold={best_threshold} with SSE: {best_sse}")
    logging.info(f"Best SSE parameters: threshold={best_threshold} with SSE: {best_sse}")

    # Output best parameters for purity if applicable

if __name__ == "__main__":
    main()