import logging
import numpy as np
import sys
from preprocessing import load_data, preprocess_data
from kmeans import kmeans, calculate_sse  # Make sure kmeans and calculate_sse are imported from your kmeans script

def main():
    # Placeholder for the best model's metrics
    best_sse = np.inf
    best_k = None
    best_threshold = None

    # Starting parameters for tuning
    k_values = range(2, 5)  # Example range for k
    threshold_values = [0.001]
    while threshold_values[-1] < 100 / 2:
        threshold_values.append(threshold_values[-1] * 2)

    # Ensure proper command line arguments
    if len(sys.argv) < 2:
        print("Usage: python tuningScript.py <Filename>")
        sys.exit()

    filename = sys.argv[1]
    data = preprocess_data(load_data(filename))
    logfilename =f"{filename.split('.')[0]}_kmeans_tuning.log"

    # Initialize logging
    logging.basicConfig(filename=logfilename, 
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Tuning k first
    for k in k_values:
        logging.info(f"Evaluating k-means with k: {k}")
        centroids, assignments = kmeans(data, k)
        sse = calculate_sse(data, centroids, assignments)
        logging.info(f"SSE for k={k}: {sse}")

        if sse < best_sse:
            best_sse = sse
            best_k = k

    # Tuning threshold next, using the best k
    for threshold in threshold_values:
        logging.info(f"Evaluating k-means with k: {best_k} and threshold: {threshold}")
        centroids, assignments = kmeans(data, best_k, threshold)  # Assume kmeans() can now accept threshold as an argument
        sse = calculate_sse(data, centroids, assignments)
        logging.info(f"SSE for threshold={threshold}: {sse}")

        if sse < best_sse:
            best_sse = sse
            best_threshold = threshold

    # Output the best parameters
    print(f"Best parameters: k={best_k}, threshold={best_threshold} with SSE: {best_sse}")
    logging.info(f"Best parameters: k={best_k}, threshold={best_threshold} with SSE: {best_sse}")

if __name__ == "__main__":
    main()
