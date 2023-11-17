import sys
import numpy as np
import pandas as pd
import logging
import math
import os
from preprocessing import load_data, preprocess_data
from dbscan import dbscan
from kmeans import calculate_sse

# Function to adjust ranges based on the performance metric
def adjust_ranges(epsilon, min_points, performance_metric, epsilon_range, min_points_range, desired_threshold):
    if performance_metric < desired_threshold:
        # Contract the ranges
        epsilon_range = (max(epsilon_range[0], epsilon - 0.1), min(epsilon_range[1], epsilon + 0.1))
        min_points_range = (max(min_points_range[0], min_points - 1), min(min_points_range[1], min_points + 1))
    else:
        # Expand the ranges
        epsilon_range = (max(epsilon_range[0], epsilon - 0.5), min(epsilon_range[1], epsilon + 0.5))
        min_points_range = (max(min_points_range[0], min_points - 5), min(min_points_range[1], min_points + 5))
    
    # Ensure the new ranges are valid
    epsilon_range = (max(0.1, epsilon_range[0]), max(epsilon_range[0], epsilon_range[1]))
    min_points_range = (max(2, min_points_range[0]), max(min_points_range[0], min_points_range[1]))
    
    return epsilon_range, min_points_range

def main():
    
    if len(sys.argv) < 4:
        print("Usage: python tuning_dbscan.py <Filename> <Threshold> <Iterations>")
        sys.exit(1)


    file_path = sys.argv[1]
    
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    log_filename = f"{base_filename}_dbscan_tuning_sse.log"

    # Initialize logging
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    uncleaned_data = load_data(file_path)
    data, class_labels = preprocess_data(uncleaned_data)

    # Initialize the best performance metrics
    best_performance = np.inf
    best_epsilon = None
    best_min_points = None

    # Define the initial range for epsilon and min_points
    epsilon_range = (.001, math.sqrt(2)/4)
    min_points_range = (2, 6)

    
    threshold = float(sys.argv[2])

    # Define the number of iterations for tuning
    num_iterations = int(sys.argv[3])

    for i in range(num_iterations):
        # Sample epsilon and min_points within the current range
        epsilon = np.random.uniform(*epsilon_range)
        min_points = np.random.randint(*min_points_range)

        logging.info(f"Iteration {i+1}/{num_iterations}: Testing epsilon={epsilon}, min_points={min_points}")
        labels = dbscan(data, epsilon, min_points)

        # Count the number of noise points
        noise_count = np.sum(np.array(labels) == -1)
        logging.info(f"Noise count: {noise_count}")


        # Create a DataFrame to store centroids
        cluster_labels = [label for label in np.unique(labels) if label != -1]  # Exclude noise
        centroids = pd.DataFrame(columns=data.columns)

        for label in cluster_labels:
            cluster_data = data.iloc[np.where(labels == label)[0]]
            centroids.loc[label] = cluster_data.mean()

        # Evaluate the clustering performance
        sse = calculate_sse(data, centroids, np.array(labels))
        logging.info(f"Epsilon={epsilon}, Min Points={min_points}, SSE={sse}")

        
        # Adjust the ranges based on the performance
        epsilon_range, min_points_range = adjust_ranges(epsilon, min_points, sse, epsilon_range, min_points_range, threshold)

        # Update the best parameters if current performance is better
        if sse < best_performance:
            best_performance = sse
            best_epsilon = epsilon
            best_min_points = min_points
            logging.info(f"New best performance: SSE={sse}, Epsilon={epsilon}, Min Points={min_points}, Noise count: {noise_count}")


    # Output the best DBSCAN parameters and clustering results
    logging.info("Best DBSCAN Parameters:")
    logging.info(f"Epsilon={best_epsilon}, Min Points={best_min_points}")
    logging.info(f"Best Performance: {best_performance}")
    print(f"Best DBSCAN Parameters: Epsilon={best_epsilon}, Min Points={best_min_points}, Best Performance: {best_performance}")

    # Run DBSCAN one more time with the best parameters to get the final labels
    final_labels = dbscan(data, best_epsilon, best_min_points)
    clusters = {}
    noise = []
    for idx, label in enumerate(final_labels):
        if label == -1:
            noise.append(idx)
        else:
            clusters.setdefault(label, []).append(idx)
    
    # Display and log the final clusters with data points
    logging.info("Final Clusters and Noise Points:")
    for cluster_id, points in clusters.items():
        logging.info(f"Cluster {cluster_id}: {len(points)} points")
        for point in points:
            logging.info(f"Data Point: {data.iloc[point,:]}")
        print(f"Cluster {cluster_id}: {len(points)} points")
        
    # Display and log noise points
    logging.info(f"Noise: {len(noise)} points")
    for point in noise:
        logging.info(f"Noise Point: {data.iloc[point,:]}")
    print(f"Noise: {len(noise)} points")


if __name__ == "__main__":
    main()
