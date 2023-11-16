import sys
import numpy as np
import logging
from preprocessing import load_data, preprocess_data
from dbscan import dbscan
from kmeans import calculate_purity

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
    logging.basicConfig(filename='tuning_dbscan.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

    if len(sys.argv) < 4:
        logging.error("Usage: python tuning_dbscan.py <Filename> <Threshold> <Iterations>")
        sys.exit(1)

    file_path = sys.argv[1]
    uncleaned_data = load_data(file_path)
    data, class_labels = preprocess_data(uncleaned_data)

    # Initialize the best performance metrics
    best_performance = -np.inf
    best_epsilon = None
    best_min_points = None
    best_noise_count = None  

    # Define the initial range for epsilon and min_points
    epsilon_range = (0.1, 10)
    min_points_range = (2, 50)

    threshold = float(sys.argv[2])
    num_iterations = int(sys.argv[3])

    for i in range(num_iterations):
        epsilon = np.random.uniform(*epsilon_range)
        min_points = np.random.randint(*min_points_range)

        logging.info(f"Iteration {i+1}/{num_iterations}: Testing epsilon={epsilon}, min_points={min_points}")
        labels = dbscan(data, epsilon, min_points)

        # Count the number of noise points
        noise_count = np.sum(np.array(labels) == -1)
        logging.info(f"Noise count: {noise_count}")

        purity, _ = calculate_purity(labels, class_labels)
        logging.info(f"Epsilon={epsilon}, Min Points={min_points}, Purity={purity}")

        epsilon_range, min_points_range = adjust_ranges(epsilon, min_points, purity, epsilon_range, min_points_range, threshold)

        if purity > best_performance:
            best_performance = purity
            best_epsilon = epsilon
            best_min_points = min_points
            best_noise_count = noise_count
            logging.info(f"New best performance: Purity={purity}, Epsilon={epsilon}, Min Points={min_points}, Noise count: {noise_count}")

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

    # Display the final clusters
    for cluster_id, points in clusters.items():
        print(f"Cluster {cluster_id}: {len(points)} points")
    print(f"Noise: {len(noise)} points")

if __name__ == "__main__":
    main()