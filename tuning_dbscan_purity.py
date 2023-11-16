import sys
import numpy as np
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
    if len(sys.argv) < 4:
        print("Usage: python tuning_dbscan.py <Filename> <Threshold> <Iterations>")
        sys.exit(1)

    file_path = sys.argv[1]
    uncleaned_data = load_data(file_path)
    data, class_labels = preprocess_data(uncleaned_data)

    # Initialize the best performance metrics
    best_performance = -np.inf
    best_epsilon = None
    best_min_points = None

    # Define the initial range for epsilon and min_points
    epsilon_range = (0.1, 10)
    min_points_range = (2, 50)

    
    threshold = float(sys.argv[2])

    # Define the number of iterations for tuning
    num_iterations = int(sys.argv[3])

    for i in range(num_iterations):
        # Sample epsilon and min_points within the current range
        epsilon = np.random.uniform(*epsilon_range)
        min_points = np.random.randint(*min_points_range)

        labels = dbscan(data, epsilon, min_points)

        # Evaluate the clustering performance
        purity, _ = calculate_purity(labels, class_labels)  # Replace with your actual performance metric function

        # Adjust the ranges based on the performance
        epsilon_range, min_points_range = adjust_ranges(epsilon, min_points, purity, epsilon_range, min_points_range, threshold)

        # Update the best parameters if current performance is better
        if purity > best_performance:
            best_performance = purity
            best_epsilon = epsilon
            best_min_points = min_points

    # Output the best DBSCAN parameters and clustering results
    print("Best DBSCAN Parameters:")
    print(f"Epsilon={best_epsilon}, Min Points={best_min_points}")
    print(f"Best Performance: {best_performance}")

if __name__ == "__main__":
    main()
