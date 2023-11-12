import sys
import pandas as pd
import numpy as np
from preprocessing import *
from DistanceCalculator import DistanceCalculator

def main():
    # call preprocessing script, returns dataframe
    argc = len(sys.argv)
    
    if argc < 3:
        print("Usage: python kmeans.py <Filename> <k>")
    
    try:
        uncleaned_data =  load_data(sys.argv[1])
        data =  preprocess_data(uncleaned_data)
        
        k = int(sys.argv[2])
        if k <= 0 or k > len(data):
            raise ValueError("Invalid number of clusters")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    centroids, clusters = kmeans(data, k)
    print("Centroids:", centroids)
    print("Clusters:", clusters)

def inital_centroids_kmeans_pp(data, k):
    # 1 Compute the overall centroid of the dataset
    overall_centroid = data.mean()

    # First centroid is farthest from the overall centroid
    distances = data.apply(lambda x: np.linalg.norm(x - overall_centroid), axis=1)
    centroids = [data.iloc[np.argmax(distances)].to_frame().T]

    # Select remaining centroids
    for _ in range(1, k):
        distances = data.apply(lambda x: min([np.linalg.norm(x - c) for c in centroids]), axis=1)
        next_centroid = data.iloc[np.argmax(distances)].to_frame().T
        centroids.append(next_centroid)

    return pd.concat(centroids, ignore_index=True)


def inital_centroids(data , k):
    return [data.sample(n=1, axis = 0).T for i in range(k)]
    ## TODO Implement KMeans++ after :D

def kmeans(data, k):
    centroids = initial_centroids_kmeans_pp(data, k)
    old_centroids = pd.DataFrame(np.zeros_like(centroids.values))
    assignments = np.zeros(data.shape[0])
    old_assignments = np.ones(data.shape[0]) * -1
    max_iterations = 100
    threshold = 0.001
    old_sse = None

    for iteration in range(max_iterations):
        # Assignment step
        for i, row in data.iterrows():
            distances = [DistanceCalculator.euclidean_distance(row, centroids.iloc[j]) for j in range(k)]
            assignments[i] = np.argmin(distances)

        # Check for stopping condition based on assignments
        if np.array_equal(assignments, old_assignments):
            break

        # Update centroids
        for i in range(k):
            cluster_points = data[assignments == i]
            if not cluster_points.empty:
                centroids.iloc[i] = cluster_points.mean()

        # Check for stopping condition based on centroids
        centroids_changed = sum(DistanceCalculator.euclidean_distance(old_centroids.iloc[i], centroids.iloc[i]) for i in range(k))
        if centroids_changed <= threshold:
            break

        # Calculate SSE and check for stopping condition
        sse = calculate_sse(data, centroids, assignments)
        if old_sse is not None and abs(old_sse - sse) <= threshold:
            break

        # Prepare for next iteration
        old_centroids = centroids.copy()
        old_assignments = assignments.copy()
        old_sse = sse
    
    return centroids, assignments        
def stopping_condition(old_centroids, centroids, old_assignments, assignments, data, threshold=0.001):
    # Condition 1: Check if there's a minimum change in assignments
    if np.array_equal(assignments, old_assignments):
        return True  # Stopping condition met

    # Condition 2: Check if there's a minimum change in centroids
    centroids_changed = sum(DistanceCalculator.euclidean_distance(np.array(oc), np.array(c)) for oc, c in zip(old_centroids, centroids))
    if centroids_changed <= threshold:
        return True  # Stopping condition met

    # Condition 3: Check for insignificant decrease in SSE
    new_sse = calculate_sse(data, centroids, assignments)
    old_sse = calculate_sse(data, old_centroids, old_assignments)
    if abs(old_sse - new_sse) <= threshold:
        return True  # Stopping condition met

    return False  # Continue the algorithm

def calculate_sse(data, centroids, assignments):
    sse = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[assignments == i]
        sse += np.sum((cluster_points - centroid) ** 2)
    return sse
            
def stopping_condition(old_centroids, centroids, threshold=0.001):
    total_movement = sum(DistanceCalculator.euclidian_distance(np.array(oc), np.array(c)) for oc, c in zip(old_centroids, centroids))
    return total_movement > threshold


if __name__ == "__main__":
    main()
    
    