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

def initial_centroids_kmeans_pp(data, k):
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


def initial_centroids(data , k):
    return [data.sample(n=1, axis = 0).T for i in range(k)]
    ## TODO Implement KMeans++ after :D

def kmeans(data, k, threshold = 0.001):
    centroids = initial_centroids_kmeans_pp(data, k)
    old_centroids = pd.DataFrame(np.zeros_like(centroids.values))
    assignments = np.zeros(data.shape[0])
    old_assignments = np.ones(data.shape[0]) * -1
    max_iterations = 100
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
    
    for i in range(k):
        cluster_points = data[assignments == i]
        cluster_center = centroids.iloc[i]
        distances = cluster_points.apply(lambda x: DistanceCalculator.euclidean_distance(x, cluster_center), axis=1)
        
        # Calculating statistics for the cluster
        max_dist = distances.max()
        min_dist = distances.min()
        avg_dist = distances.mean()
        sse = np.sum(distances ** 2)

        # Printing cluster information
        print(f"Cluster {i}:")
        print(f"Center: {cluster_center.to_list()}")
        print(f"Max Dist. to Center: {max_dist}")
        print(f"Min Dist. to Center: {min_dist}")
        print(f"Avg Dist. to Center: {avg_dist}")
        print(f"SSE: {sse}")
        print(f"{len(cluster_points)} Points:")
        for point in cluster_points.values:
            print(point)
            
        print() 
    
    return centroids, assignments

def calculate_sse(data, centroids, assignments):
    sse = 0
    for i, centroid in centroids.iterrows():
        cluster_points = data[assignments == i]
        sse += np.sum(np.square(cluster_points - centroid).sum(axis=1))  # axis=1 sums across columns
    return sse

            
def stopping_condition(old_centroids, centroids, threshold=0.001):
    total_movement = sum(DistanceCalculator.euclidian_distance(np.array(oc), np.array(c)) for oc, c in zip(old_centroids, centroids))
    return total_movement > threshold


if __name__ == "__main__":
    main()
    
    