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
    centroids = inital_centroids_kmeans_pp(data, k)
    
    old_centroids = pd.DataFrame(np.zeros_like(centroids.values))
    max_iterations = 100
    threshold = 0.001
     
    
    for _ in range(max_iterations):
        # Assigns all points to no cluster
        clusters = [set() for _ in range(k)]
        cluster_sums = np.zeros((k, data.shape[1]))
        
        # Calculate distances to centroids and assign
        for _, row in data.iterrows():
            distances = [DistanceCalculator.euclidean_distance(row, centroids.iloc[i]) for i in range(k)]
            cluster = np.argmin(distances)
            clusters[cluster].add(tuple(row))
            cluster_sums[cluster] += row
        
        # Recalculate centroids
        for i in range(k):
            if len(clusters[i]) > 0:
                centroids.iloc[i] = cluster_sums[i] / len(clusters[i])
        
        # Check stopping condition
        total_movement = sum(DistanceCalculator.euclidean_distance(old_centroids.iloc[i], centroids.iloc[i]) for i in range(k))
        if total_movement <= threshold:
            break
        old_centroids = centroids.copy()
    
    return centroids, clusters

        
            

def stopping_condition(old_centroids, centroids, threshold=0.001):
    total_movement = sum(DistanceCalculator.euclidian_distance(np.array(oc), np.array(c)) for oc, c in zip(old_centroids, centroids))
    return total_movement > threshold


if __name__ == "__main__":
    main()
    
    