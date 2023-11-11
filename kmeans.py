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


def initial_centroids(data , k):
    return [data.sample(n=1, axis = 0).T for i in range(k)]
    ## TODO Implement KMeans++ after :D
    
def kmeans(data, k):
    centroids = initial_centroids(data, k)
    
    old_centroids = []
    for i in range(k):
        old_centroids.append(np.zeros(len(data.columns)))
    
    while stopping_condition(old_centroids, centroids):
        # No clue why its called s
        s = []
        
        # Assigns all points to no cluster, resets s to 0's
        for i in range(k):
            s.append(np.zeros(len(data.columns)))
            clusters = [set() for _ in range(k)]
        
        # Calculate distances to centroids and assign
        for _,row in data.iterrows():
            distances = np.zeros(k).tolist()
            for i in range(k):
                distances[i] = DistanceCalculator.euclidian_distance(row, centroids[i])
            
            cluster = np.array(distances).argmin()
            clusters[cluster].add(tuple(row.to_list()))
            s[cluster] += row.to_numpy()
        
        # Recalculate centroids
        for i in range(k):
            old_centroids[i] = centroids[i]
            centroids[i] = s[i] / float(len(clusters[i]))

        
            

def stopping_condition(old_centroids, centroids, threshold=0.001):
    total_movement = sum(DistanceCalculator.euclidian_distance(np.array(oc), np.array(c)) for oc, c in zip(old_centroids, centroids))
    return total_movement > threshold


if __name__ == "__main__":
    main()
    
    