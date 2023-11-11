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
    except:
        print("Couldn't parse command line")
        
    kmeans(data, k)
    
    pass


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
    
    