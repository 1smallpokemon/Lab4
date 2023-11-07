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
        
        k = sys.argv[2]
    except:
        print("Couldn't parse command line")
        
    
    pass


def initial_centroids(data , k):
    return data.random_sample(n=k)
    ## TODO Implement KMeans++ after :D
    
def kmeans(data, k):
    centriods = initial_centroids(data, k)
    
    while stopping_condition(data, k):
        # No clue why its called s
        s = []
        
        # Assigns all points to no cluster, resets s to 0's
        for i in range(k):
            s.append(np.zeros(len(data[:1])).tolist())
            clusters = [set() for _ in range(k)]
        
        for _,row in data.iterrows():
            distances = np.zeros(k).tolist
            for i in range(k):
                distances[i] = DistanceCalculator.euclidian_distance(row, centriods[k])
            
            cluster = np.array(distances).argmin()
            
            clusters[cluster] | row
            
            s[i]
            
            
        
def stopping_condition(data, k):
    return

if __name__ == "__main__":
    main()
    
    