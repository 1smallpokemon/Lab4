import pandas as pd
import numpy as np

class DistanceCalculator:
    """
    x and y are rows of a pandas dataframe, each representing a datapoint
    """
    def euclidian_distance(x, y):
        # dist = sqrt(sum((x-y)^2))
        diff = x - y
        return np.sqrt(np.sum(diff ** 2, axis = 0))
    
    def squared_euclidian_distance(x, y):
        # dist = sum((x-y)^2))
        diff = x - y
        return np.sum(diff ** 2, axis=0)
    
    def manhattan_distance(x, y):
        diff = x - y
        return np.sum(abs(diff),axis=0)
    
    def minkowski_distance(x, y, l):
        """
        l = lambda 
        l = 1 manhattan distance, l = 2 euclidian distance
        """
        diff = x - y
        return np.power(np.sum((diff ** l),axis=0), 1/l)
    
    def chebyshev_distance(x, y):
        diff = x - y
        return np.max(abs(diff))