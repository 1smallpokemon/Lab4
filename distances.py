import numpy as np

class Distance:
    #TODO normalize numerical columns
    def normalize(X):
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        return np.nan_to_num((X - mins) / (maxs - mins))
    
    def euclidean_dist(x1, x2):
        diff = x1-x2
        return np.sqrt(diff @ diff)
    
    def squared_euclidean_dist(x1, x2):
        diff = x1 - x2
        return diff @ diff

    def manhattan_dist(x1, x2):
        return sum(abs(x1-x2))
    
    def minkowski_dist(power_root, x1, x2):
        """power_root=1 is Manhattan distance
        power_root=2 is Euclidean distance
        Partial application needed for compatibility with predict.
        """
        diff = x1 - x2
        return np.power(sum(diff ** power_root), 1/power_root)
    
    def chebyshev_dist(x1, x2):
        return np.max(np.abs(x1-x2))
    
    def cosine_similarity(x1, x2):
        if sum(x1) == 0 or sum(x2) == 0:
            return 0 #not sure what to do here
        return -((x1 @ x2) / (np.sqrt(x1 @ x1)*np.sqrt(x2 @ x2))) #return the opposite of the cosine similarity to be compatible with sort by increasing value
    
