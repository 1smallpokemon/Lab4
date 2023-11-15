import sys
import pandas as pd
import numpy as np
from preprocessing import *
from DistanceCalculator import DistanceCalculator
import json

class DendrogramNode:
    def __init__(self, type, height, nodes, data):
        self.type = type
        self.height = height
        self.nodes = nodes
        self.data = data
        #print(self.type, self.height, self.nodes, self.data)

    def get_data_points(self, data_points):
        """
        return a list of all the data points connected to this node
        """
        #data_points = list()
        if self.type == 'leaf':
            data_points.append(self.data)
            return data_points
        for node in self.nodes:
            data_points = node.get_data_points(data_points)
        return data_points

    def combine_nodes(self, other_node, parent_type, parent_height):
        # parent type usually "node"
        # but if combining the last two clusters, type should be "root"
        new_nodes = list()
        new_nodes.append(self)
        new_nodes.append(other_node)
        return DendrogramNode(parent_type, parent_height, new_nodes, None)
    
    def print_data_point(self, data_point):
        return data_point.astype(str).str.cat(sep=', ')
    
    def convert_to_dict(self):
        if self.type == "leaf":
            return {
                "type": self.type,
                "height": self.height, 
                "data": self.print_data_point(self.data)
            }
        nodes_list = list()
        for node in self.nodes:
            nodes_list.append(node.convert_to_dict())
        return {
            "type": self.type,
            "height": self.height, 
            "nodes": nodes_list
        }


def main():
    """
    takes in filename and threshold (if no threshold was passed, threshold is None)
    """
    # call preprocessing script, returns dataframe
    argc = len(sys.argv)
    
    if argc < 2 or argc > 3:
        print("Usage: python hclustering.py <Filename> <threshold (optional)>")
    
    try:
        uncleaned_data =  load_data(sys.argv[1])
        data =  preprocess_data(uncleaned_data)
        
        if argc > 2:
            threshold = float(sys.argv[2])
        else:
            threshold = None
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    tree = hclusters(data)
    output_json(tree)
    if threshold is not None:
        print("CLUSTERS:")
        clusters = compute_clusters(threshold, tree)
        print("SSE=", compute_sse(clusters))
        return clusters
    
    
def output_json(tree):
    tree_dict = tree.convert_to_dict()
    print(json.dumps(tree_dict, indent=4))
    
def compute_clusters(threshold, tree):
    clusters = cut_at_threshold(tree, threshold)
    count = 1
    for cluster in clusters:
        print("Cluster", count)
        #print(json.dumps(cluster.convert_to_dict(), indent=4))
        evaluate_cluster(cluster)
        count += 1
    return clusters

def compute_sse(clusters):
    sse = 0
    for cluster in clusters:
        distances = evaluate_cluster(cluster)
        for dist in distances:
            sse += (distances[dist])**2
    return sse

def evaluate_cluster(cluster):
    """
    evaluate and print cluster info
    """
    points = list()
    points = cluster.get_data_points(points)
    # calculate centroid
    sum = 0
    for point in points:
        sum += point
    centroid = sum / len(points)

    # calculate distance of each point from centroid
    # store distances in dict mapping point to distance - points are stored as tuples, not Series
    distances = dict()
    sum_distances = 0
    max_dist = 0
    min_dist = -1
    for point in points:
        dist = DistanceCalculator.euclidean_distance(point, centroid)
        distances[tuple(point)] = dist
        sum_distances += dist
        if dist > max_dist:
            max_dist = dist
        if dist < min_dist or min_dist == -1:
            min_dist = dist

    # print cluster info
    avg_dist = sum_distances / len(points)
    print("Center:", cluster.print_data_point(centroid))
    print("Max Dist. to Center:", max_dist)
    print("Min Dist. to Center:", min_dist)
    print("Avg Dist. to Center:", avg_dist)
    print(len(points), "Points")
    for point in points:
        cluster.print_data_point(point)

    return distances


def cut_at_threshold(tree, threshold):
    """
    All nodes of the dendrogram with labels greater than threshold are removed from the dendrogram, 
    together with any adjacent edges. 
    The resulting forest represents the clusters found by a hierarchical clustering
    method that constructed the dendrogram, at the threshold
    """
    queue = list()
    queue.append(tree)
    clusters = list()
    while len(queue) > 0:
        node = queue[0]
        if node.height > threshold:
            for child in node.nodes:
                queue.append(child)
        else:
            clusters.append(node)
        queue.remove(node)
    return clusters

"""
1. On step 1, each point x ∈ D is assigned to its own cluster.
2. On each step, the algorithm computes the distance matrix for the current
list of clusters.
3. It then selects a pair of clusters with the shortest distance, and merges these
two clusters into one (constructing the apprpriate part of the dendrogram).
4. The algorithm stops when all points are merged into a single cluster.
"""
def hclusters(data):
    df = data[0]
    class_labels = data[1]
    # each point x ∈ D is assigned to its own cluster
    # make a cluster for every point - leaf node for each row in data frame
    clusters = list()
    #df.apply(lambda x: clusters.append(DendrogramNode("leaf", 0.0, None, x)), axis=1)
    for index, row in df.iterrows():
        clusters.append(DendrogramNode("leaf", 0.0, None, row))
    #for cluster in clusters:
    #    print("CLUSTER:", cluster.type, cluster.data)
    #print(clusters)
    # while len(cluster) > 1:
    while len(clusters) > 1:
        # compute distance matrix for current list of clusters
        distance_matrix = compute_distance_matrix(clusters)
        # select pair of clusters w the shortest distance
        shortest_dist = -1
        cluster_pair = None
        for pair in distance_matrix.keys():
            if shortest_dist < 0 or distance_matrix[pair] < shortest_dist:
                shortest_dist = distance_matrix[pair]
                cluster_pair = pair
        # merge pair of clusters into one
        if len(clusters) == 2:
            clusters.append(cluster_pair[0].combine_nodes(cluster_pair[1], "root", shortest_dist))
        else:
            clusters.append(cluster_pair[0].combine_nodes(cluster_pair[1], "node", shortest_dist))
        clusters.remove(cluster_pair[0])
        clusters.remove(cluster_pair[1])
        # construct part of dendrogram
    return clusters[0]

def compute_distance_matrix(clusters):
    """
    given a list of clusters, calculate the distance of each cluster from each of the other clusters
    will later use this ^ to then identify the two clusters have the shortest distance between them
    form of distance matrix: dictionary mapping (cluster1, cluster2) to the distance between the clusters
    """
    distance_matrix = dict()
    for i in range(len(clusters)):   # each cluster is dendrogram node
        cluster1 = clusters[i]
        #print("cluster1", cluster1.type, cluster1.data)
        for j in range(i + 1, len(clusters)):
            cluster2 = clusters[j]
            #print("cluster2", cluster2.type, cluster2.data)
            if cluster1 != cluster2 and (cluster1, cluster2) not in distance_matrix and \
                    (cluster2, cluster1) not in distance_matrix:
                dist = cluster_distance_single_link(cluster1, cluster2)
                distance_matrix[(cluster1, cluster2)] = dist
    return distance_matrix


def cluster_distance_single_link(cluster1, cluster2):
    """
    given two clusters, compute the distance between them using the single link method
    single link - distance between the clusters is the distance between the two closest points in the clusters
    """
    cluster1_points = list()
    cluster1_points = cluster1.get_data_points(cluster1_points)
    #print(cluster1_points)
    cluster2_points = list()
    cluster2_points = cluster2.get_data_points(cluster2_points)
    #print(cluster2_points)
    lowest_distance = -1
    for x1 in cluster1_points:
        if type(x1) == list:
            x1 = x1[0]
        for x2 in cluster2_points:
            if type(x2) == list:
                x2 = x2[0]
            dist = DistanceCalculator.euclidean_distance(x1, x2)
            if dist < lowest_distance or lowest_distance < 0:
                lowest_distance = dist
    return lowest_distance


if __name__ == "__main__":
    main()