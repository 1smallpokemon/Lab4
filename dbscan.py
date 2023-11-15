import sys
import pandas as pd
import numpy as np
from preprocessing import load_data, preprocess_data
from DistanceCalculator import DistanceCalculator

def find_epsilon_neighborhood(point, data, epsilon):
    neighbors = []
    for idx, other_point in data.iterrows():
        if DistanceCalculator.euclidean_distance(point, other_point) < epsilon:
            neighbors.append(idx)
    return neighbors

def dbscan(data, epsilon, min_points):
    labels = [0]*data.shape[0]  # 0 indicates unvisited
    cluster_id = 0

    for idx, point in data.iterrows():
        if labels[idx] != 0:  # Skip if already visited
            continue

        # Find neighbors
        neighbors = find_epsilon_neighborhood(point, data, epsilon)

        if len(neighbors) < min_points:
            labels[idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            labels[idx] = cluster_id
            grow_cluster(data, labels, cluster_id, neighbors, epsilon, min_points)

    return labels

def grow_cluster(data, labels, cluster_id, neighbors, epsilon, min_points):
    i = 0
    while i < len(neighbors):
        point_idx = neighbors[i]
        if labels[point_idx] == -1:
            labels[point_idx] = cluster_id  # Change noise to border point
        elif labels[point_idx] == 0:
            labels[point_idx] = cluster_id  # Add new point to cluster
            point_neighbors = find_epsilon_neighborhood(data.iloc[point_idx], data, epsilon)
            if len(point_neighbors) >= min_points:
                neighbors += point_neighbors  # Add new neighbors to the list
        i += 1

def main():
    argc = len(sys.argv)
    if argc < 4:
        print("Usage: python dbscan.py <Filename> <epsilon> <NumPoints>")
        sys.exit(1)

    try:
        file_path = sys.argv[1]
        epsilon = float(sys.argv[2])
        min_points = int(sys.argv[3])

        uncleaned_data = load_data(file_path)
        data, class_labels = preprocess_data(uncleaned_data)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    labels = dbscan(data, epsilon, min_points)

    # Process labels into clusters and noise
    clusters = {}
    noise = []
    for idx, label in enumerate(labels):
        if label == -1:
            noise.append(idx)
        else:
            clusters.setdefault(label, []).append(idx)

    # Output results
    for cluster_id, points in clusters.items():
        print(f"Cluster {cluster_id}: {len(points)} points")
        # Further output details for each cluster

    print(f"Noise: {len(noise)} points")
    # Further output details for noise

if __name__ == "__main__":
    main()
