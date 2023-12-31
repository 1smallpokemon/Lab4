import sys
import pandas as pd
import numpy as np
from preprocessing import load_data, preprocess_data
from DistanceCalculator import DistanceCalculator

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def find_epsilon_neighborhood(point_idx, data, epsilon):
    neighbors = []
    point = data.iloc[point_idx]
    for idx in range(len(data)):
        if euclidean_distance(point, data.iloc[idx]) < epsilon:
            neighbors.append(idx)
    return neighbors

def expand_cluster(data, labels, point_idx, neighbors, cluster_id, epsilon, min_points):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == 0:  # Point is unvisited
            labels[neighbor_idx] = cluster_id
            point_neighbors = find_epsilon_neighborhood(neighbor_idx, data, epsilon)
            if len(point_neighbors) >= min_points:
                neighbors = neighbors + point_neighbors
        i += 1

def dbscan(data, epsilon, min_points):
    labels = [0] * len(data)  # 0 indicates unvisited
    cluster_id = 0
    for point_idx in range(len(data)):
        if labels[point_idx] != 0:
            continue
        neighbors = find_epsilon_neighborhood(point_idx, data, epsilon)
        if len(neighbors) < min_points:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(data, labels, point_idx, neighbors, cluster_id, epsilon, min_points)
    return labels

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
