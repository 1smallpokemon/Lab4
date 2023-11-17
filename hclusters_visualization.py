import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from preprocessing import load_data, preprocess_data
from hclustering import hclusters, compute_clusters, DendrogramNode

def main():
    if len(sys.argv) < 2:
        print("Usage: python tuningScript.py <Filename>")
        sys.exit()

    # Load and preprocess data
    filename = sys.argv[1]
    data, class_labels = preprocess_data(load_data(filename))

    threshold = 0.1
    tree = hclusters((data, class_labels))
    clusters = compute_clusters(threshold, tree)

    # make clusters into a form that can be graphed
    # current form - clusters is a dendrogram node
    # want to have a dataframe or series where each attribute of each datapt is an att in df, and cluster is another att
    clusters_dict = {
        'x': list(),
        'y': list(),
        'cluster': list()
    }
    cluster_num = 0
    for cluster in clusters:
        data_points = list()
        data_points = cluster.get_data_points(data_points)
        for point in data_points:
            clusters_dict['x'].append(point[0])
            clusters_dict['y'].append(point[1])
            clusters_dict['cluster'].append(cluster_num)
        cluster_num += 1
    clusters_df = pd.DataFrame(clusters_dict)
    print(clusters_df)

    #clusters_df.plot.scatter(x='x', y='y', c='cluster')
    scatter = plt.scatter(clusters_df.x, clusters_df.y, c=clusters_df.cluster)
    plt.title(filename)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

if __name__ == "__main__":
    main()