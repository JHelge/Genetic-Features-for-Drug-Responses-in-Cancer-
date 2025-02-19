import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import fnmatch
def load_features_from_directories(base_dir):
    feature_sets = {}
    directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) 
                   if os.path.isdir(os.path.join(base_directory, d)) and fnmatch.fnmatch(d, 'Drug*_analysis')]
    for subdir in directories:
        dir_path = os.path.join(base_dir, subdir)
        features_path = os.path.join(dir_path, 'best/features_0.csv')
        if os.path.isfile(features_path):
            features_df = pd.read_csv(features_path, header=None)
            feature_set = set(features_df[0].values)
            feature_sets[subdir] = feature_set
    return feature_sets

def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def calculate_szymkiewicz_simpson(set1, set2):


    intersection = len(set1.intersection(set2))
    smaller_set_size = min(len(set1), len(set2))
    szymkiewicz_simpson = intersection / smaller_set_size
    return szymkiewicz_simpson

def create_similarity_matrix(feature_sets):
    directories = list(feature_sets.keys())
    n = len(directories)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                #similarity_matrix[i, j] = calculate_jaccard_similarity(feature_sets[directories[i]], feature_sets[directories[j]])
                similarity_matrix[i, j] = calculate_szymkiewicz_simpson(feature_sets[directories[i]], feature_sets[directories[j]])
            else:
                similarity_matrix[i, j] = 1.0  # Similarity with itself
    return directories, similarity_matrix

def perform_clustering(directories, similarity_matrix, threshold):
    # Convert similarity matrix to distance matrix
    distance_matrix = 1 - similarity_matrix
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=directories, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Directory")
    plt.ylabel("Distance")
    #plt.show()
    
    # Form flat clusters
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
    clustered_directories = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_directories:
            clustered_directories[cluster_id] = []
        clustered_directories[cluster_id].append(directories[i])
    
    return clustered_directories
    
def count_clusters(directories, similarity_matrix, thresholds):
    distance_matrix = 1 - similarity_matrix
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
    cluster_counts = []
    clusters_list = []
    
    for threshold in thresholds:
        clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
        num_clusters = len(set(clusters))
        cluster_counts.append(num_clusters)
        clusters_list.append(clusters)
    
    return cluster_counts, clusters_list, linkage_matrix
    
def save_cluster_features(base_dir, clustered_directories, feature_sets):
    for cluster_id, dirs in clustered_directories.items():
        cluster_folder = os.path.join(base_dir, f"Cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)

        # Initialize union and intersection sets
        union_features = set()
        intersection_features = None

        for dir in dirs:
            features = feature_sets[dir]
            if intersection_features is None:
                # Initialize intersection with the first set of features
                intersection_features = features.copy()
            union_features |= features
            intersection_features &= features

        # Save union of features to CSV
        pd.DataFrame(list(union_features)).to_csv(os.path.join(cluster_folder, 'union.csv'), index=False, header=False)

        # Save intersection of features to CSV
        if intersection_features is not None:
            pd.DataFrame(list(intersection_features)).to_csv(os.path.join(cluster_folder, 'intersection.csv'), index=False, header=False)

# Save the list of directories in the cluster
        pd.DataFrame(dirs, columns=['Directory']).to_csv(os.path.join(cluster_folder, 'directories.csv'), index=False, header=False)


def plot_clusters_vs_threshold(thresholds, cluster_counts):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, cluster_counts, marker='o')
    plt.title("Number of Clusters vs Threshold Distance")
    plt.xlabel("Threshold Distance")
    plt.ylabel("Number of Clusters")
    plt.axvline(x=1.0, color='r', linestyle='--', label=f'Threshold = 1.0')
    plt.grid(True)
    plt.show()

def plot_dendrogram(linkage_matrix, directories):
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=directories, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Directory")
    plt.axhline(y=1, color='r', linestyle='--')
    plt.ylabel("Distance")
    plt.show()

def print_clusters_at_minimum(directories, clusters_list, cluster_counts):
    min_clusters = min(cluster_counts)
    min_index = cluster_counts.index(min_clusters)
    best_clusters = clusters_list[min_index]
    
    clusters_dict = {}
    for dir_index, cluster_id in enumerate(best_clusters):
        dir_name = directories[dir_index]
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = []
        clusters_dict[cluster_id].append(dir_name)
    
    print(f"Number of Clusters at Minimum: {min_clusters}")
    for cluster_id, dirs in clusters_dict.items():
        print(f"Cluster {cluster_id}: {dirs}")

def plot_heatmap(similarity_matrix, directories):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=directories, yticklabels=directories, cmap="YlGnBu", annot=True)
    plt.title("Similarity Matrix Heatmap")
    plt.show()

def plot_heatmap(similarity_matrix, directories):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=directories, yticklabels=directories, cmap="YlGnBu", annot=True)
    plt.title("Similarity Matrix Heatmap")
    plt.show()

def plot_pca(similarity_matrix, clusters):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(similarity_matrix)
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", len(set(clusters)))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette=palette, legend="full")
    plt.title("PCA of Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.show()

def plot_tsne(similarity_matrix, clusters):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_data = tsne.fit_transform(similarity_matrix)
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", len(set(clusters)))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette=palette, legend="full")
    plt.title("t-SNE of Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster")
    plt.show()

def plot_umap(similarity_matrix, clusters):
    reducer = umap.UMAP()
    reduced_data = reducer.fit_transform(similarity_matrix)
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", len(set(clusters)))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette=palette, legend="full")
    plt.title("UMAP of Clusters")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend(title="Cluster")
    plt.show()
    
    

def main(base_dir, min_threshold=0.5, max_threshold=0.999, step=0.01):
    feature_sets = load_features_from_directories(base_dir)
    directories, similarity_matrix = create_similarity_matrix(feature_sets)
    
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    cluster_counts, clusters_list, linkage_matrix = count_clusters(directories, similarity_matrix, thresholds)
    
    print("Thresholds:", thresholds)
    print("Cluster counts:", cluster_counts)
    
    plot_clusters_vs_threshold(thresholds, cluster_counts)
    plot_dendrogram(linkage_matrix, directories)
    print_clusters_at_minimum(directories, clusters_list, cluster_counts)

    min_clusters = min(cluster_counts)
    min_index = cluster_counts.index(min_clusters)
    best_clusters = clusters_list[min_index]
    
    #plot_heatmap(similarity_matrix, directories)
    plot_pca(similarity_matrix, best_clusters)
    plot_tsne(similarity_matrix, best_clusters)
    plot_umap(similarity_matrix, best_clusters)

    # Perform clustering at a specific threshold
    clustered_directories = perform_clustering(directories, similarity_matrix, threshold=max_threshold)
    
    # Save the union of features for each cluster
    output_base_dir = "clustered_features"
    save_cluster_features(base_dir, clustered_directories, feature_sets)


if __name__ == "__main__":
    base_directory = "./"  # Replace with the base directory containing your subdirectories
    min_threshold_distance = 0.6  # Minimum threshold distance to consider
    max_threshold_distance = 0.999  # Maximum threshold distance to consider
    threshold_step = 0.001  # Step size for threshold distances
    main(base_directory, min_threshold_distance, max_threshold_distance, threshold_step)

#def main(base_dir, threshold):
#    feature_sets = load_features_from_directories(base_dir)
#    directories, similarity_matrix = create_similarity_matrix(feature_sets)
#    clustered_directories = perform_clustering(directories, similarity_matrix, threshold)
    
#    print("Clustered directories:")
#    for cluster_id, dirs in clustered_directories.items():
#        #print(len(dirs))
#        if len(dirs) < 2:
#            print(f"Cluster {cluster_id}: {dirs}")

#if __name__ == "__main__":
#    base_directory = "./"  # Replace with the base directory containing your subdirectories
#    threshold_distance = 0.957  # Adjust as necessary
#    main(base_directory, threshold_distance)

