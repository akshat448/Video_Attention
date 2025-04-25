import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import hdbscan # Import HDBSCAN
import os

def visualize_embeddings_with_clustering(embedding_file_path, perplexity=30, n_iter=300, learning_rate=200, min_cluster_size=5):
    """
    Loads embeddings, performs HDBSCAN clustering, t-SNE for dimensionality reduction,
    and visualizes the t-SNE results colored by cluster labels.

    Args:
        embedding_file_path (str): Path to the .npy file containing embeddings.
        perplexity (int): The perplexity parameter for t-SNE.
        n_iter (int): The number of iterations for t-SNE.
        learning_rate (float): The learning rate for t-SNE.
        min_cluster_size (int): The minimum size of clusters for HDBSCAN.
    """
    if not os.path.exists(embedding_file_path):
        print(f"Error: Embedding file not found at {embedding_file_path}")
        return

    print(f"Loading embeddings from {embedding_file_path}...")
    try:
        embeddings = np.load(embedding_file_path)
        print(f"Embeddings loaded successfully with shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    # Check if embeddings are empty or too small
    if embeddings.size == 0:
        print("Error: Embeddings array is empty.")
        return
    if embeddings.shape[0] < 2:
        print(f"Error: Not enough samples ({embeddings.shape[0]}) for t-SNE or clustering. Need at least 2.")
        return
    if embeddings.shape[0] <= max(1, perplexity): # t-SNE requirement
         print(f"Error: Number of samples ({embeddings.shape[0]}) must be greater than perplexity ({perplexity}) and 1 for t-SNE.")
         print("Consider reducing perplexity or providing more data.")
         return

    # --- HDBSCAN Clustering ---
    print(f"Performing HDBSCAN clustering with min_cluster_size={min_cluster_size}...")
    try:
        # Apply HDBSCAN on the original high-dimensional embeddings
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(embeddings)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"HDBSCAN completed. Found {n_clusters} clusters (including noise points labeled -1).")
        unique_labels = sorted(list(set(cluster_labels)))
        print(f"Unique cluster labels found: {unique_labels}")

    except Exception as e:
        print(f"Error during HDBSCAN clustering: {e}")
        print("Clustering failed. Proceeding with t-SNE visualization without cluster coloring.")
        cluster_labels = None # Set labels to None if clustering fails

    # --- t-SNE Dimensionality Reduction ---
    print("Performing t-SNE dimensionality reduction...")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate)
        embeddings_2d = tsne.fit_transform(embeddings)
        print("t-SNE completed.")
    except Exception as e:
        print(f"Error during t-SNE: {e}")
        # This can happen if parameters like perplexity are not suitable for the data size/structure
        print("Consider adjusting t-SNE parameters like perplexity, n_iter, or learning_rate.")
        return # Cannot visualize without successful t-SNE

    # --- Visualization ---
    print("Creating visualization...")
    plt.figure(figsize=(12, 10)) # Slightly larger figure

    if cluster_labels is not None:
        # Plot points colored by HDBSCAN cluster label
        # HDBSCAN labels noise as -1. We can shift labels by +1 for coloring
        # so noise is index 0 in the colormap.
        adjusted_labels = cluster_labels + 1

        # Choose a colormap suitable for discrete categories
        # 'tab20' is a good choice for up to 20 distinct categories.
        # Adjust colormap size based on the number of unique adjusted labels
        max_label = np.max(adjusted_labels) if adjusted_labels.size > 0 else 0
        # Ensure colormap has at least 1 color if only noise or one cluster exists
        cmap = plt.get_cmap('tab20', max(1, max_label + 1))

        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=adjusted_labels, cmap=cmap, s=10, alpha=0.7) # Add some transparency

        plt.title(f't-SNE Visualization of Embeddings with HDBSCAN Clusters (min_cluster_size={min_cluster_size})')

        # Create a colorbar with correct ticks and labels
        unique_adjusted_labels = sorted(list(set(adjusted_labels)))
        # Map adjusted labels back to original HDBSCAN labels for display (-1, 0, 1...)
        original_labels_for_ticks = [label - 1 for label in unique_adjusted_labels]

        cbar = plt.colorbar(scatter, ticks=unique_adjusted_labels)
        cbar.ax.set_yticklabels([str(label) for label in original_labels_for_ticks])
        cbar.set_label('HDBSCAN Cluster ID (-1 = Noise)')

    else:
        # Plot without coloring if clustering was skipped or failed
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], cmap='viridis', s=10) # Default coloring
        plt.title('t-SNE Visualization of Embeddings (Clustering Skipped/Failed)')
        plt.colorbar(scatter, label='Data Point Index/Density (No Clusters)')


    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace with the actual path to your saved embeddings file
    your_embedding_file = "swin_embeddings/video---YU8YcWeUU_embeddings.npy"

    # --- HDBSCAN Configuration ---
    # Adjust this parameter based on your data size and expected cluster density
    # A smaller value will find smaller clusters, a larger value larger clusters.
    hdbscan_min_cluster_size = 5

    # --- t-SNE Configuration ---
    tsne_perplexity = 30
    tsne_n_iter = 300
    tsne_learning_rate = 200


    # --- Run Visualization ---
    visualize_embeddings_with_clustering(
        your_embedding_file,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
        learning_rate=tsne_learning_rate,
        min_cluster_size=hdbscan_min_cluster_size # Pass the HDBSCAN parameter
    )