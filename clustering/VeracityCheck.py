import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def visualize_embeddings(embedding_file_path, perplexity=30, n_iter=300, learning_rate=200):
    """
    Loads embeddings from a file, performs t-SNE for dimensionality reduction, and visualizes the results
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

    # Check if embeddings are empty
    if embeddings.size == 0:
        print("Error: Embeddings array is empty.")
        return

    # Check if there are enough samples for t-SNE
    if embeddings.shape[0] <= max(1, perplexity):
         print(f"Error: Number of samples ({embeddings.shape[0]}) must be greater than perplexity ({perplexity}) and 1 for t-SNE.")
         print("Consider reducing perplexity or providing more data.")
         return

    print("Performing t-SNE dimensionality reduction...")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate)
        embeddings_2d = tsne.fit_transform(embeddings)
        print("t-SNE completed.")
    except Exception as e:
        print(f"Error during t-SNE: {e}")
        # This can happen if parameters like perplexity are not suitable for the data size/structure
        print("Consider adjusting t-SNE parameters like perplexity, n_iter, or learning_rate.")
        return


    print("Creating visualization...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], cmap='viridis', s=10) # You can add 'c' for color coding if you have labels
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, label='(Optional) Color Label') # Add a label for the color bar if you use 'c'
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace with the actual path to your saved embeddings file
    your_embedding_file = "swin_embeddings/video---YU8YcWeUU_embeddings.npy"

    # --- Run Visualization ---
    visualize_embeddings(your_embedding_file)