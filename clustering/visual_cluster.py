import numpy as np
import hdbscan
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil
from PIL import Image
import logging
import warnings
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedHDBSCANClustering:
    """Optimized HDBSCAN clustering with adaptive parameters and cluster frame storage."""
    
    def __init__(self, embeddings_dir, frames_dir=None, output_dir="optimized_clustering_results"):
        """Initialize the optimized HDBSCAN clustering framework."""
        self.embeddings_dir = Path(embeddings_dir)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load all embedding files
        self.embedding_files = list(self.embeddings_dir.glob('*.npy'))
        logger.info(f"Found {len(self.embedding_files)} embedding files in {embeddings_dir}")
        
        # Best parameter set with adaptive min_cluster_size
        self.best_params = {
            'min_samples': 5, 
            'cluster_selection_epsilon': 0.0, 
            'cluster_selection_method': 'eom'
        }
    
    def adaptive_min_size(self, frames):
        """Calculate adaptive minimum cluster size based on video length."""
        return max(3, int(frames * 0.03))  # 3% of video length or at least 3 frames

    def load_frames(self, video_id):
        """Load frames for a video."""
        # (Code unchanged for loading frames)
        pass

    def visualize_clusters(self, embeddings, labels, video_id):
        """Visualize clusters using t-SNE."""
        try:
            logger.info(f"Visualizing clusters for {video_id}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300, learning_rate=200)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            plt.figure(figsize=(12, 10))
            adjusted_labels = labels + 1  # Shift labels to make noise (label -1) index 0
            cmap = plt.get_cmap('tab20', max(1, np.max(adjusted_labels) + 1))
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=adjusted_labels, cmap=cmap, s=10, alpha=0.7)
            
            plt.title(f't-SNE Visualization of Clusters for {video_id}')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Cluster ID (-1 = Noise)')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True)
            
            output_path = self.output_dir / f"{video_id}_tsne_visualization.png"
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Visualization saved to {output_path}")
        except Exception as e:
            logger.error(f"Error visualizing clusters for {video_id}: {e}")

    def run_clustering(self, sample_size=None):
        """Run clustering on selected videos."""
        logger.info("=== Running Optimized HDBSCAN Clustering ===")
        
        if sample_size is not None and sample_size < len(self.embedding_files):
            files_to_process = np.random.choice(self.embedding_files, size=sample_size, replace=False)
        else:
            files_to_process = self.embedding_files
        
        results = []
        
        for file_path in tqdm(files_to_process, desc="Processing videos"):
            video_id = file_path.stem.replace('_embeddings', '')
            result = self.process_video(video_id, file_path)
            if result:
                results.append(result)
        
        # Save results
        if results:
            df = pd.DataFrame(results)
            output_path = self.output_dir / "clustering_results.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
    
    def process_video(self, video_id, embedding_path):
        """Process a single video with optimized HDBSCAN parameters."""
        try:
            embeddings = np.load(embedding_path)
            min_cluster_size = self.adaptive_min_size(embeddings.shape[0])
            logger.info(f"Using adaptive min_cluster_size={min_cluster_size} for {video_id} ({embeddings.shape[0]} frames)")
            
            if embeddings.shape[0] < min_cluster_size:
                logger.warning(f"Skipping {video_id}: Not enough samples ({embeddings.shape[0]}) for clustering")
                return None
        except Exception as e:
            logger.error(f"Error loading {embedding_path}: {e}")
            return None
        
        try:
            params = self.best_params.copy()
            params['min_cluster_size'] = min_cluster_size
            
            clusterer = hdbscan.HDBSCAN(**params)
            clusterer.fit(embeddings)
            labels = clusterer.labels_
            
            video_output_dir = self.output_dir / video_id
            video_output_dir.mkdir(exist_ok=True, parents=True)
            
            self.store_cluster_frames(video_id, labels)
            self.visualize_clusters(embeddings, labels, video_id)  # Call visualization here
            
            result = {
                'video_id': video_id,
                'total_frames': embeddings.shape[0],
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'noise_points': np.sum(labels == -1),
                'noise_percentage': 100 * np.sum(labels == -1) / len(labels),
                'min_cluster_size': min_cluster_size
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error clustering {video_id}: {e}")
            return None
    
    def store_cluster_frames(self, video_id, labels):
        """Store frames from each cluster in separate folders."""
        # (Code unchanged for storing frames)
        pass

def main():
    parser = argparse.ArgumentParser(description="Optimized HDBSCAN Clustering")
    parser.add_argument("--embeddings_dir", type=str, default="swin_embeddings",
                        help="Directory containing embedding .npy files")
    parser.add_argument("--frames_dir", type=str, default="/Users/akshat/Developer/Vid_Attention/data_extraction/data/QuerYD/processed_frames",
                        help="Directory containing video frames (for storing cluster frames)")
    parser.add_argument("--output_dir", type=str, default="optimized_clustering_results",
                        help="Directory to save results")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of videos to sample (None = all)")
                        
    args = parser.parse_args()
    
    clustering = OptimizedHDBSCANClustering(
        embeddings_dir=args.embeddings_dir,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir
    )
    
    clustering.run_clustering(sample_size=args.sample_size)
    
if __name__ == "__main__":
    main()
