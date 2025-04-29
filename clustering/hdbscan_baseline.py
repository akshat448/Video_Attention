import numpy as np
import hdbscan
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import warnings
import itertools
import time
import h5py
from scipy.ndimage import median_filter
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from hdbscan.validity import validity_index  # Import for DBCV metric
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # Add t-SNE for visualization
import os

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HDBSCANGridSearch:
    """Grid search for optimal HDBSCAN parameters on video embeddings."""
    
    def __init__(self, embeddings_dir, frames_dir=None, output_dir="hdbscan_grid_search_results"):
        """
        Initialize grid search framework.
        
        Args:
            embeddings_dir: Directory containing embedding .npy files
            frames_dir: Directory containing video frames (for visualization)
            output_dir: Directory to save results
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a subdirectory for visualizations
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Load all embedding files
        self.embedding_files = list(self.embeddings_dir.glob('*.npy'))
        logger.info(f"Found {len(self.embedding_files)} embedding files in {embeddings_dir}")
        
        # Parameter grid - simplified to focus on important parameters
        self.param_grid = {
            'min_cluster_size': [3, 5, 10, 12],
            'min_samples': [3, 5],
            'cluster_selection_method': ['eom']  # Keep this fixed
        }
        
        # Store all results
        self.all_results = []
        
        # Store raw clustering results for best visualizations
        self.clustering_results = {}
        
        # Store the best overall parameter combination
        self.best_global_params = None
        
        # Keep track of processed videos
        self.processed_videos = set()
    
    def temporal_coherence(self, labels):
        """Calculate temporal coherence (percentage of frame transitions)."""
        if len(labels) <= 1:
            return 0.0
            
        transitions = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                transitions += 1
        
        transition_percentage = 100 * transitions / (len(labels) - 1)
        return transition_percentage
    
    def temporal_smoothing(self, labels, window_size=3):
        """Apply median filter to reduce temporal noise."""
        return median_filter(labels, size=window_size, mode='nearest')
    
    def calculate_metrics(self, embeddings, labels):
        """Calculate clustering quality metrics."""
        metrics = {}

        # Basic statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = np.sum(labels == -1)
        noise_ratio = 100 * noise_points / len(labels) if len(labels) > 0 else 100.0

        metrics['n_clusters'] = n_clusters
        metrics['noise_points'] = noise_points
        metrics['noise_ratio'] = noise_ratio

        # Temporal coherence
        metrics['temporal_coherence'] = self.temporal_coherence(labels)

        # Skip other metrics if all points are noise or only one cluster
        if n_clusters <= 1:
            metrics['silhouette'] = None
            metrics['dbcv'] = None
            return metrics

        # Get valid points (non-noise)
        valid_indices = labels != -1
        valid_embeddings = embeddings[valid_indices]
        valid_labels = labels[valid_indices]

        # Skip if too few valid points
        if len(valid_labels) < 3:
            metrics['silhouette'] = None
            metrics['dbcv'] = None
            return metrics

        # Calculate silhouette score
        try:
            metrics['silhouette'] = silhouette_score(valid_embeddings, valid_labels)
        except Exception as e:
            logger.error(f"Error calculating silhouette score: {e}")
            metrics['silhouette'] = None

        # Calculate DBCV score - most important for density-based clustering
        try:
            # Ensure embeddings are float64 for numerical stability
            valid_embeddings_64 = valid_embeddings.astype(np.float64)

            # DBCV calculation can be unstable, so add more error checking
            # Only calculate if we have enough samples per cluster
            min_points_per_cluster = 3
            cluster_sizes = np.bincount(valid_labels)
            if np.all(cluster_sizes >= min_points_per_cluster):
                dbcv_score = validity_index(valid_embeddings_64, valid_labels)
                if np.isnan(dbcv_score):
                    logger.warning("DBCV score is nan, skipping this combination")
                    metrics['dbcv'] = None
                else:
                    metrics['dbcv'] = dbcv_score
                    logger.info(f"DBCV score: {metrics['dbcv']:.4f}")
            else:
                logger.warning("Skipping DBCV: some clusters have too few points")
                metrics['dbcv'] = None

        except Exception as e:
            logger.error(f"Error calculating DBCV score: {e}")
            metrics['dbcv'] = None

        return metrics
    
    def visualize_clusters(self, embeddings, labels, video_id, params, best=False):
        """
        Visualize clusters using t-SNE dimensionality reduction.
        
        Args:
            embeddings: Embedding vectors
            labels: Cluster labels from HDBSCAN
            video_id: ID of the video
            params: HDBSCAN parameters used for clustering
            best: Whether this is the best parameter set (for filename)
        """
        try:
            logger.info(f"Creating t-SNE visualization for {video_id}...")
            
            # Dynamically adjust perplexity based on number of samples
            perplexity = min(30, max(5, embeddings.shape[0] // 10))
            
            # Perform t-SNE dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42, 
                        perplexity=perplexity, 
                        n_iter=300, 
                        learning_rate=200)
            
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Adjust labels for visualization (shift by +1 so noise is 0)
            adjusted_labels = labels + 1
            
            # Choose a colormap suitable for discrete categories
            max_label = np.max(adjusted_labels) if adjusted_labels.size > 0 else 0
            cmap = plt.get_cmap('tab20', max(1, max_label + 1))
            
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                c=adjusted_labels, cmap=cmap, s=10, alpha=0.7)
            
            # Add title with key parameters
            min_cluster_size = params['min_cluster_size']
            min_samples = params['min_samples']
            plt.title(f't-SNE Visualization for {video_id}\n'
                      f'HDBSCAN: min_cluster={min_cluster_size}, min_samples={min_samples} '
                      f'{"(Best Global Parameters)" if best else ""}')
            
            # Create a colorbar with correct labels
            unique_adjusted_labels = sorted(list(set(adjusted_labels)))
            original_labels = [label - 1 for label in unique_adjusted_labels]
            
            cbar = plt.colorbar(scatter, ticks=unique_adjusted_labels)
            cbar.ax.set_yticklabels([str(label) for label in original_labels])
            cbar.set_label('HDBSCAN Cluster ID (-1 = Noise)')
            
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True)
            
            # Save the visualization
            param_str = f"mcs{min_cluster_size}_ms{min_samples}"
            if best:
                param_str = f"BEST_GLOBAL_{param_str}"
            viz_filename = f"{video_id}_{param_str}_tsne.png"
            viz_path = self.viz_dir / viz_filename
            plt.savefig(viz_path, dpi=150)
            plt.close()
            
            logger.info(f"Saved t-SNE visualization to {viz_path}")
            
        except Exception as e:
            logger.error(f"Error creating t-SNE visualization for {video_id}: {e}")

    def run_parameter_combination(self, video_id, embeddings, params):
        """Run clustering with a specific parameter combination."""
        # Skip short videos
        if embeddings.shape[0] < params['min_cluster_size']:
            logger.warning(f"Skipping {video_id}: Not enough samples ({embeddings.shape[0]}) for clustering with min_cluster_size={params['min_cluster_size']}")
            return None

        try:
            start_time = time.time()

            # Run HDBSCAN
            clusterer = hdbscan.HDBSCAN(**params)
            clusterer.fit(embeddings)
            labels = clusterer.labels_

            # Apply temporal smoothing
            labels = self.temporal_smoothing(labels)

            # Calculate metrics
            metrics = self.calculate_metrics(embeddings, labels)

            # Store raw clustering results for later visualization
            key = (video_id, params['min_cluster_size'], params['min_samples'], 
                  params.get('cluster_selection_epsilon', 0.0))
            
            # Add video to processed set
            self.processed_videos.add(video_id)
            
            # Only store if we have valid DBCV score and at least one non-noise cluster
            if metrics.get('dbcv') is not None and metrics.get('n_clusters', 0) > 0:
                self.clustering_results[key] = {
                    'embeddings': embeddings,
                    'labels': labels,
                    'metrics': metrics,
                    'params': params.copy()
                }

            # Calculate processing time
            runtime = time.time() - start_time

            # Combine parameters and metrics
            result = {
                'video_id': video_id,
                'total_frames': embeddings.shape[0],
                'runtime': runtime,
                **params,
                **metrics
            }

            return result

        except Exception as e:
            logger.error(f"Error running clustering on {video_id} with params {params}: {e}")
            return None

    def run_grid_search(self, sample_size=None):
        """Run grid search on all videos and parameter combinations."""
        logger.info("=== Running HDBSCAN Grid Search ===")
        
        # Sample videos if needed
        if sample_size is not None and sample_size < len(self.embedding_files):
            # Set a random seed for reproducibility
            np.random.seed(42)
            files_to_process = np.random.choice(self.embedding_files, size=sample_size, replace=False)
        else:
            files_to_process = self.embedding_files
        
        # Generate all parameter combinations
        param_combinations = list(self._generate_param_combinations())
        logger.info(f"Testing {len(param_combinations)} parameter combinations on {len(files_to_process)} videos")
        
        # Process each video
        for file_path in tqdm(files_to_process, desc="Processing videos"):
            video_id = file_path.stem.replace('_embeddings', '')
            
            try:
                # Load embeddings
                embeddings = np.load(file_path)
                logger.info(f"Processing {video_id} with {embeddings.shape[0]} frames")
                
                # Test each parameter combination
                for params in tqdm(param_combinations, desc=f"Testing parameters for {video_id}", leave=False):
                    result = self.run_parameter_combination(video_id, embeddings, params)
                    if result:
                        self.all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
        
        # Save and analyze results
        self._save_and_analyze_results()
        
        # Apply the best global parameters to all videos
        if self.best_global_params:
            logger.info(f"Applying best global parameters to all videos: {self.best_global_params}")
            self.apply_best_params_to_all_videos(self.best_global_params)
        else:
            logger.error("No best global parameters found. Cannot generate visualizations and clusters.")
    
    def _generate_param_combinations(self):
        """Generate all combinations of parameters in the grid."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
    
    def _save_and_analyze_results(self):
        """Save results and identify best parameter combinations."""
        if not self.all_results:
            logger.warning("No valid results to analyze")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)
        
        # Save raw results
        output_path = self.output_dir / "grid_search_all_results.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"All results saved to {output_path}")
        
        # Average metrics for each parameter combination
        param_columns = list(self.param_grid.keys())
        metrics_of_interest = ['n_clusters', 'noise_ratio', 'temporal_coherence', 'dbcv']
        
        # Filter out rows with missing dbcv values
        df_valid = df.dropna(subset=['dbcv'])
        
        if len(df_valid) == 0:
            logger.warning("No valid DBCV scores to analyze")
            return
        
        # Group by parameter combination and calculate average metrics
        grouped_results = df_valid.groupby(param_columns)[metrics_of_interest].mean().reset_index()
        
        # Calculate a combined score: higher DBCV is better, lower temporal_coherence is better
        # Normalize to 0-1 range
        max_dbcv = grouped_results['dbcv'].max()
        min_dbcv = grouped_results['dbcv'].min()
        dbcv_range = max_dbcv - min_dbcv if max_dbcv > min_dbcv else 1.0
        
        max_tc = grouped_results['temporal_coherence'].max()
        min_tc = grouped_results['temporal_coherence'].min()
        tc_range = max_tc - min_tc if max_tc > min_tc else 1.0
        
        # Normalize: higher is better for both components
        grouped_results['dbcv_norm'] = (grouped_results['dbcv'] - min_dbcv) / dbcv_range
        grouped_results['tc_norm'] = 1 - (grouped_results['temporal_coherence'] - min_tc) / tc_range
        
        # Combined score with weights (DBCV more important)
        # Modified: Add small random noise to break ties between identical parameter sets
        np.random.seed(42)  # For reproducibility
        grouped_results['combined_score'] = 0.7 * grouped_results['dbcv_norm'] + 0.3 * grouped_results['tc_norm']
        grouped_results['combined_score'] += np.random.uniform(0, 0.0001, size=len(grouped_results))
        
        # Filter for noise_ratio < 25% (more lenient threshold)
        low_noise_results = grouped_results[grouped_results['noise_ratio'] < 25]
        
        if len(low_noise_results) == 0:
            logger.warning("No parameter combinations with noise_ratio < 25%, using all results")
            low_noise_results = grouped_results
        
        # Sort by combined score (descending)
        best_params = low_noise_results.sort_values('combined_score', ascending=False)
        
        # Save best parameter combinations
        best_params_path = self.output_dir / "best_parameter_combinations.csv"
        best_params.to_csv(best_params_path, index=False)
        logger.info(f"Best parameter combinations saved to {best_params_path}")
        
        # Print top 3 parameter combinations
        logger.info("=== Top 3 Parameter Combinations ===")
        top_3 = best_params.head(3)
        for i, row in top_3.iterrows():
            params_str = ", ".join([f"{param}={row[param]}" for param in param_columns])
            logger.info(f"Rank {i+1}: {params_str}")
            logger.info(f"  DBCV: {row['dbcv']:.4f}, Temporal Coherence: {row['temporal_coherence']:.2f}%, Noise Ratio: {row['noise_ratio']:.2f}%")
            logger.info(f"  Combined Score: {row['combined_score']:.4f}")
        
        # Also save parameter stats by video
        by_video = df_valid.groupby(['video_id', *param_columns])[metrics_of_interest].mean().reset_index()
        by_video_path = self.output_dir / "parameter_stats_by_video.csv"
        by_video.to_csv(by_video_path, index=False)
        logger.info(f"Parameter statistics by video saved to {by_video_path}")
        
        # Store the best overall parameter combination
        if len(best_params) > 0:
            best_row = best_params.iloc[0]
            self.best_global_params = {
                'min_cluster_size': best_row['min_cluster_size'],
                'min_samples': best_row['min_samples'],
                'cluster_selection_method': best_row['cluster_selection_method']
            }
            
            # Print the best global parameter combination
            logger.info("\n=== BEST OVERALL PARAMETER COMBINATION ===")
            logger.info(f"min_cluster_size = {self.best_global_params['min_cluster_size']}")
            logger.info(f"min_samples = {self.best_global_params['min_samples']}")
            logger.info(f"cluster_selection_method = {self.best_global_params['cluster_selection_method']}")
            logger.info(f"DBCV: {best_row['dbcv']:.4f}, Temporal Coherence: {best_row['temporal_coherence']:.2f}%, Noise Ratio: {best_row['noise_ratio']:.2f}%")
            logger.info(f"Combined Score: {best_row['combined_score']:.4f}")
            logger.info("This combination will be used for all visualizations and cluster generation.")
            logger.info("=============================================")
        else:
            logger.warning("No best parameters found")
    
    def apply_best_params_to_all_videos(self, best_params):
        """
        Apply the best global parameter combination to all videos.
        Generate visualizations and store cluster frames for each video.
        
        Args:
            best_params: The best parameter combination determined by grid search
        """
        logger.info("=== Applying Best Global Parameters to All Videos ===")
        
        # Process each video with the best parameters
        for file_path in tqdm(self.embedding_files, desc="Processing videos with best parameters"):
            video_id = file_path.stem.replace('_embeddings', '')
            
            try:
                # Load embeddings
                embeddings = np.load(file_path)
                logger.info(f"Processing {video_id} with best parameters, frames: {embeddings.shape[0]}")
                
                # Run clustering with best parameters
                clusterer = hdbscan.HDBSCAN(**best_params)
                clusterer.fit(embeddings)
                labels = self.temporal_smoothing(clusterer.labels_)
                
                # Calculate metrics for reporting
                metrics = self.calculate_metrics(embeddings, labels)
                
                # Create t-SNE visualization
                self.visualize_clusters(embeddings, labels, video_id, best_params, best=True)
                
                # Store cluster frames
                if self.frames_dir:
                    # Use a consistent output naming convention
                    self.store_cluster_frames(video_id, labels, best=True)
                
                # Log metrics for this video
                if metrics.get('dbcv') is not None:
                    logger.info(f"Video {video_id} with best parameters: DBCV={metrics['dbcv']:.4f}, " +
                              f"Clusters={metrics['n_clusters']}, Noise={metrics['noise_ratio']:.2f}%, " +
                              f"Temporal Coherence={metrics['temporal_coherence']:.2f}%")
                else:
                    logger.info(f"Video {video_id} with best parameters: No valid DBCV, " +
                              f"Clusters={metrics['n_clusters']}, Noise={metrics['noise_ratio']:.2f}%, " +
                              f"Temporal Coherence={metrics['temporal_coherence']:.2f}%")
                
            except Exception as e:
                logger.error(f"Error processing {video_id} with best parameters: {e}")
        
        logger.info("=== Completed Processing All Videos with Best Parameters ===")
    
    def load_frames(self, video_id):
        """
        Load frames for a given video ID from either NPY or H5 file.
        
        Args:
            video_id: ID of the video to load frames for
            
        Returns:
            frames: Numpy array of frames or None if loading fails
        """
        if self.frames_dir is None:
            logger.warning(f"No frames directory provided for {video_id}")
            return None
            
        # Try different possible locations and formats
        video_dir = self.frames_dir / video_id
        npy_path = video_dir / f"{video_id}_frames.npy"
        h5_path = video_dir / f"{video_id}_frames.h5"
        
        # Add more potential frame locations based on the QuerYD dataset structure
        frames_dir_locations = [
            self.frames_dir / video_id,  # /frames_dir/video_id/
            self.frames_dir,             # /frames_dir/
            self.frames_dir / "frames",  # /frames_dir/frames/
            self.frames_dir.parent / "frames" / video_id  # /parent_of_frames_dir/frames/video_id/
        ]
        
        # Log what we're checking
        logger.info(f"Looking for frames for {video_id} in multiple locations...")
        
        # Try multiple directory structures
        for frames_dir in frames_dir_locations:
            # Skip if directory doesn't exist
            if not frames_dir.exists():
                continue
                
            logger.info(f"Checking directory: {frames_dir}")
            
            # Try NPY files
            potential_npy_files = [
                frames_dir / f"{video_id}_frames.npy",
                frames_dir / f"{video_id}.npy",
                frames_dir / "frames.npy"
            ]
            
            for npy_file in potential_npy_files:
                if npy_file.exists():
                    try:
                        logger.info(f"Loading frames from NPY: {npy_file}")
                        frames = np.load(npy_file)
                        logger.info(f"Loaded frames with shape: {frames.shape}")
                        return frames
                    except Exception as e:
                        logger.error(f"Error loading NPY file {npy_file}: {e}")
            
            # Try H5 files
            potential_h5_files = [
                frames_dir / f"{video_id}_frames.h5",
                frames_dir / f"{video_id}.h5",
                frames_dir / "frames.h5"
            ]
            
            for h5_file in potential_h5_files:
                if h5_file.exists():
                    try:
                        logger.info(f"Loading frames from H5: {h5_file}")
                        with h5py.File(h5_file, 'r') as f:
                            # Check different possible dataset names
                            for dataset_name in ['frames', 'images', 'data', video_id]:
                                if dataset_name in f:
                                    frames = f[dataset_name][()]
                                    logger.info(f"Loaded frames with shape: {frames.shape}")
                                    return frames
                            logger.error(f"No valid dataset found in H5 file: {h5_file}")
                    except Exception as e:
                        logger.error(f"Error loading H5 file {h5_file}: {e}")
            
            # Try loading individual image frames
            frames_folder = frames_dir
            if frames_folder.is_dir():
                image_files = sorted([f for f in frames_folder.glob("*.jpg") or frames_folder.glob("*.png")])
                if image_files:
                    try:
                        logger.info(f"Loading {len(image_files)} individual frames from {frames_folder}")
                        first_img = Image.open(image_files[0])
                        img_shape = np.array(first_img).shape
                        frames = np.empty((len(image_files), *img_shape), dtype=np.uint8)
                        
                        for i, img_path in enumerate(tqdm(image_files, desc="Loading frames")):
                            frames[i] = np.array(Image.open(img_path))
                        
                        logger.info(f"Loaded {len(image_files)} frames with shape {frames.shape}")
                        return frames
                    except Exception as e:
                        logger.error(f"Error loading image frames from {frames_folder}: {e}")
        
        # Try directly in processed_frames directory (common QuerYD structure)
        try:
            direct_npy_path = self.frames_dir / f"{video_id}.npy"
            if direct_npy_path.exists():
                logger.info(f"Loading frames directly from: {direct_npy_path}")
                frames = np.load(direct_npy_path)
                logger.info(f"Loaded frames with shape: {frames.shape}")
                return frames
        except Exception as e:
            logger.error(f"Error loading direct NPY file: {e}")
        
        # If we get here, log an error and return None
        logger.error(f"Could not find frames for {video_id} in any expected location")
        return None
    
    def store_cluster_frames(self, video_id, labels, best=False):
        """
        Store frames from each cluster in separate folders.
        
        Args:
            video_id: ID of the video
            labels: HDBSCAN cluster labels
            best: Whether this is using the best parameters
        """
        try:
            frames = self.load_frames(video_id)
            
            if frames is None:
                logger.warning(f"Skipping frame storage for {video_id}: Could not load frames")
                return
                
            if len(frames) != len(labels):
                logger.error(f"Error: Number of frames ({len(frames)}) doesn't match labels ({len(labels)})")
                return
                
            logger.info(f"Organizing frames for {video_id} into cluster folders...")
            
            video_output_dir = self.output_dir / video_id
            if best:
                video_output_dir = self.output_dir / f"{video_id}_best"
            
            video_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Remove existing cluster folders
            for cluster_dir in video_output_dir.glob("cluster_*"):
                if cluster_dir.is_dir():
                    shutil.rmtree(cluster_dir)
            
            # Create folders for each cluster
            unique_labels = sorted(set(labels))
            for label in unique_labels:
                cluster_folder_name = f"cluster_{label}" if label != -1 else "cluster_noise"
                (video_output_dir / cluster_folder_name).mkdir(exist_ok=True)
            
            # Save frames into corresponding cluster folders
            for i, (frame, label) in enumerate(tqdm(zip(frames, labels), 
                                               desc=f"Saving frames for {video_id}", 
                                               total=len(frames))):
                cluster_folder_name = f"cluster_{label}" if label != -1 else "cluster_noise"
                frame_filename = f"frame_{i:05d}.png"
                frame_output_path = video_output_dir / cluster_folder_name / frame_filename
                
                try:
                    if frame.dtype != np.uint8:
                        if np.max(frame) <= 1.0 and np.min(frame) >= 0.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                    
                    if len(frame.shape) == 2:  # Grayscale image
                        frame = np.stack([frame] * 3, axis=2)  # Convert to RGB
                    
                    img = Image.fromarray(frame)
                    img.save(str(frame_output_path))
                    
                except Exception as e:
                    logger.error(f"Error saving frame {i} for video {video_id}: {e}")
            
            logger.info(f"Saved all frames by cluster for {video_id}")
            
        except Exception as e:
            logger.error(f"Error storing cluster frames for {video_id}: {e}")

    def create_summary_plots(self):
        """Create summary plots of the grid search results."""
        try:
            # Load results
            results_path = self.output_dir / "grid_search_all_results.csv"
            if not results_path.exists():
                logger.error("Cannot create summary plots: results file not found")
                return
            
            df = pd.read_csv(results_path)
            df_valid = df.dropna(subset=['dbcv'])
            
            if len(df_valid) == 0:
                logger.error("Cannot create summary plots: no valid DBCV scores")
                return
            
            logger.info("Creating summary plots...")
            
            # Create plots directory
            plots_dir = self.output_dir / "summary_plots"
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            # 1. DBCV vs noise ratio scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(df_valid['noise_ratio'], df_valid['dbcv'], alpha=0.5)
            plt.xlabel('Noise Ratio (%)')
            plt.ylabel('DBCV Score')
            plt.title('DBCV Score vs Noise Ratio')
            plt.grid(True)
            plt.savefig(plots_dir / "dbcv_vs_noise.png", dpi=150)
            plt.close()
            
            # 2. DBCV vs temporal coherence scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(df_valid['temporal_coherence'], df_valid['dbcv'], alpha=0.5)
            plt.xlabel('Temporal Coherence (% transitions)')
            plt.ylabel('DBCV Score')
            plt.title('DBCV Score vs Temporal Coherence')
            plt.grid(True)
            plt.savefig(plots_dir / "dbcv_vs_temporal.png", dpi=150)
            plt.close()
            
            # 3. Result distribution for the best parameter combination
            if self.best_global_params:
                best_params_mask = True
                for param, value in self.best_global_params.items():
                    best_params_mask = best_params_mask & (df_valid[param] == value)
                
                best_params_results = df_valid[best_params_mask]
                
                if len(best_params_results) > 0:
                    # DBCV scores distribution for best params
                    plt.figure(figsize=(12, 6))
                    plt.hist(best_params_results['dbcv'], bins=20, alpha=0.7)
                    plt.axvline(best_params_results['dbcv'].mean(), color='red', linestyle='dashed', 
                                linewidth=2, label=f'Mean: {best_params_results["dbcv"].mean():.4f}')
                    plt.xlabel('DBCV Score')
                    plt.ylabel('Count')
                    plt.title(f'DBCV Scores Distribution for Best Parameters\n'
                             f'min_cluster_size={self.best_global_params["min_cluster_size"]}, '
                             f'min_samples={self.best_global_params["min_samples"]}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(plots_dir / "best_params_dbcv_distribution.png", dpi=150)
                    plt.close()
                    
                    # Number of clusters distribution for best params
                    plt.figure(figsize=(12, 6))
                    plt.hist(best_params_results['n_clusters'], bins=range(0, int(best_params_results['n_clusters'].max()) + 2), 
                             alpha=0.7, align='left')
                    plt.axvline(best_params_results['n_clusters'].mean(), color='red', linestyle='dashed', 
                                linewidth=2, label=f'Mean: {best_params_results["n_clusters"].mean():.2f}')
                    plt.xlabel('Number of Clusters')
                    plt.ylabel('Count')
                    plt.title(f'Number of Clusters Distribution for Best Parameters\n'
                             f'min_cluster_size={self.best_global_params["min_cluster_size"]}, '
                             f'min_samples={self.best_global_params["min_samples"]}')
                    plt.xticks(range(0, int(best_params_results['n_clusters'].max()) + 1))
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(plots_dir / "best_params_clusters_distribution.png", dpi=150)
                    plt.close()
            
            logger.info(f"Summary plots saved to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error creating summary plots: {e}")

def main():
    parser = argparse.ArgumentParser(description="HDBSCAN Grid Search for Video Clustering")
    parser.add_argument("--embeddings_dir", type=str, default="swin_embeddings",
                        help="Directory containing embedding .npy files")
    parser.add_argument("--frames_dir", type=str, default="/Users/akshat/Developer/Vid_Attention/data_extraction/data/QuerYD/processed_frames",
                        help="Directory containing video frames (for storing cluster frames)")
    parser.add_argument("--output_dir", type=str, default="hdbscan_grid_search_results",
                        help="Directory to save results")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of videos to sample (None = all)")
    args = parser.parse_args()
    
    # Initialize grid search
    grid_search = HDBSCANGridSearch(
        embeddings_dir=args.embeddings_dir,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir
    )
    
    # Run grid search
    grid_search.run_grid_search(sample_size=args.sample_size)
    
    # Create summary plots
    grid_search.create_summary_plots()
    
if __name__ == "__main__":
    main()