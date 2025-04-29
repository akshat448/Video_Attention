import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hdbscan
import argparse
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.ndimage import median_filter
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm
import logging
import h5py
from PIL import Image
import os
import torch
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set default style for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

class ClusteringVisualizer:
    """Create visualizations to understand and explain HDBSCAN clustering results."""
    
    def __init__(self, embeddings_dir, output_dir, frames_dir=None):
        """
        Initialize the visualizer.
        
        Args:
            embeddings_dir: Directory containing embedding .npy files
            output_dir: Directory to save visualization results
            frames_dir: Optional directory containing video frames
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set the best parameters based on our grid search
        self.best_params = {
            'min_cluster_size': 5,
            'min_samples': 10,
            'cluster_selection_method': 'eom'
        }
        
        # Find all embedding files
        self.embedding_files = list(self.embeddings_dir.glob("*_embeddings.npy"))
        logger.info(f"Found {len(self.embedding_files)} embedding files in {embeddings_dir}")
        
        # Store results for analysis
        self.all_video_metrics = []
        self.processed_videos = {}
    
    def temporal_smoothing(self, labels, window_size=3):
        """Apply median filter to reduce temporal noise."""
        return median_filter(labels, size=window_size, mode='nearest')
    
    def load_frames(self, video_id):
        """Load frames for a video if frames_dir is provided."""
        if self.frames_dir is None:
            return None
            
        # Try different possible locations and formats
        frames_dir_locations = [
            self.frames_dir / video_id,
            self.frames_dir,
            self.frames_dir / "frames",
            self.frames_dir.parent / "frames" / video_id
        ]
        
        logger.info(f"Looking for frames for {video_id}...")
        
        # Try multiple directory structures
        for frames_dir in frames_dir_locations:
            if not frames_dir.exists():
                continue
                
            # Try NPY files
            potential_npy_files = [
                frames_dir / f"{video_id}_frames.npy",
                frames_dir / f"{video_id}.npy",
                frames_dir / "frames.npy"
            ]
            
            for npy_file in potential_npy_files:
                if npy_file.exists():
                    try:
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
                        with h5py.File(h5_file, 'r') as f:
                            # Check different possible dataset names
                            for dataset_name in ['frames', 'images', 'data', video_id]:
                                if dataset_name in f:
                                    frames = f[dataset_name][()]
                                    logger.info(f"Loaded frames with shape: {frames.shape}")
                                    return frames
                    except Exception as e:
                        logger.error(f"Error loading H5 file {h5_file}: {e}")
            
            # Try individual image frames
            frames_folder = frames_dir
            if frames_folder.is_dir():
                image_files = sorted([f for f in frames_folder.glob("*.jpg") or frames_folder.glob("*.png")])
                if image_files:
                    try:
                        first_img = Image.open(image_files[0])
                        img_shape = np.array(first_img).shape
                        frames = np.empty((len(image_files), *img_shape), dtype=np.uint8)
                        
                        for i, img_path in enumerate(tqdm(image_files, desc="Loading frames")):
                            frames[i] = np.array(Image.open(img_path))
                        
                        logger.info(f"Loaded {len(image_files)} frames with shape {frames.shape}")
                        return frames
                    except Exception as e:
                        logger.error(f"Error loading image frames from {frames_folder}: {e}")
        
        # Try directly in processed_frames directory
        try:
            direct_npy_path = self.frames_dir / f"{video_id}.npy"
            if direct_npy_path.exists():
                frames = np.load(direct_npy_path)
                logger.info(f"Loaded frames with shape: {frames.shape}")
                return frames
        except Exception:
            pass
        
        logger.warning(f"Could not find frames for {video_id}")
        return None
    
    def run_clustering(self, embeddings):
        """Run HDBSCAN clustering with the best parameters."""
        clusterer = hdbscan.HDBSCAN(**self.best_params)
        clusterer.fit(embeddings)
        labels = clusterer.labels_
        
        # Apply temporal smoothing
        labels = self.temporal_smoothing(labels)
        
        return labels
    
    def calculate_dbcv(self, embeddings, labels):
        """Calculate DBCV score for clustering quality."""
        try:
            # Skip if all points are noise or only one cluster
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters <= 1:
                return None
            
            # Get valid points (non-noise)
            valid_indices = labels != -1
            valid_embeddings = embeddings[valid_indices]
            valid_labels = labels[valid_indices]
            
            # Skip if too few valid points
            if len(valid_labels) < 3:
                return None
                
            # Ensure embeddings are float64 for numerical stability
            valid_embeddings_64 = valid_embeddings.astype(np.float64)
            
            # Check cluster sizes
            min_points_per_cluster = 3
            cluster_sizes = np.bincount(valid_labels)
            if np.all(cluster_sizes >= min_points_per_cluster):
                from hdbscan.validity import validity_index
                dbcv_score = validity_index(valid_embeddings_64, valid_labels)
                if np.isnan(dbcv_score):
                    return None
                return dbcv_score
        
        except Exception as e:
            logger.error(f"Error calculating DBCV score: {e}")
            
        return None
    
    def create_tsne_projection(self, embeddings, labels, video_id, perplexity=30, quality_tag=""):
        """Create and save an enhanced t-SNE projection."""
        logger.info(f"Creating t-SNE projection for {video_id}...")
        
        # Dynamically adjust perplexity based on number of samples
        perplexity = min(perplexity, max(5, embeddings.shape[0] // 10))
        
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, 
                    perplexity=perplexity, n_iter=300, learning_rate=200)
        
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Calculate DBCV score if possible
        dbcv_score = self.calculate_dbcv(embeddings, labels)
        dbcv_text = f"DBCV: {dbcv_score:.4f}" if dbcv_score is not None else "DBCV: N/A"
        
        # Create a DataFrame for seaborn
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': labels,
            'is_noise': labels == -1,
            'frame_idx': np.arange(len(labels))
        })
        
        # Enhanced t-SNE with cluster info
        plt.figure(figsize=(14, 10))
        
        # Plot non-noise points with cluster coloring
        sns.scatterplot(
            data=df[~df['is_noise']],
            x='x', y='y',
            hue='cluster',
            palette='tab20',
            s=80,
            alpha=0.7,
            legend='full'
        )
        
        # Plot noise points in gray and different shape
        if np.any(df['is_noise']):
            sns.scatterplot(
                data=df[df['is_noise']],
                x='x', y='y',
                color='gray',
                marker='X',
                s=80,
                alpha=0.5,
                label='Noise'
            )
        
        plt.title(f"t-SNE Visualization for {video_id} {quality_tag}\n" + 
                 f"HDBSCAN: min_cluster={self.best_params['min_cluster_size']}, " +
                 f"min_samples={self.best_params['min_samples']}\n" +
                 f"{dbcv_text}, Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        
        # Move legend outside plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the visualization
        output_path = self.output_dir / f"{video_id}_tsne_projection{quality_tag}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved t-SNE projection to {output_path}")
        
        return embeddings_2d
    
    def create_pca_projection(self, embeddings, labels, video_id, quality_tag=""):
        """Create and save a PCA projection for comparison."""
        logger.info(f"Creating PCA projection for {video_id}...")
        
        # Perform PCA dimensionality reduction
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create a DataFrame for seaborn
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': labels,
            'is_noise': labels == -1
        })
        
        # PCA projection with cluster info
        plt.figure(figsize=(14, 10))
        
        # Plot non-noise points with cluster coloring
        sns.scatterplot(
            data=df[~df['is_noise']],
            x='x', y='y',
            hue='cluster',
            palette='tab20',
            s=80,
            alpha=0.7,
            legend='full'
        )
        
        # Plot noise points in gray and different shape
        if np.any(df['is_noise']):
            sns.scatterplot(
                data=df[df['is_noise']],
                x='x', y='y',
                color='gray',
                marker='X',
                s=80,
                alpha=0.5,
                label='Noise'
            )
        
        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        
        plt.title(f"PCA Visualization for {video_id} {quality_tag}\n" + 
                 f"HDBSCAN: min_cluster={self.best_params['min_cluster_size']}, " +
                 f"min_samples={self.best_params['min_samples']}\n" +
                 f"Explained variance: x={explained_var[0]:.2f}, y={explained_var[1]:.2f}")
        
        plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        plt.grid(True, alpha=0.3)
        
        # Move legend outside plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the visualization
        output_path = self.output_dir / f"{video_id}_pca_projection{quality_tag}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved PCA projection to {output_path}")
    
    def create_temporal_consistency_plot(self, labels, video_id, quality_tag=""):
        """Create and save a temporal consistency plot."""
        logger.info(f"Creating temporal consistency plot for {video_id}...")
        
        frame_indices = np.arange(len(labels))
        
        plt.figure(figsize=(15, 5))
        plt.plot(frame_indices, labels, 'o-', markersize=4, alpha=0.7)
        
        # Count transitions between clusters
        transitions = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                transitions += 1
        
        # Add vertical lines at transition points
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                plt.axvline(x=i, color='r', linestyle='--', alpha=0.3)
        
        transition_percentage = 100 * transitions / (len(labels) - 1) if len(labels) > 1 else 0
        
        plt.yticks(sorted(set(labels)))
        plt.xlabel('Frame Index')
        plt.ylabel('Cluster ID')
        plt.title(f'Cluster Membership Over Time for {video_id} {quality_tag}\n' +
                 f'Transitions: {transitions} ({transition_percentage:.2f}% of frames)')
        plt.grid(True, alpha=0.3)
        
        # Save the visualization
        output_path = self.output_dir / f"{video_id}_temporal_consistency{quality_tag}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved temporal consistency plot to {output_path}")
    
    def create_interactive_3d_plot(self, embeddings, labels, video_id, quality_tag=""):
        """Create and save an interactive 3D plot using Plotly."""
        logger.info(f"Creating interactive 3D plot for {video_id}...")
        
        # Perform PCA for 3D visualization
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        
        # Create a DataFrame for plotly
        df = pd.DataFrame({
            'x': embeddings_3d[:, 0],
            'y': embeddings_3d[:, 1],
            'z': embeddings_3d[:, 2],
            'cluster': labels.astype(str),  # Convert to string for categorical coloring
            'frame_idx': np.arange(len(labels))
        })
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='cluster',
            hover_data={'frame_idx': True},
            title=f'3D Cluster Visualization for {video_id} {quality_tag}'
        )
        
        # Set axis labels
        explained_var = pca.explained_variance_ratio_
        fig.update_layout(
            scene=dict(
                xaxis_title=f'PC1 ({explained_var[0]:.1%})',
                yaxis_title=f'PC2 ({explained_var[1]:.1%})',
                zaxis_title=f'PC3 ({explained_var[2]:.1%})'
            )
        )
        
        # Save as HTML for interactivity
        output_path = self.output_dir / f"{video_id}_interactive_3d{quality_tag}.html"
        pio.write_html(fig, file=str(output_path), auto_open=False)
        
        logger.info(f"Saved interactive 3D plot to {output_path}")
    
    def evaluate_all_videos(self):
        """Evaluate all videos to select the best and worst ones."""
        logger.info("Evaluating all videos to find the best and worst performers...")
        
        for file_path in tqdm(self.embedding_files, desc="Evaluating videos"):
            video_id = file_path.stem.replace('_embeddings', '')
            
            try:
                # Load embeddings
                embeddings = np.load(file_path)
                
                if embeddings.shape[0] == 0:
                    logger.warning(f"Empty embeddings for {video_id}, skipping.")
                    continue
                
                # Run clustering with best parameters
                labels = self.run_clustering(embeddings)
                
                # Calculate metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = 100 * np.sum(labels == -1) / len(labels)
                
                # Calculate temporal coherence
                transitions = 0
                for i in range(1, len(labels)):
                    if labels[i] != labels[i-1]:
                        transitions += 1
                transition_percentage = 100 * transitions / (len(labels) - 1) if len(labels) > 1 else 0
                
                # Calculate DBCV score
                dbcv_score = self.calculate_dbcv(embeddings, labels)
                
                # Store metrics
                self.all_video_metrics.append({
                    'video_id': video_id,
                    'embeddings_file': file_path,
                    'total_frames': len(labels),
                    'n_clusters': n_clusters,
                    'noise_ratio': noise_ratio,
                    'temporal_coherence': transition_percentage,
                    'dbcv': dbcv_score,
                    'labels': labels,
                    'embeddings': embeddings
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {video_id}: {e}")
        
        # Filter out videos with invalid DBCV scores
        valid_metrics = [m for m in self.all_video_metrics if m['dbcv'] is not None]
        
        if not valid_metrics:
            logger.warning("No videos with valid DBCV scores found. Defaulting to noise ratio ranking.")
            # Rank by noise ratio (lower is better)
            sorted_metrics = sorted(self.all_video_metrics, key=lambda x: x.get('noise_ratio', 100))
        else:
            # Calculate quality score
            for metrics in valid_metrics:
                # Higher DBCV is better, lower noise ratio is better, moderate temporal coherence is better
                dbcv_norm = metrics['dbcv']  # Higher is better
                noise_penalty = min(1.0, metrics['noise_ratio'] / 100)  # 0 to 1, lower is better
                
                # For temporal coherence, we want some changes but not too many
                # Best around 20-30%, penalize if too high or too low
                tc_score = 1.0 - abs(metrics['temporal_coherence'] - 25) / 100
                tc_score = max(0, min(1, tc_score))
                
                # Combined score - weighted sum
                metrics['quality_score'] = (0.6 * dbcv_norm) - (0.3 * noise_penalty) + (0.1 * tc_score)
            
            # Sort by quality score (higher is better)
            sorted_metrics = sorted(valid_metrics, key=lambda x: x.get('quality_score', -1), reverse=True)
        
        # Get top 5 and bottom 5
        if len(sorted_metrics) >= 10:
            top_videos = sorted_metrics[:5]
            bottom_videos = sorted_metrics[-5:]
        else:
            # If we have fewer than 10 videos, split them
            split_point = max(1, len(sorted_metrics) // 2)
            top_videos = sorted_metrics[:split_point]
            bottom_videos = sorted_metrics[-split_point:] if len(sorted_metrics) > split_point else []
        
        # Mark top and bottom videos
        for video in top_videos:
            video['quality_category'] = 'top'
        for video in bottom_videos:
            video['quality_category'] = 'bottom'
        
        # Store the videos to process
        self.videos_to_process = top_videos + bottom_videos
        
        # Log results
        logger.info(f"Selected {len(top_videos)} top videos and {len(bottom_videos)} bottom videos for visualization")
        
        # Log top videos
        logger.info("=== Top Videos ===")
        for i, video in enumerate(top_videos):
            if video.get('dbcv') is not None:
                logger.info(f"{i+1}. {video['video_id']}: DBCV={video['dbcv']:.4f}, Clusters={video['n_clusters']}, "
                          f"Noise={video['noise_ratio']:.2f}%, Temporal={video['temporal_coherence']:.2f}%")
            else:
                logger.info(f"{i+1}. {video['video_id']}: No DBCV, Clusters={video['n_clusters']}, "
                          f"Noise={video['noise_ratio']:.2f}%, Temporal={video['temporal_coherence']:.2f}%")
        
        # Log bottom videos
        logger.info("=== Bottom Videos ===")
        for i, video in enumerate(bottom_videos):
            if video.get('dbcv') is not None:
                logger.info(f"{i+1}. {video['video_id']}: DBCV={video['dbcv']:.4f}, Clusters={video['n_clusters']}, "
                          f"Noise={video['noise_ratio']:.2f}%, Temporal={video['temporal_coherence']:.2f}%")
            else:
                logger.info(f"{i+1}. {video['video_id']}: No DBCV, Clusters={video['n_clusters']}, "
                          f"Noise={video['noise_ratio']:.2f}%, Temporal={video['temporal_coherence']:.2f}%")
        
        return self.videos_to_process
    
    def visualize_selected_videos(self):
        """Generate visualizations for selected videos."""
        if not hasattr(self, 'videos_to_process') or not self.videos_to_process:
            logger.error("No videos selected for visualization. Run evaluate_all_videos() first.")
            return
        
        logger.info(f"Generating visualizations for {len(self.videos_to_process)} selected videos...")
        
        for video_data in tqdm(self.videos_to_process, desc="Creating visualizations"):
            video_id = video_data['video_id']
            embeddings = video_data['embeddings']
            labels = video_data['labels']
            quality_tag = f"_{video_data['quality_category']}"
            
            # Generate the four visualizations we're keeping
            self.create_tsne_projection(embeddings, labels, video_id, quality_tag=quality_tag)
            self.create_pca_projection(embeddings, labels, video_id, quality_tag=quality_tag)
            self.create_temporal_consistency_plot(labels, video_id, quality_tag=quality_tag)
            self.create_interactive_3d_plot(embeddings, labels, video_id, quality_tag=quality_tag)
            
            # Store processed video info for summary
            self.processed_videos[video_id] = {
                'total_frames': video_data['total_frames'],
                'n_clusters': video_data['n_clusters'],
                'noise_ratio': video_data['noise_ratio'],
                'temporal_coherence': video_data['temporal_coherence'],
                'dbcv': video_data['dbcv'],
                'params': self.best_params,
                'quality_category': video_data['quality_category']
            }
    
    def create_summary_plots(self):
        """Create summary plots for all processed videos."""
        if not self.processed_videos:
            logger.warning("No processed videos to create summary plots.")
            return
            
        logger.info("Creating summary plots...")
        
        # Convert results to DataFrame
        df = pd.DataFrame([
            {'video_id': k, **v} for k, v in self.processed_videos.items()
        ])
        
        # Filter out rows with missing dbcv values for some plots
        df_valid = df.dropna(subset=['dbcv'])
        
        # 1. DBCV vs noise ratio scatter plot
        if len(df_valid) > 0:
            plt.figure(figsize=(12, 8))
            scatter = sns.scatterplot(
                data=df_valid, 
                x='noise_ratio', 
                y='dbcv', 
                size='total_frames',
                hue='quality_category', 
                palette={'top': 'green', 'bottom': 'red'},
                sizes=(100, 500), 
                alpha=0.7,
                style='quality_category'
            )
            
            # Add video labels to points
            for _, row in df_valid.iterrows():
                plt.text(
                    row['noise_ratio'] + 1, 
                    row['dbcv'], 
                    row['video_id'],
                    fontsize=9
                )
                
            plt.xlabel('Noise Ratio (%)')
            plt.ylabel('DBCV Score')
            plt.title('DBCV Score vs Noise Ratio - Top vs Bottom Videos')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the visualization
            output_path = self.output_dir / "summary_dbcv_vs_noise.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
        
        # 2. DBCV vs temporal coherence scatter plot
        if len(df_valid) > 0:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=df_valid, 
                x='temporal_coherence', 
                y='dbcv', 
                size='total_frames',
                hue='quality_category', 
                palette={'top': 'green', 'bottom': 'red'},
                sizes=(100, 500), 
                alpha=0.7,
                style='quality_category'
            )
            
            # Add video labels to points
            for _, row in df_valid.iterrows():
                plt.text(
                    row['temporal_coherence'] + 1, 
                    row['dbcv'], 
                    row['video_id'],
                    fontsize=9
                )
                
            plt.xlabel('Temporal Coherence (% transitions)')
            plt.ylabel('DBCV Score')
            plt.title('DBCV Score vs Temporal Coherence - Top vs Bottom Videos')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the visualization
            output_path = self.output_dir / "summary_dbcv_vs_temporal.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
        
        # 3. Number of clusters comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='quality_category', y='n_clusters', palette={'top': 'green', 'bottom': 'red'})
        sns.stripplot(data=df, x='quality_category', y='n_clusters', color='black', alpha=0.5)
        
        plt.xlabel('Video Quality Category')
        plt.ylabel('Number of Clusters')
        plt.title('Number of Clusters by Quality Category')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save the visualization
        output_path = self.output_dir / "summary_n_clusters_comparison.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        # 4. Correlation matrix of metrics
        if len(df_valid) > 0:
            plt.figure(figsize=(10, 8))
            
            # Select numeric columns for correlation
            numeric_cols = ['total_frames', 'n_clusters', 'noise_ratio', 'temporal_coherence', 'dbcv']
            corr_df = df_valid[numeric_cols].corr()
            
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            plt.title('Correlation Matrix of Clustering Metrics')
            plt.tight_layout()
            
            # Save the visualization
            output_path = self.output_dir / "summary_correlation_matrix.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
        
        # 5. Metrics comparison by category
        plt.figure(figsize=(14, 10))
        
        # Create a melted dataframe for category comparison
        metrics_to_compare = ['noise_ratio', 'temporal_coherence']
        if len(df_valid) > 0:
            metrics_to_compare.append('dbcv')
            
        melted_df = pd.melt(
            df, 
            id_vars=['video_id', 'quality_category'],
            value_vars=metrics_to_compare,
            var_name='Metric', 
            value_name='Value'
        )
        
        # Create a facet grid
        g = sns.catplot(
            data=melted_df,
            x='quality_category',
            y='Value',
            col='Metric',
            kind='box',
            height=5,
            aspect=0.8,
            palette={'top': 'green', 'bottom': 'red'},
            sharey=False
        )
        
        # Add individual data points
        for ax, metric in zip(g.axes.flat, metrics_to_compare):
            sns.stripplot(
                data=melted_df[melted_df['Metric'] == metric],
                x='quality_category',
                y='Value',
                ax=ax,
                color='black',
                alpha=0.5
            )
            
        g.set_axis_labels("Video Quality Category", "Value")
        g.set_titles("{col_name}")
        g.fig.suptitle("Comparison of Metrics Between Top and Bottom Videos", y=1.05, fontsize=16)
        
        # Save the visualization
        output_path = self.output_dir / "summary_metrics_comparison.png"
        g.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved summary plots to {self.output_dir}")
        
        # Also save results as CSV
        results_path = self.output_dir / "visualization_results.csv"
        df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")
    
    def run(self):
        """Run visualizations for the best and worst videos."""
        logger.info(f"Starting visualization process for the top and bottom videos...")
        
        # First, evaluate all videos to find the best and worst ones
        self.evaluate_all_videos()
        
        # Generate visualizations for the selected videos
        self.visualize_selected_videos()
        
        # Create summary plots
        self.create_summary_plots()
        
        logger.info("Visualization process complete!")

def main():
    parser = argparse.ArgumentParser(description="Create visualizations for top and bottom performing videos with HDBSCAN clustering.")
    parser.add_argument("--embeddings_dir", type=str, default="swin_embeddings",
                        help="Directory containing embedding .npy files")
    parser.add_argument("--frames_dir", type=str, default="/Users/akshat/Developer/Vid_Attention/data_extraction/data/QuerYD/processed_frames",
                        help="Directory containing video frames (for thumbnails)")
    parser.add_argument("--output_dir", type=str, default="hdbscan_visualizations",
                        help="Directory to save visualization results")
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ClusteringVisualizer(
        embeddings_dir=args.embeddings_dir,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir
    )
    
    # Run visualizations
    visualizer.run()

if __name__ == "__main__":
    main()