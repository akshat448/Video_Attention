import numpy as np
import hdbscan
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import warnings
import time
import h5py
from scipy.ndimage import median_filter
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from hdbscan.validity import validity_index  # Import for DBCV metric
from PIL import Image
import shutil
import gc
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Particle:
    """
    Represents a particle in the PSO algorithm, which corresponds to a 
    specific HDBSCAN parameter configuration.
    """
    def __init__(self, bounds, cluster_selection_method='eom'):
        """
        Initialize a particle with random position within the parameter bounds.
        
        Args:
            bounds: Dictionary mapping parameter names to (min, max) tuples
            cluster_selection_method: The cluster selection method to use
        """
        self.position = {}
        self.velocity = {}
        self.best_position = {}
        self.fitness = -float('inf')  # Initialize with worst possible fitness
        self.best_fitness = -float('inf')
        
        # Initialize position and velocity for each parameter
        for param, (min_val, max_val) in bounds.items():
            # For integer parameters, randomly initialize within bounds
            if param in ['min_cluster_size', 'min_samples']:
                self.position[param] = np.random.randint(min_val, max_val + 1)
                # Initialize velocity as a float that will be rounded later
                self.velocity[param] = np.random.uniform(-1, 1)
            
        # Set the fixed parameter
        self.position['cluster_selection_method'] = cluster_selection_method
        
        # Initialize best position to current position
        self.best_position = self.position.copy()

class HDBSCANOptimizer:
    """PSO-based optimization for HDBSCAN parameters."""
    
    def __init__(self, embeddings_dir, frames_dir=None, output_dir="hdbscan_pso_results", checkpoint_interval=10):
        """
        Initialize PSO optimization framework.
        
        Args:
            embeddings_dir: Directory containing embedding .npy files
            frames_dir: Directory containing video frames (for visualization)
            output_dir: Directory to save results
            checkpoint_interval: Save checkpoints every N videos processed
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_interval = checkpoint_interval
        
        # Create a subdirectory for visualizations
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Load all embedding files
        self.embedding_files = list(self.embeddings_dir.glob('*.npy'))
        logger.info(f"Found {len(self.embedding_files)} embedding files in {embeddings_dir}")
        
        # Define parameter bounds for PSO
        self.param_bounds = {
            'min_cluster_size': (2, 25),  # Expanded range for min_cluster_size
            'min_samples': (2, 20)        # Expanded range for min_samples
        }
        
        # PSO hyperparameters
        self.n_particles = 25             # Increased for better exploration
        self.n_iterations = 30            # Increased for better convergence
        self.w = 0.7                      # Inertia weight
        self.c1 = 1.5                     # Cognitive weight (particle's own best)
        self.c2 = 1.5                     # Social weight (global best)
        
        # Store all results from PSO
        self.all_results = []
        
        # Store top parameter sets
        self.top_parameter_sets = []
        
        # Store raw clustering results for best visualizations
        self.clustering_results = {}
        
        # Store the best overall parameter combination
        self.best_global_params = None
        
        # Keep track of processed videos
        self.processed_videos = set()
        
        # Store particles and global best
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = -float('inf')
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / "pso_checkpoint.npz"
    
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
            metrics['calinski_harabasz'] = None
            metrics['davies_bouldin'] = None
            return metrics

        # Get valid points (non-noise)
        valid_indices = labels != -1
        valid_embeddings = embeddings[valid_indices]
        valid_labels = labels[valid_indices]

        # Skip if too few valid points
        if len(valid_labels) < 3:
            metrics['silhouette'] = None
            metrics['dbcv'] = None
            metrics['calinski_harabasz'] = None
            metrics['davies_bouldin'] = None
            return metrics

        # Calculate silhouette score
        try:
            metrics['silhouette'] = silhouette_score(valid_embeddings, valid_labels)
        except Exception as e:
            logger.error(f"Error calculating silhouette score: {e}")
            metrics['silhouette'] = None

        # Calculate Calinski-Harabasz score
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(valid_embeddings, valid_labels)
        except Exception as e:
            logger.error(f"Error calculating Calinski-Harabasz score: {e}")
            metrics['calinski_harabasz'] = None
            
        # Calculate Davies-Bouldin score (lower is better)
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(valid_embeddings, valid_labels)
        except Exception as e:
            logger.error(f"Error calculating Davies-Bouldin score: {e}")
            metrics['davies_bouldin'] = None

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
                    metrics['dbcv'] = None
                else:
                    metrics['dbcv'] = dbcv_score
            else:
                metrics['dbcv'] = None

        except Exception as e:
            logger.error(f"Error calculating DBCV score: {e}")
            metrics['dbcv'] = None

        return metrics
    
    def evaluate_particle(self, particle, video_data_list):
        """
        Evaluate a particle's fitness across multiple videos.
        
        Args:
            particle: Particle object with position to evaluate
            video_data_list: List of dictionaries with video embeddings data
            
        Returns:
            fitness: Combined fitness score
        """
        all_metrics = []
        
        for video_data in video_data_list:
            video_id = video_data['video_id']
            embeddings = video_data['embeddings']
            
            # Skip videos with too few frames for clustering
            if embeddings.shape[0] < particle.position['min_cluster_size']:
                continue
            
            try:
                # Run HDBSCAN with the particle's parameters
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=particle.position['min_cluster_size'],
                    min_samples=particle.position['min_samples'],
                    cluster_selection_method=particle.position['cluster_selection_method']
                )
                clusterer.fit(embeddings)
                labels = clusterer.labels_
                
                # Apply temporal smoothing
                labels = self.temporal_smoothing(labels)
                
                # Calculate metrics
                metrics = self.calculate_metrics(embeddings, labels)
                
                # Store the result
                result = {
                    'video_id': video_id,
                    'total_frames': embeddings.shape[0],
                    'iteration': getattr(particle, 'iteration', 0),
                    **particle.position,
                    **metrics
                }
                
                all_metrics.append(metrics)
                self.all_results.append(result)
                
                # Store raw clustering results for visualization (only for best sets)
                key = (video_id, particle.position['min_cluster_size'], 
                      particle.position['min_samples'])
                
                # Only store if we have valid scores and at least one non-noise cluster
                has_valid_score = metrics.get('dbcv') is not None or metrics.get('calinski_harabasz') is not None
                if has_valid_score and metrics.get('n_clusters', 0) > 0:
                    self.clustering_results[key] = {
                        'embeddings': embeddings,
                        'labels': labels,
                        'metrics': metrics,
                        'params': particle.position.copy()
                    }
                
            except Exception as e:
                logger.error(f"Error evaluating particle on {video_id}: {e}")
        
        # Calculate combined fitness across all videos
        if not all_metrics:
            return -float('inf')  # Worst possible fitness if no videos could be processed
        
        # Extract valid scores from all metrics
        valid_dbcv = [m['dbcv'] for m in all_metrics if m.get('dbcv') is not None]
        valid_silhouette = [m['silhouette'] for m in all_metrics if m.get('silhouette') is not None]
        valid_ch = [m['calinski_harabasz'] for m in all_metrics if m.get('calinski_harabasz') is not None]
        valid_db = [m['davies_bouldin'] for m in all_metrics if m.get('davies_bouldin') is not None]
        valid_tc = [m['temporal_coherence'] for m in all_metrics if m.get('temporal_coherence') is not None]
        valid_nr = [m['noise_ratio'] for m in all_metrics if m.get('noise_ratio') is not None]
        
        # Compute a comprehensive fitness score using all available metrics
        fitness_components = []
        
        # DBCV score - specific to density-based clustering (higher is better)
        if valid_dbcv:
            dbcv_score = np.mean(valid_dbcv)
            fitness_components.append(("dbcv", dbcv_score, 0.40))  # Highest weight
        
        # Silhouette score (higher is better)
        if valid_silhouette:
            silhouette_score = np.mean(valid_silhouette)
            # Normalize to 0-1 range (silhouette is between -1 and 1)
            silhouette_norm = (silhouette_score + 1) / 2
            fitness_components.append(("silhouette", silhouette_norm, 0.15))
        
        # Calinski-Harabasz score (higher is better)
        if valid_ch:
            ch_mean = np.mean(valid_ch)
            # Log transform to handle potentially large values
            ch_norm = min(1.0, np.log1p(ch_mean) / 10.0)  
            fitness_components.append(("calinski_harabasz", ch_norm, 0.10))
            
        # Davies-Bouldin score (lower is better)
        if valid_db:
            db_mean = np.mean(valid_db)
            # Invert and normalize (typical range is 0 to 2+)
            db_norm = max(0, 1.0 - (db_mean / 4.0))
            fitness_components.append(("davies_bouldin", db_norm, 0.10))
        
        # Temporal coherence (we want moderate values around 25%)
        if valid_tc:
            tc_mean = np.mean(valid_tc)
            # Penalize if too high or too low - peak at 25%
            tc_norm = 1.0 - abs(tc_mean - 25) / 100
            tc_norm = max(0, min(1, tc_norm))
            fitness_components.append(("temporal_coherence", tc_norm, 0.15))
        
        # Noise ratio (lower is better but some noise is acceptable)
        if valid_nr:
            nr_mean = np.mean(valid_nr)
            # Penalize excessive noise (>50%), but allow some noise
            nr_norm = 1.0 - min(nr_mean, 50) / 50
            fitness_components.append(("noise_ratio", nr_norm, 0.10))

        # If we don't have any metrics, return worst possible fitness
        if not fitness_components:
            return -float('inf')
            
        # Compute weighted average
        total_weight = sum(weight for _, _, weight in fitness_components)
        fitness = sum(score * weight for _, score, weight in fitness_components) / total_weight
        
        # Add a small stability bonus for more clusters
        n_clusters_mean = np.mean([m['n_clusters'] for m in all_metrics])
        cluster_bonus = min(0.05, n_clusters_mean / 100)  # Small bonus for more clusters
        
        return fitness + cluster_bonus
    
    def update_particle(self, particle):
        """
        Update a particle's velocity and position based on PSO equations.
        
        Args:
            particle: Particle object to update
        """
        for param in self.param_bounds.keys():
            # Update velocity
            r1, r2 = np.random.rand(2)  # Random values between 0 and 1
            
            cognitive_component = self.c1 * r1 * (particle.best_position[param] - particle.position[param])
            social_component = self.c2 * r2 * (self.global_best_position[param] - particle.position[param])
            
            particle.velocity[param] = self.w * particle.velocity[param] + cognitive_component + social_component
            
            # Update position (ensure integer values for HDBSCAN parameters)
            new_position = particle.position[param] + round(particle.velocity[param])
            
            # Ensure position stays within bounds
            min_val, max_val = self.param_bounds[param]
            particle.position[param] = max(min_val, min(max_val, new_position))
    
    def save_checkpoint(self, iteration, video_data_processed):
        """Save checkpoint of PSO state to resume later if needed."""
        try:
            # Extract essential state
            particles_data = [
                {
                    'position': p.position,
                    'velocity': p.velocity,
                    'best_position': p.best_position,
                    'best_fitness': p.best_fitness
                } for p in self.particles
            ]
            
            checkpoint = {
                'iteration': iteration,
                'global_best_position': self.global_best_position,
                'global_best_fitness': self.global_best_fitness,
                'particles_data': particles_data,
                'n_videos_processed': len(video_data_processed),
                'processed_videos': list(self.processed_videos)
            }
            
            # Convert to DataFrame and save
            pd.to_pickle(checkpoint, self.checkpoint_file)
            logger.info(f"Saved checkpoint at iteration {iteration}")
            
            # Also save current results to CSV
            results_df = pd.DataFrame(self.all_results)
            if not results_df.empty:
                results_df.to_csv(self.output_dir / "pso_interim_results.csv", index=False)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self):
        """Load PSO state from checkpoint if available."""
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint found, starting fresh")
            return None, set()
        
        try:
            checkpoint = pd.read_pickle(self.checkpoint_file)
            logger.info(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
            
            # Restore particles
            self.particles = []
            for p_data in checkpoint['particles_data']:
                p = Particle(self.param_bounds)
                p.position = p_data['position']
                p.velocity = p_data['velocity']
                p.best_position = p_data['best_position']
                p.best_fitness = p_data['best_fitness']
                self.particles.append(p)
            
            # Restore global best
            self.global_best_position = checkpoint['global_best_position']
            self.global_best_fitness = checkpoint['global_best_fitness']
            
            # Restore processed videos
            self.processed_videos = set(checkpoint['processed_videos'])
            
            return checkpoint['iteration'], self.processed_videos
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None, set()
    
    def free_memory(self):
        """Free memory to prevent out-of-memory errors."""
        gc.collect()
    
    def run_pso(self, sample_size=None, resume=True):
        """
        Run PSO to find optimal HDBSCAN parameters.
        
        Args:
            sample_size: Number of videos to sample (None = all)
            resume: Whether to resume from checkpoint if available
        """
        logger.info("=== Running PSO for HDBSCAN Parameter Optimization ===")
        start_time = time.time()
        
        # Sample videos if needed
        if sample_size is not None and sample_size < len(self.embedding_files):
            # Set a random seed for reproducibility
            np.random.seed(42)
            random.seed(42)
            files_to_process = np.random.choice(self.embedding_files, size=sample_size, replace=False)
        else:
            files_to_process = self.embedding_files
        
        # Check for checkpoint
        start_iteration = 0
        processed_videos = set()
        if resume:
            start_iteration, processed_videos = self.load_checkpoint()
        
        # Load video data in batches to manage memory
        logger.info(f"Loading and processing videos in batches...")
        
        all_video_data = []
        batch_size = min(50, len(files_to_process))  # Process in batches of 50 videos or fewer
        
        for batch_start in range(0, len(files_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(files_to_process))
            batch_files = files_to_process[batch_start:batch_end]
            
            # Load video data
            batch_video_data = []
            for file_path in tqdm(batch_files, desc=f"Loading videos {batch_start+1}-{batch_end}/{len(files_to_process)}"):
                video_id = file_path.stem.replace('_embeddings', '')
                
                # Skip already processed videos when resuming
                if video_id in processed_videos and resume:
                    continue
                    
                try:
                    embeddings = np.load(file_path)
                    if embeddings.shape[0] > 0:  # Skip empty embeddings
                        batch_video_data.append({
                            'video_id': video_id,
                            'embeddings': embeddings
                        })
                        self.processed_videos.add(video_id)
                    else:
                        logger.warning(f"Empty embeddings for {video_id}, skipping.")
                except Exception as e:
                    logger.error(f"Error loading embeddings for {video_id}: {e}")
            
            all_video_data.extend(batch_video_data)
            
            # Free memory after processing each batch
            self.free_memory()
        
        logger.info(f"Loaded {len(all_video_data)} valid videos for optimization")
        
        # Initialize particles if starting fresh
        if start_iteration == 0:
            self.particles = [Particle(self.param_bounds) for _ in range(self.n_particles)]
            logger.info(f"Initialized {self.n_particles} particles for PSO")
        else:
            logger.info(f"Resuming with {len(self.particles)} particles from iteration {start_iteration}")
        
        # Main PSO loop
        for iteration in range(start_iteration, self.n_iterations):
            logger.info(f"Starting PSO iteration {iteration+1}/{self.n_iterations}")
            iteration_start = time.time()
            
            # Set iteration attribute for tracking
            for p in self.particles:
                p.iteration = iteration
            
            # Evaluate fitness for each particle
            for i, particle in enumerate(tqdm(self.particles, desc=f"Evaluating particles (iteration {iteration+1})")):
                fitness = self.evaluate_particle(particle, all_video_data)
                
                # Update particle's best if current fitness is better
                if fitness > particle.best_fitness:
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = fitness
                
                # Update global best if this particle is better
                if fitness > self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.global_best_fitness = fitness
                    logger.info(f"New global best: {self.global_best_position}, fitness: {self.global_best_fitness:.4f}")
            
            # Save checkpoint after each iteration
            self.save_checkpoint(iteration + 1, all_video_data)
            
            # Calculate iteration time
            iteration_time = time.time() - iteration_start
            logger.info(f"Iteration {iteration+1} completed in {iteration_time:.2f}s")
            
            # Free memory after each iteration
            self.free_memory()
            
            # Early stopping if we're in the last iteration
            if iteration == self.n_iterations - 1:
                break
                
            # Update particles for next iteration
            for particle in tqdm(self.particles, desc=f"Updating particles for iteration {iteration+2}"):
                self.update_particle(particle)
        
    # Store the best parameters found
        if self.global_best_position:
            self.best_global_params = self.global_best_position.copy()
            logger.info("\n=== BEST PSO PARAMETERS ===")
            for param, value in self.best_global_params.items():
                logger.info(f"{param} = {value}")
            logger.info(f"Fitness: {self.global_best_fitness:.4f}")
            logger.info("================================")
            
            # Apply best parameters to all videos
            self.apply_best_params_to_all_videos(self.best_global_params)
        else:
            logger.error("PSO failed to find a valid parameter set")
        
        # Save and analyze results
        self._save_and_analyze_results()
        
        # Generate summary visualization plots
        self.generate_summary_plots()
        
        # Log total runtime
        total_time = time.time() - start_time
        logger.info(f"PSO completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    def _save_and_analyze_results(self):
        """Save results and identify top parameter combinations."""
        if not self.all_results:
            logger.warning("No valid results to analyze")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)
        
        # Save raw results
        output_path = self.output_dir / "pso_all_results.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"All results saved to {output_path}")
        
        # Filter rows for analysis - use any available metric
        df_valid = df[(df['dbcv'].notna()) | 
                      (df['silhouette'].notna()) | 
                      (df['calinski_harabasz'].notna()) |
                      (df['davies_bouldin'].notna())]
        
        if len(df_valid) == 0:
            logger.warning("No valid clustering scores to analyze")
            return
        
        # Group by parameter combination
        param_columns = ['min_cluster_size', 'min_samples', 'cluster_selection_method']
        metrics_of_interest = ['n_clusters', 'noise_ratio', 'temporal_coherence', 
                              'dbcv', 'silhouette', 'calinski_harabasz', 'davies_bouldin']
        
        try:
            # Group by parameter combination and calculate metrics
            grouped_results = df_valid.groupby(param_columns)[metrics_of_interest].mean().reset_index()
            
            # Create a weighted score for ranking
            # Prepare component scores, handling NaN values
            grouped_results['dbcv_norm'] = grouped_results['dbcv'].fillna(0)
            
            # Silhouette: transform from [-1,1] to [0,1]
            grouped_results['silhouette_norm'] = (grouped_results['silhouette'].fillna(0) + 1) / 2
            
            # Calinski-Harabasz: log-normalize (higher is better)
            grouped_results['ch_norm'] = np.log1p(grouped_results['calinski_harabasz'].fillna(1)) / 10.0
            grouped_results['ch_norm'] = grouped_results['ch_norm'].clip(upper=1.0)
            
            # Davies-Bouldin: invert and normalize (lower is better)
            grouped_results['db_norm'] = 1.0 - grouped_results['davies_bouldin'].fillna(4) / 4.0
            grouped_results['db_norm'] = grouped_results['db_norm'].clip(lower=0.0)
            
            # Temporal coherence: penalize if far from 25%
            grouped_results['tc_norm'] = 1.0 - abs(grouped_results['temporal_coherence'] - 25) / 100
            grouped_results['tc_norm'] = grouped_results['tc_norm'].clip(lower=0.0, upper=1.0)
            
            # Noise ratio: penalize high noise
            grouped_results['nr_norm'] = 1.0 - grouped_results['noise_ratio'].clip(upper=50) / 50
            
            # Compute combined score with weights
            grouped_results['combined_score'] = (
                0.30 * grouped_results['dbcv_norm'] +
                0.15 * grouped_results['silhouette_norm'] +
                0.15 * grouped_results['ch_norm'] +
                0.10 * grouped_results['db_norm'] +
                0.15 * grouped_results['tc_norm'] +
                0.15 * grouped_results['nr_norm']
            )
            
            # Sort by combined score
            sorted_results = grouped_results.sort_values('combined_score', ascending=False)
            
            # Save sorted parameter combinations
            best_params_path = self.output_dir / "pso_parameter_ranking.csv"
            sorted_results.to_csv(best_params_path, index=False)
            logger.info(f"Parameter ranking saved to {best_params_path}")
            
            # Get top 5 parameter combinations
            top_5_params = sorted_results.head(5)
            
            # Save to text file
            top_params_path = self.output_dir / "top_5_parameters.txt"
            
            with open(top_params_path, 'w') as f:
                f.write("=== TOP 5 HDBSCAN PARAMETER SETS ===\n\n")
                for i, (_, row) in enumerate(top_5_params.iterrows()):
                    f.write(f"Rank {i+1}:\n")
                    f.write(f"  min_cluster_size = {int(row['min_cluster_size'])}\n")
                    f.write(f"  min_samples = {int(row['min_samples'])}\n")
                    f.write(f"  cluster_selection_method = '{row['cluster_selection_method']}'\n\n")
                    
                    f.write(f"  Clustering Metrics:\n")
                    f.write(f"    DBCV: {row['dbcv']:.4f}\n" if not pd.isna(row['dbcv']) else "    DBCV: N/A\n")
                    f.write(f"    Silhouette: {row['silhouette']:.4f}\n" if not pd.isna(row['silhouette']) else "    Silhouette: N/A\n")
                    f.write(f"    Calinski-Harabasz: {row['calinski_harabasz']:.2f}\n" if not pd.isna(row['calinski_harabasz']) else "    Calinski-Harabasz: N/A\n")
                    f.write(f"    Davies-Bouldin: {row['davies_bouldin']:.4f}\n" if not pd.isna(row['davies_bouldin']) else "    Davies-Bouldin: N/A\n")
                    f.write(f"    Avg. Clusters: {row['n_clusters']:.2f}\n")
                    f.write(f"    Avg. Noise Ratio: {row['noise_ratio']:.2f}%\n")
                    f.write(f"    Avg. Temporal Coherence: {row['temporal_coherence']:.2f}%\n")
                    f.write(f"    Combined Score: {row['combined_score']:.4f}\n\n")
                    
                    # Store for later use
                    self.top_parameter_sets.append({
                        'min_cluster_size': int(row['min_cluster_size']),
                        'min_samples': int(row['min_samples']),
                        'cluster_selection_method': row['cluster_selection_method']
                    })
            
            logger.info(f"Top 5 parameter sets saved to {top_params_path}")
            
            # Save parameter stats by video
            by_video = df_valid.groupby(['video_id', *param_columns])[metrics_of_interest].mean().reset_index()
            by_video_path = self.output_dir / "pso_parameter_stats_by_video.csv"
            by_video.to_csv(by_video_path, index=False)
            logger.info(f"Parameter statistics by video saved to {by_video_path}")
            
        except Exception as e:
            logger.error(f"Error during result analysis: {e}")
    
    def apply_best_params_to_all_videos(self, best_params):
        """
        Apply the best parameter combination to all videos.
        Generate visualizations and store frames in clusters.
        
        Args:
            best_params: The best parameter combination found by PSO
        """
        logger.info("=== Applying Best PSO Parameters to All Videos ===")
        
        # Process each video with the best parameters (in batches to manage memory)
        batch_size = 50  # Process in batches of 50 videos
        
        for batch_start in range(0, len(self.embedding_files), batch_size):
            batch_end = min(batch_start + batch_size, len(self.embedding_files))
            batch_files = self.embedding_files[batch_start:batch_end]
            
            for file_path in tqdm(batch_files, 
                                 desc=f"Processing videos {batch_start+1}-{batch_end}/{len(self.embedding_files)} with best parameters"):
                video_id = file_path.stem.replace('_embeddings', '')
                
                try:
                    # Load embeddings
                    embeddings = np.load(file_path)
                    if embeddings.shape[0] < best_params['min_cluster_size']:
                        logger.warning(f"Skipping {video_id}: Not enough frames ({embeddings.shape[0]}) for min_cluster_size={best_params['min_cluster_size']}")
                        continue
                    
                    # Run clustering with best parameters
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=best_params['min_cluster_size'],
                        min_samples=best_params['min_samples'],
                        cluster_selection_method=best_params['cluster_selection_method']
                    )
                    clusterer.fit(embeddings)
                    labels = self.temporal_smoothing(clusterer.labels_)
                    
                    # Calculate metrics for reporting
                    metrics = self.calculate_metrics(embeddings, labels)
                    
                    # Create t-SNE visualization
                    self.visualize_clusters(embeddings, labels, video_id, best_params, best=True)
                    
                    # Store cluster frames
                    if self.frames_dir:
                        self.store_cluster_frames(video_id, labels, best=True)
                    
                    # Log metrics for this video
                    logger.info(f"Video {video_id}: Clusters={metrics['n_clusters']}, Noise={metrics['noise_ratio']:.2f}%, " +
                              f"Temporal Coherence={metrics['temporal_coherence']:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error processing {video_id} with best parameters: {e}")
            
            # Free memory after each batch
            self.free_memory()
        
        logger.info("=== Completed Processing All Videos with Best Parameters ===")
    
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
            import matplotlib.pyplot as plt
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
                      f'{"(Best PSO Parameters)" if best else ""}')
            
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
                param_str = f"BEST_PSO_{param_str}"
            viz_filename = f"{video_id}_{param_str}_tsne.png"
            viz_path = self.viz_dir / viz_filename
            plt.savefig(viz_path, dpi=150)
            plt.close()
            
            logger.info(f"Saved t-SNE visualization to {viz_path}")
            
            # Free memory
            del embeddings_2d
            self.free_memory()
            
        except Exception as e:
            logger.error(f"Error creating t-SNE visualization for {video_id}: {e}")
    
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
                video_output_dir = self.output_dir / f"{video_id}_best_pso"
            
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
            
            # Free memory
            del frames
            self.free_memory()
            
        except Exception as e:
            logger.error(f"Error storing cluster frames for {video_id}: {e}")
            
    def generate_summary_plots(self):
        """Generate comprehensive summary plots to visualize PSO optimization results."""
        logger.info("Generating explanatory visualization plots...")
        
        # Create plots directory
        plots_dir = self.output_dir / "summary_plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Load results from CSV if available, otherwise use the in-memory results
        try:
            results_csv = self.output_dir / "pso_all_results.csv"
            if results_csv.exists():
                results_df = pd.read_csv(results_csv)
                logger.info(f"Loaded {len(results_df)} results from {results_csv}")
            else:
                results_df = pd.DataFrame(self.all_results)
                logger.info(f"Using {len(results_df)} results from memory")
            
            # Load parameter ranking if available
            params_csv = self.output_dir / "pso_parameter_ranking.csv"
            if params_csv.exists():
                params_df = pd.read_csv(params_csv)
                logger.info(f"Loaded parameter ranking from {params_csv}")
            else:
                # We'll do without it
                params_df = None
                
            # 1. PSO Convergence Plot
            self._plot_convergence(results_df, plots_dir)
            
            # 2. Parameter Distribution and Performance 
            self._plot_parameter_performance(results_df, plots_dir)
            
            # 3. Top Parameters Comparison
            self._plot_top_parameters_comparison(results_df, params_df, plots_dir)
            
            # 4. Metric Correlations
            self._plot_metric_correlations(results_df, plots_dir)
            
            # 5. Cluster Distribution Analysis
            self._plot_cluster_distribution(results_df, plots_dir)
            
            # 6. Parameter Sensitivity Analysis
            self._plot_parameter_sensitivity(results_df, plots_dir)
            
            # 7. Summary Dashboard
            self._create_summary_dashboard(results_df, params_df, plots_dir)
            
            logger.info(f"All summary plots saved to {plots_dir}")
            
        except Exception as e:
            logger.error(f"Error generating summary plots: {e}")
            
    def _plot_convergence(self, results_df, plots_dir):
        """Plot PSO convergence over iterations."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Get average fitness per iteration
            if 'iteration' in results_df.columns:
                iteration_data = results_df.groupby('iteration')
                
                # Prepare data for plotting
                metrics = ['dbcv', 'silhouette', 'noise_ratio', 'n_clusters', 'temporal_coherence']
                available_metrics = [m for m in metrics if m in results_df.columns]
                
                # Create plot with two y-axes
                fig, ax1 = plt.subplots(figsize=(12, 8))
                ax2 = ax1.twinx()
                
                # Plot metric trends
                for i, metric in enumerate(available_metrics):
                    if metric in ['noise_ratio', 'temporal_coherence']:
                        # Plot percentages on right y-axis
                        mean_values = iteration_data[metric].mean()
                        ax2.plot(mean_values.index, mean_values.values, 
                                marker='o', linestyle='-', 
                                label=f'Avg {metric.replace("_", " ").title()}')
                    else:
                        # Plot scores on left y-axis
                        mean_values = iteration_data[metric].mean()
                        ax1.plot(mean_values.index, mean_values.values, 
                                marker='s', linestyle='-', 
                                label=f'Avg {metric.replace("_", " ").title()}')
                
                # Find global best position for each iteration
                global_best = results_df.groupby('iteration').apply(
                    lambda x: x.sort_values(by='dbcv', ascending=False).iloc[0] 
                    if 'dbcv' in x.columns and not x['dbcv'].isnull().all() 
                    else x.iloc[0]
                )
                
                # Add annotations showing the best parameters
                for i, row in global_best.iterrows():
                    if i % 5 == 0 or i == global_best.index[-1]:  # Label every 5th iteration
                        if 'min_cluster_size' in row and 'min_samples' in row:
                            ax1.annotate(f"MCS={int(row['min_cluster_size'])}, MS={int(row['min_samples'])}",
                                        xy=(i, row.get('dbcv', 0)),
                                        xytext=(10, 10),
                                        textcoords='offset points',
                                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
                
                ax1.set_xlabel('PSO Iteration', fontsize=12)
                ax1.set_ylabel('Clustering Quality Metrics', fontsize=12)
                ax2.set_ylabel('Percentage (%)', fontsize=12)
                
                # Add legends to both axes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
                
                plt.title('PSO Convergence: Clustering Metrics Over Iterations', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(plots_dir / 'pso_convergence.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved PSO convergence plot to {plots_dir / 'pso_convergence.png'}")
                
            else:
                logger.warning("No iteration data available for convergence plot")
                
        except Exception as e:
            logger.error(f"Error creating convergence plot: {e}")

    def _plot_parameter_performance(self, results_df, plots_dir):
        """Create heatmaps showing how different parameter combinations performed."""
        try:
            # Skip if there are no valid scores to assess
            if 'min_cluster_size' not in results_df.columns or 'min_samples' not in results_df.columns:
                logger.warning("Required parameters not found for parameter performance plot")
                return
            
            # Filter rows with valid scores for each metric
            metrics = ['dbcv', 'silhouette', 'calinski_harabasz', 'davies_bouldin']
            
            for metric in metrics:
                if metric in results_df.columns:
                    # Filter rows with valid metric values
                    valid_df = results_df[results_df[metric].notna()]
                    
                    if len(valid_df) < 10:  # Skip if not enough data
                        continue
                    
                    # Create a pivot table of mean scores for each parameter combination
                    pivot_df = valid_df.pivot_table(
                        index='min_cluster_size',
                        columns='min_samples',
                        values=metric,
                        aggfunc='mean'
                    )
                    
                    # Create the heatmap
                    plt.figure(figsize=(12, 10))
                    
                    # Adjust cmap based on metric (some are better when higher, some when lower)
                    cmap = 'viridis' if metric != 'davies_bouldin' else 'viridis_r'
                    
                    ax = sns.heatmap(pivot_df, annot=True, fmt=".3f", linewidths=.5, cmap=cmap)
                    
                    # Improve readability
                    plt.title(f'Parameter Performance: {metric.replace("_", " ").title()} Score', fontsize=14)
                    plt.xlabel('min_samples', fontsize=12)
                    plt.ylabel('min_cluster_size', fontsize=12)
                    
                    # Adjust colorbar label based on metric
                    if metric == 'davies_bouldin':
                        plt.colorbar(ax.collections[0], label='Score (lower is better)')
                    else:
                        plt.colorbar(ax.collections[0], label='Score (higher is better)')
                    
                    # Save the plot
                    plt.tight_layout()
                    plt.savefig(plots_dir / f'parameter_heatmap_{metric}.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Saved parameter heatmap for {metric} to {plots_dir / f'parameter_heatmap_{metric}.png'}")
            
        except Exception as e:
            logger.error(f"Error creating parameter performance plots: {e}")

    def _plot_top_parameters_comparison(self, results_df, params_df, plots_dir):
        """Create a comparison chart of the top parameter combinations."""
        try:
            # Use the top parameter sets from self.top_parameter_sets or params_df
            if params_df is not None and len(params_df) > 0:
                top_params = params_df.head(5)
            elif self.top_parameter_sets:
                # Convert top_parameter_sets to DataFrame
                top_params = pd.DataFrame(self.top_parameter_sets)
            else:
                logger.warning("No top parameter sets available for comparison plot")
                return
                
            # Prepare the plot
            plt.figure(figsize=(14, 10))
            
            # Get parameter combinations as strings for the x-axis
            if len(top_params) > 0:
                param_labels = [f"MCS={row['min_cluster_size']}, MS={row['min_samples']}" 
                            for _, row in top_params.iterrows()]
                
                # Plot available metrics for each parameter combination
                metrics = ['dbcv', 'silhouette', 'n_clusters', 'noise_ratio', 
                        'temporal_coherence', 'combined_score']
                
                # Filter to only plot metrics available in the dataframe
                metrics = [m for m in metrics if m in top_params.columns]
                
                # Use a secondary y-axis for percentage metrics
                fig, ax1 = plt.subplots(figsize=(14, 10))
                ax2 = ax1.twinx()
                
                bar_width = 0.15
                positions = np.arange(len(param_labels))
                
                # Plot each metric
                for i, metric in enumerate(metrics):
                    if metric in top_params.columns:
                        if metric in ['noise_ratio', 'temporal_coherence', 'n_clusters']:
                            # Plot on right y-axis (percentages and counts)
                            ax2.bar(positions + (i * bar_width), 
                                top_params[metric], 
                                width=bar_width, 
                                label=metric.replace('_', ' ').title())
                        else:
                            # Plot on left y-axis (scores)
                            ax1.bar(positions + (i * bar_width), 
                                top_params[metric], 
                                width=bar_width, 
                                label=metric.replace('_', ' ').title())
                
                # Set labels and title
                ax1.set_xlabel('Parameter Combinations', fontsize=12)
                ax1.set_ylabel('Clustering Quality Scores', fontsize=12)
                ax2.set_ylabel('Count / Percentage (%)', fontsize=12)
                
                ax1.set_xticks(positions + bar_width * (len(metrics) - 1) / 2)
                ax1.set_xticklabels(param_labels, rotation=45, ha='right')
                
                # Add legends to both axes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1, labels1, loc='upper left', fontsize=10)
                ax2.legend(lines2, labels2, loc='upper right', fontsize=10)
                
                plt.title('Comparison of Top 5 Parameter Combinations', fontsize=14)
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(plots_dir / 'top_parameters_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved top parameters comparison plot to {plots_dir / 'top_parameters_comparison.png'}")
                
        except Exception as e:
            logger.error(f"Error creating top parameters comparison plot: {e}")

    def _plot_metric_correlations(self, results_df, plots_dir):
        """Plot correlations between different clustering metrics."""
        try:
            # List of metrics to compare
            metrics = ['dbcv', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 
                    'n_clusters', 'noise_ratio', 'temporal_coherence']
            
            # Get available metrics from the DataFrame
            available_metrics = [m for m in metrics if m in results_df.columns]
            
            if len(available_metrics) < 2:
                logger.warning("Not enough metrics available for correlation plot")
                return
                
            # Filter rows with valid values for all metrics
            valid_rows = results_df.dropna(subset=available_metrics)
            
            if len(valid_rows) < 10:  # Skip if not enough data
                logger.warning("Not enough valid data points for correlation plot")
                return
            
            # Create correlation matrix
            corr_matrix = valid_rows[available_metrics].corr()
            
            # Create the heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create a mask for the upper triangle
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                        cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
            
            plt.title('Correlation Between Clustering Metrics', fontsize=14)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(plots_dir / 'metric_correlations.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved metric correlations plot to {plots_dir / 'metric_correlations.png'}")
            
            # Create pairwise scatter plots for key metrics
            key_metrics = ['dbcv', 'silhouette', 'n_clusters', 'noise_ratio']
            available_key_metrics = [m for m in key_metrics if m in results_df.columns]
            
            if len(available_key_metrics) >= 2:
                # Sample if too many points to avoid overcrowded plot
                plot_df = valid_rows[available_key_metrics]
                if len(plot_df) > 1000:
                    plot_df = plot_df.sample(1000, random_state=42)
                
                # Create pairplot
                g = sns.pairplot(plot_df, diag_kind='kde', height=2.5, 
                                plot_kws={'alpha': 0.5, 's': 30, 'edgecolor': 'w'})
                
                g.fig.suptitle('Pairwise Relationships Between Key Metrics', y=1.02, fontsize=14)
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(plots_dir / 'metric_pairplots.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved metric pairplots to {plots_dir / 'metric_pairplots.png'}")
            
        except Exception as e:
            logger.error(f"Error creating metric correlation plots: {e}")

    def _plot_cluster_distribution(self, results_df, plots_dir):
        """Visualize the distribution of clusters across videos."""
        try:
            if 'n_clusters' not in results_df.columns or 'video_id' not in results_df.columns:
                logger.warning("Required columns not found for cluster distribution plot")
                return
                
            # Group by video and get best clustering (highest DBCV score)
            best_by_video = results_df.loc[results_df.groupby('video_id')['dbcv'].idxmax()] if 'dbcv' in results_df.columns else None
            
            # If can't use DBCV, try silhouette
            if best_by_video is None or len(best_by_video) < 5:
                best_by_video = results_df.loc[results_df.groupby('video_id')['silhouette'].idxmax()] if 'silhouette' in results_df.columns else None
                
            # If still can't get good data, use all results
            if best_by_video is None or len(best_by_video) < 5:
                best_by_video = results_df
                
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(18, 14))
            gs = GridSpec(2, 2, figure=fig)
            
            # 1. Histogram of number of clusters
            ax1 = fig.add_subplot(gs[0, 0])
            sns.histplot(best_by_video['n_clusters'], kde=True, ax=ax1)
            ax1.set_title('Distribution of Number of Clusters', fontsize=12)
            ax1.set_xlabel('Number of Clusters', fontsize=10)
            ax1.set_ylabel('Count of Videos', fontsize=10)
            
            # 2. Scatter plot of n_clusters vs noise_ratio
            if 'noise_ratio' in best_by_video.columns:
                ax2 = fig.add_subplot(gs[0, 1])
                scatter = ax2.scatter(best_by_video['n_clusters'], 
                                    best_by_video['noise_ratio'],
                                    alpha=0.6)
                ax2.set_title('Clusters vs. Noise Ratio', fontsize=12)
                ax2.set_xlabel('Number of Clusters', fontsize=10)
                ax2.set_ylabel('Noise Ratio (%)', fontsize=10)
                ax2.grid(True, alpha=0.3)
            
            # 3. Boxplot of best parameter sets for n_clusters
            ax3 = fig.add_subplot(gs[1, 0])
            if 'min_cluster_size' in best_by_video.columns and 'min_samples' in best_by_video.columns:
                # Create parameter combination labels
                best_by_video['param_combo'] = best_by_video.apply(
                    lambda x: f"MCS={int(x['min_cluster_size'])}, MS={int(x['min_samples'])}", axis=1)
                
                # Get top 5 most frequent parameter combinations
                top_combos = best_by_video['param_combo'].value_counts().head(8).index.tolist()
                combo_data = best_by_video[best_by_video['param_combo'].isin(top_combos)]
                
                # Create boxplot
                sns.boxplot(x='param_combo', y='n_clusters', data=combo_data, ax=ax3)
                ax3.set_title('Number of Clusters by Parameter Combination', fontsize=12)
                ax3.set_xlabel('Parameter Combination', fontsize=10)
                ax3.set_ylabel('Number of Clusters', fontsize=10)
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # 4. Temporal coherence vs. n_clusters
            if 'temporal_coherence' in best_by_video.columns:
                ax4 = fig.add_subplot(gs[1, 1])
                scatter = ax4.scatter(best_by_video['n_clusters'], 
                                    best_by_video['temporal_coherence'],
                                    alpha=0.6)
                ax4.set_title('Clusters vs. Temporal Coherence', fontsize=12)
                ax4.set_xlabel('Number of Clusters', fontsize=10)
                ax4.set_ylabel('Temporal Coherence (%)', fontsize=10)
                ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Cluster Distribution Analysis Across Videos', fontsize=16)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(plots_dir / 'cluster_distribution_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved cluster distribution analysis to {plots_dir / 'cluster_distribution_analysis.png'}")
            
        except Exception as e:
            logger.error(f"Error creating cluster distribution plots: {e}")

    def _plot_parameter_sensitivity(self, results_df, plots_dir):
        """Create a sensitivity analysis plot for the parameters."""
        try:
            if 'min_cluster_size' not in results_df.columns or 'min_samples' not in results_df.columns:
                logger.warning("Required parameters not found for sensitivity analysis")
                return
                
            # Set up the figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            
            # Choose metrics to analyze
            metrics = ['dbcv', 'silhouette', 'n_clusters', 'noise_ratio'] 
            
            for i, metric in enumerate(metrics):
                if metric not in results_df.columns or i >= len(axes):
                    continue
                    
                # Filter for valid metric values
                valid_df = results_df[results_df[metric].notna()]
                
                if len(valid_df) < 10:  # Skip if not enough data
                    continue
                    
                # Group by min_cluster_size and calculate mean and std
                mcs_grouped = valid_df.groupby('min_cluster_size')[metric].agg(['mean', 'std']).reset_index()
                mcs_grouped['std'] = mcs_grouped['std'].fillna(0)
                
                # Group by min_samples and calculate mean and std
                ms_grouped = valid_df.groupby('min_samples')[metric].agg(['mean', 'std']).reset_index()
                ms_grouped['std'] = ms_grouped['std'].fillna(0)
                
                # Plot for min_cluster_size
                axes[i].errorbar(mcs_grouped['min_cluster_size'], mcs_grouped['mean'], 
                            yerr=mcs_grouped['std'], fmt='o-', capsize=5, 
                            label='By min_cluster_size')
                
                # Plot for min_samples  
                axes[i].errorbar(ms_grouped['min_samples'], ms_grouped['mean'], 
                            yerr=ms_grouped['std'], fmt='s--', capsize=5, 
                            label='By min_samples')
                
                axes[i].set_title(f'Sensitivity of {metric.replace("_", " ").title()}', fontsize=12)
                axes[i].set_xlabel('Parameter Value', fontsize=10)
                axes[i].set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=10)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # Add a horizontal line for reference if appropriate
                if metric == 'silhouette':
                    axes[i].axhline(y=0, color='r', linestyle=':')
                elif metric == 'temporal_coherence':
                    axes[i].axhline(y=25, color='r', linestyle=':')
            
            plt.suptitle('Parameter Sensitivity Analysis', fontsize=16)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(plots_dir / 'parameter_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved parameter sensitivity analysis to {plots_dir / 'parameter_sensitivity_analysis.png'}")
            
        except Exception as e:
            logger.error(f"Error creating parameter sensitivity plots: {e}")

    def _create_summary_dashboard(self, results_df, params_df, plots_dir):
        """Create a summary dashboard with key insights from the PSO optimization."""
        try:
            # Calculate key statistics
            total_videos = results_df['video_id'].nunique() if 'video_id' in results_df.columns else 0
            total_iterations = results_df['iteration'].max() + 1 if 'iteration' in results_df.columns else 0
            total_evaluations = len(results_df)
            
            # Get best parameters
            if params_df is not None and len(params_df) > 0:
                best_params = params_df.iloc[0]
            elif self.best_global_params:
                best_params = pd.Series(self.best_global_params)
            else:
                best_params = None
            
            # Create summary figure
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.95, 'HDBSCAN PSO Optimization Summary', 
                    fontsize=18, ha='center', weight='bold')
            
            # Experiment details
            ax.text(0.05, 0.85, 'Experiment Overview:', fontsize=14, weight='bold')
            ax.text(0.05, 0.81, f'Videos Analyzed: {total_videos}', fontsize=12)
            ax.text(0.05, 0.78, f'PSO Iterations: {total_iterations}', fontsize=12)
            ax.text(0.05, 0.75, f'Total Evaluations: {total_evaluations}', fontsize=12)
            
            # Best parameters found
            if best_params is not None:
                ax.text(0.05, 0.70, 'Optimal Parameters:', fontsize=14, weight='bold')
                y_pos = 0.66
                
                for param, value in best_params.items():
                    if param in ['min_cluster_size', 'min_samples', 'cluster_selection_method']:
                        ax.text(0.05, y_pos, f'{param}: {value}', fontsize=12)
                        y_pos -= 0.03
                
                # Performance metrics for best params
                if 'dbcv' in best_params:
                    ax.text(0.05, y_pos-0.03, 'Performance Metrics:', fontsize=14, weight='bold')
                    y_pos -= 0.07
                    
                    metrics = ['dbcv', 'silhouette', 'n_clusters', 'noise_ratio', 'temporal_coherence']
                    for metric in metrics:
                        if metric in best_params:
                            ax.text(0.05, y_pos, f'{metric.replace("_", " ").title()}: {best_params[metric]:.4f}', fontsize=12)
                            y_pos -= 0.03
            
            # Key insights
            ax.text(0.05, y_pos-0.05, 'Key Insights:', fontsize=14, weight='bold')
            
            # Calculate some insights
            insights = []
            
            # Check if we have enough data for meaningful insights
            if 'n_clusters' in results_df.columns:
                avg_clusters = results_df['n_clusters'].mean()
                insights.append(f'Average clusters per video: {avg_clusters:.2f}')
                
            if 'noise_ratio' in results_df.columns:
                avg_noise = results_df['noise_ratio'].mean()
                insights.append(f'Average noise ratio: {avg_noise:.2f}%')
                
            if 'min_cluster_size' in results_df.columns and 'dbcv' in results_df.columns:
                # Find optimal min_cluster_size range
                mcs_perf = results_df.groupby('min_cluster_size')['dbcv'].mean().sort_values(ascending=False)
                if len(mcs_perf) > 0:
                    top_mcs = mcs_perf.index[0]
                    insights.append(f'Optimal min_cluster_size range: {top_mcs}')
                
            if 'min_samples' in results_df.columns and 'dbcv' in results_df.columns:
                # Find optimal min_samples range
                ms_perf = results_df.groupby('min_samples')['dbcv'].mean().sort_values(ascending=False)
                if len(ms_perf) > 0:
                    top_ms = ms_perf.index[0]
                    insights.append(f'Optimal min_samples range: {top_ms}')
            
            # Add the insights to the plot
            y_pos -= 0.03
            for insight in insights:
                ax.text(0.05, y_pos, f'• {insight}', fontsize=12)
                y_pos -= 0.03
            
            # Recommendations
            ax.text(0.05, y_pos-0.05, 'Recommendations:', fontsize=14, weight='bold')
            y_pos -= 0.08
            
            recommendations = [
                "Use the top parameter set for general video scene clustering",
                "For videos with many scene changes, consider using higher min_samples",
                "For videos with subtle scene changes, consider lower min_cluster_size",
                "Review the other visualization plots for detailed parameter effects"
            ]
            
            for rec in recommendations:
                ax.text(0.05, y_pos, f'• {rec}', fontsize=12)
                y_pos -= 0.03
            
            # Add a decorative border
            border = plt.Rectangle((0.02, 0.02), 0.96, 0.96, 
                                fill=False, edgecolor='gray', linewidth=1)
            ax.add_patch(border)
            
            plt.savefig(plots_dir / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved summary dashboard to {plots_dir / 'summary_dashboard.png'}")
            
        except Exception as e:
            logger.error(f"Error creating summary dashboard: {e}")
            
    def analyze_existing_results(self, results_file=None):
        """
        Analyze existing results without running PSO again.
        
        Args:
            results_file: Path to an existing results CSV file (defaults to pso_all_results.csv in output_dir)
        """
        if results_file is None:
            results_file = self.output_dir / "pso_all_results.csv"
        
        logger.info(f"Analyzing existing results from {results_file}")
        
        if not Path(results_file).exists():
            logger.error(f"Results file {results_file} not found")
            return
        
        # Load existing results
        self.all_results = pd.read_csv(results_file).to_dict('records')
        
        # Run analysis and visualization
        self._save_and_analyze_results()
        self.generate_summary_plots()
        
        logger.info("Analysis and visualization of existing results completed")
    
def main():
    parser = argparse.ArgumentParser(description="HDBSCAN Parameter Optimization with PSO for Large Datasets")
    parser.add_argument("--embeddings_dir", type=str, default="swin_embeddings",
                        help="Directory containing embedding .npy files")
    parser.add_argument("--frames_dir", type=str, default="/Users/akshat/Developer/Vid_Attention/data_extraction/data/QuerYD/processed_frames",
                        help="Directory containing video frames (for storing cluster frames)")
    parser.add_argument("--output_dir", type=str, default="hdbscan_pso_results",
                        help="Directory to save results")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of videos to sample (None = all)")
    parser.add_argument("--n_particles", type=int, default=25,
                        help="Number of particles in the swarm")
    parser.add_argument("--n_iterations", type=int, default=30,
                        help="Number of PSO iterations")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Save checkpoints every N videos processed")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze existing results without running PSO again")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to existing results file for analysis")
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = HDBSCANOptimizer(
        embeddings_dir=args.embeddings_dir,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Set PSO hyperparameters
    optimizer.n_particles = args.n_particles
    optimizer.n_iterations = args.n_iterations
    
    if args.analyze_only:
        # Only analyze existing results
        optimizer.analyze_existing_results(args.results_file)
    else:
        # Run PSO
        optimizer.run_pso(sample_size=args.sample_size, resume=args.resume)
    
if __name__ == "__main__":
    main()