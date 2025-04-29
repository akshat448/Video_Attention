# import argparse
# import os
# import logging
# import torch
# import torch.nn as nn
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# import re # For parsing frame indices from filenames

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class CrossAttentionAggregator(nn.Module):
#     """
#     Aggregates a set of frame embeddings for a cluster into a single embedding
#     using a cross-attention mechanism.
#     """
#     def __init__(self, embed_dim, num_heads=1):
#         """
#         Args:
#             embed_dim (int): The dimensionality of the input frame embeddings.
#             num_heads (int): Number of attention heads. For a single query attending
#                              to a set of items, 1 head is often sufficient, but
#                              multiple heads can capture different aspects.
#         """
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads

#         # Learnable query vector that attends to all frame embeddings in the cluster
#         # Shape: (1, 1, embed_dim) - SeqLen=1, Batch=1, EmbedDim
#         self.cluster_query = nn.Parameter(torch.randn(1, 1, embed_dim))

#         # Multihead attention layer
#         # We use batch_first=False because the sequence length is the number of frames in the cluster
#         # and the batch size is effectively 1 (processing one cluster at a time).
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)

#         # Optional: A linear layer after attention
#         # self.fc = nn.Linear(embed_dim, embed_dim) # Identity mapping by default

#     def forward(self, frame_embeddings):
#         """
#         Args:
#             frame_embeddings (torch.Tensor): Tensor of embeddings for frames in a cluster.
#                                              Shape: (N_cluster, embed_dim)

#         Returns:
#             torch.Tensor: A single aggregated embedding for the cluster. Shape: (embed_dim,)
#         """
#         N_cluster = frame_embeddings.size(0)

#         if N_cluster == 0:
#             logger.warning("Attempted to aggregate an empty cluster. Returning zero vector.")
#             return torch.zeros(self.embed_dim, device=frame_embeddings.device)
#         elif N_cluster == 1:
#             logger.debug("Cluster has only one frame. Returning its embedding directly.")
#             return frame_embeddings.squeeze(0) # Shape (1, embed_dim) -> (embed_dim,)

#         # Reshape frame embeddings for MultiheadAttention: (SeqLen, Batch, EmbedDim)
#         # Here, SeqLen is N_cluster, Batch is 1.
#         frame_embeddings = frame_embeddings.unsqueeze(1) # Shape: (N_cluster, 1, embed_dim)

#         # Expand the query to match the batch size (which is 1) - although MultiheadAttention
#         # handles query batching, explicit unsqueeze makes roles clearer.
#         query = self.cluster_query # Shape: (1, 1, embed_dim)

#         # Apply multihead attention
#         # query shape: (1, 1, embed_dim)
#         # key, value shapes: (N_cluster, 1, embed_dim)
#         # attn_output shape: (1, 1, embed_dim) - output for the query attending over sequence
#         attn_output, attn_weights = self.multihead_attn(query=query, key=frame_embeddings, value=frame_embeddings)

#         # The output for the single query is the aggregated embedding.
#         # Squeeze to remove SeqLen=1 and Batch=1 dimensions.
#         aggregated_embedding = attn_output.squeeze(0).squeeze(0) # Shape: (embed_dim,)

#         # Optional: Apply a linear layer
#         # aggregated_embedding = self.fc(aggregated_embedding)

#         return aggregated_embedding

# def parse_frame_index_from_filename(filename: Path) -> int:
#     """
#     Parses the frame index from a filename like 'frame_00042.png'.
#     """
#     match = re.search(r'frame_(\d+)\.png$', filename.name)
#     if match:
#         return int(match.group(1))
#     else:
#         raise ValueError(f"Could not parse frame index from filename: {filename}")


# def generate_cluster_embeddings(
#     clustered_frames_dir: Path,
#     original_embeddings_dir: Path,
#     output_cluster_embeddings_dir: Path,
#     embed_dim: int,
#     num_attention_heads: int = 1,
#     device: torch.device = torch.device("mps")
# ):
#     """
#     Generates a single aggregated embedding for each cluster within each video
#     using a cross-attention mechanism on the original frame embeddings.

#     Args:
#         clustered_frames_dir (Path): Directory containing video/cluster subfolders
#                                      from the clustering script (e.g., 'clustered_frames').
#                                      Structure: clustered_frames_dir / video_id / cluster_X / frame_YYYYY.png
#         original_embeddings_dir (Path): Directory containing the original per-frame
#                                         embedding files (.npy) from the first script.
#                                         Structure: original_embeddings_dir / video_id_embeddings.npy
#         output_cluster_embeddings_dir (Path): Directory to save the generated cluster
#                                              embeddings (.npy).
#                                              Structure created: output_cluster_embeddings_dir / video_id / cluster_X_embedding.npy
#         embed_dim (int): The dimensionality of the frame embeddings.
#         num_attention_heads (int): Number of heads for the attention mechanism.
#         device (torch.device): Device to perform computations on ('cuda' or 'cpu').
#     """
#     logger.info(f"Starting cluster embedding generation.")
#     logger.info(f"Clustered frames directory: {clustered_frames_dir}")
#     logger.info(f"Original embeddings directory: {original_embeddings_dir}")
#     logger.info(f"Output cluster embeddings directory: {output_cluster_embeddings_dir}")
#     logger.info(f"Embedding dimension: {embed_dim}")
#     logger.info(f"Attention heads: {num_attention_heads}")
#     logger.info(f"Using device: {device}")

#     output_cluster_embeddings_dir.mkdir(parents=True, exist_ok=True)

#     # --- Initialize the Aggregation Model ---
#     try:
#         aggregator = CrossAttentionAggregator(embed_dim=embed_dim, num_heads=num_attention_heads).to(device)
#         aggregator.eval() # Set model to evaluation mode
#         # Save the model state dict for visualization purposes
#         model_state_path = output_cluster_embeddings_dir / "aggregator_state_dict.pth"
#         torch.save(aggregator.state_dict(), model_state_path)
#         logger.info(f"Aggregator model state dict saved to {model_state_path}")
#         logger.info("CrossAttentionAggregator initialized.")
#     except Exception as e:
#         logger.error(f"Failed to initialize or save CrossAttentionAggregator: {e}")
#         return

#     # Find all video subdirectories in the clustered frames output
#     video_dirs = [d for d in clustered_frames_dir.iterdir() if d.is_dir()]

#     if not video_dirs:
#         logger.warning(f"No video subdirectories found in {clustered_frames_dir}. Exiting.")
#         return

#     logger.info(f"Found {len(video_dirs)} videos to process.")

#     for video_dir in tqdm(video_dirs, desc="Processing Videos"):
#         video_id = video_dir.name
#         logger.info(f"Processing video: {video_id}")

#         output_video_embedding_dir = output_cluster_embeddings_dir / video_id
#         output_video_embedding_dir.mkdir(parents=True, exist_ok=True)

#         # --- Load Original Frame Embeddings for this Video ---
#         original_embedding_path = original_embeddings_dir / f"{video_id}_embeddings.npy"
#         if not original_embedding_path.exists():
#             logger.error(f"Original embedding file not found for {video_id}: {original_embedding_path}. Skipping.")
#             continue

#         try:
#             original_embeddings_np = np.load(original_embedding_path)
#             original_embeddings_torch = torch.from_numpy(original_embeddings_np).float().to(device)
#             logger.info(f"Loaded original frame embeddings for {video_id}. Shape: {original_embeddings_torch.shape}")

#             # Verify embedding dimension
#             if original_embeddings_torch.shape[-1] != embed_dim:
#                  logger.error(f"Mismatch in embedding dimension for {video_id}. Expected {embed_dim}, got {original_embeddings_torch.shape[-1]}. Skipping.")
#                  continue

#         except Exception as e:
#             logger.error(f"Error loading original embeddings for {video_id} from {original_embedding_path}: {e}. Skipping.")
#             continue


#         # Find all cluster subdirectories for this video
#         cluster_dirs = [d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('cluster_')]

#         if not cluster_dirs:
#             logger.warning(f"No cluster subdirectories found for {video_id} in {video_dir}. Skipping.")
#             continue

#         logger.info(f"Found {len(cluster_dirs)} clusters for {video_id}.")

#         for cluster_dir in tqdm(cluster_dirs, desc=f"Aggregating Clusters for {video_id}", leave=False):
#             cluster_name = cluster_dir.name # e.g., 'cluster_0', 'cluster_-1'
#             cluster_label_str = cluster_name.replace('cluster_', '')

#             # Optionally skip noise cluster (-1) if you don't want an embedding for it
#             if cluster_label_str == '-1':
#                  logger.info(f"Skipping noise cluster ({cluster_name}) for {video_id}.")
#                  continue

#             output_cluster_embedding_path = output_video_embedding_dir / f"{cluster_name}_embedding.npy"

#             if output_cluster_embedding_path.exists():
#                 logger.debug(f"Cluster embedding already exists for {video_id}/{cluster_name}, skipping.")
#                 continue

#             # Get the list of frame files in this cluster
#             frame_files = sorted(list(cluster_dir.glob("frame_*.png")))

#             if not frame_files:
#                 logger.warning(f"Cluster folder {video_id}/{cluster_name} is empty. Skipping.")
#                 continue

#             # Extract original frame indices from filenames
#             try:
#                 frame_indices = [parse_frame_index_from_filename(f) for f in frame_files]
#                 logger.debug(f"Found {len(frame_indices)} frames in cluster {cluster_name} for {video_id}.")
#                 # logger.debug(f"Frame indices: {frame_indices[:10]}...") # Log first few indices
#             except ValueError as e:
#                 logger.error(f"Error parsing frame index in {video_id}/{cluster_name}: {e}. Skipping cluster.")
#                 continue


#             # Check if indices are within bounds of original embeddings
#             max_index = original_embeddings_torch.shape[0] - 1
#             if any(idx > max_index for idx in frame_indices):
#                  logger.error(f"Frame index out of bounds in {video_id}/{cluster_name}. Max index in embeddings is {max_index}. Skipping cluster.")
#                  continue

#             # Select the original frame embeddings for this cluster
#             cluster_frame_embeddings = original_embeddings_torch[frame_indices]
#             logger.debug(f"Selected {cluster_frame_embeddings.shape[0]} frame embeddings for aggregation.")


#             # --- Generate Cluster Embedding using Aggregator ---
#             with torch.no_grad(): # No gradients needed for inference
#                 try:
#                     aggregated_embedding_torch = aggregator(cluster_frame_embeddings)
#                     logger.debug(f"Aggregated embedding shape: {aggregated_embedding_torch.shape}")

#                     # Convert back to NumPy and save
#                     aggregated_embedding_np = aggregated_embedding_torch.cpu().numpy()
#                     np.save(output_cluster_embedding_path, aggregated_embedding_np)
#                     logger.info(f"Saved cluster embedding for {video_id}/{cluster_name} to {output_cluster_embedding_path}")

#                 except Exception as e:
#                      logger.error(f"Error during aggregation for {video_id}/{cluster_name}: {e}. Skipping cluster.")
#                      # Optionally remove partially created file if save failed
#                      if output_cluster_embedding_path.exists():
#                          output_cluster_embedding_path.unlink()


#     logger.info("=== Cluster Embedding Generation Complete ===")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate single embeddings per cluster using cross-attention.")
#     parser.add_argument("--clustered_frames_dir", type=Path, default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/clusters",
#                         help="Directory containing video/cluster subfolders from the clustering script.")
#     parser.add_argument("--original_embeddings_dir", type=Path, default="swin_embeddings",
#                         help="Directory containing the original per-frame embedding files (.npy).")
#     parser.add_argument("--output_cluster_embeddings_dir", type=Path, default="cluster_embeddings",
#                         help="Directory to save the generated cluster embeddings.")
#     parser.add_argument("--embed_dim", type=int, default=768,
#                         help="The dimensionality of the frame embeddings generated by the first script.")
#     parser.add_argument("--num_attention_heads", type=int, default=4,
#                         help="Number of attention heads for the aggregation mechanism.")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps",
#                         help="Device to use ('cuda' or 'cpu').")

#     args = parser.parse_args()

#     device = torch.device(args.device)

#     generate_cluster_embeddings(
#         clustered_frames_dir=args.clustered_frames_dir,
#         original_embeddings_dir=args.original_embeddings_dir,
#         output_cluster_embeddings_dir=args.output_cluster_embeddings_dir,
#         embed_dim=args.embed_dim,
#         num_attention_heads=args.num_attention_heads,
#         device=device
#     )


import argparse
import os
import logging
import torch
import torch.nn as nn
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
import re # For parsing frame indices from filenames

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrossAttentionAggregator(nn.Module):
    """
    Aggregates a set of frame embeddings for a cluster into a single embedding
    using a cross-attention mechanism.
    """
    def __init__(self, embed_dim, num_heads=1):
        """
        Args:
            embed_dim (int): The dimensionality of the input frame embeddings.
            num_heads (int): Number of attention heads. For a single query attending
                             to a set of items, 1 head is often sufficient, but
                             multiple heads can capture different aspects.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Learnable query vector that attends to all frame embeddings in the cluster
        # Shape: (1, 1, embed_dim) - SeqLen=1, Batch=1, EmbedDim
        self.cluster_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Multihead attention layer
        # We use batch_first=False because the sequence length is the number of frames in the cluster
        # and the batch size is effectively 1 (processing one cluster at a time).
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)

        # Optional: A linear layer after attention
        # self.fc = nn.Linear(embed_dim, embed_dim) # Identity mapping by default

    def forward(self, frame_embeddings):
        """
        Args:
            frame_embeddings (torch.Tensor): Tensor of embeddings for frames in a cluster.
                                             Shape: (N_cluster, embed_dim)

        Returns:
            torch.Tensor: A single aggregated embedding for the cluster. Shape: (embed_dim,)
        """
        N_cluster = frame_embeddings.size(0)

        if N_cluster == 0:
            logger.warning("Attempted to aggregate an empty cluster. Returning zero vector.")
            return torch.zeros(self.embed_dim, device=frame_embeddings.device)
        elif N_cluster == 1:
            logger.debug("Cluster has only one frame. Returning its embedding directly.")
            return frame_embeddings.squeeze(0) # Shape (1, embed_dim) -> (embed_dim,)

        # Reshape frame embeddings for MultiheadAttention: (SeqLen, Batch, EmbedDim)
        # Here, SeqLen is N_cluster, Batch is 1.
        frame_embeddings = frame_embeddings.unsqueeze(1) # Shape: (N_cluster, 1, embed_dim)

        # Expand the query to match the batch size (which is 1) - although MultiheadAttention
        # handles query batching, explicit unsqueeze makes roles clearer.
        query = self.cluster_query # Shape: (1, 1, embed_dim)

        # Apply multihead attention
        # query shape: (1, 1, embed_dim)
        # key, value shapes: (N_cluster, 1, embed_dim)
        # attn_output shape: (1, 1, embed_dim) - output for the query attending over sequence
        attn_output, attn_weights = self.multihead_attn(query=query, key=frame_embeddings, value=frame_embeddings)

        # The output for the single query is the aggregated embedding.
        # Squeeze to remove SeqLen=1 and Batch=1 dimensions.
        aggregated_embedding = attn_output.squeeze(0).squeeze(0) # Shape: (embed_dim,)

        # Optional: Apply a linear layer
        # aggregated_embedding = self.fc(aggregated_embedding)

        return aggregated_embedding, attn_weights.squeeze(0).squeeze(0)  # Return attention weights for analysis

def parse_frame_index_from_filename(filename: Path) -> int:
    """
    Parses the frame index from a filename like 'frame_00042.png'.
    """
    match = re.search(r'frame_(\d+)\.png$', filename.name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not parse frame index from filename: {filename}")

def load_frame_timestamps(timestamps_dir, video_id):
    """
    Load timestamps for frames in a video.
    Returns a numpy array of timestamps.
    """
    timestamps_dir = Path(timestamps_dir)  # Ensure it's a Path object
    
    # Clean video ID format for consistent handling
    clean_id = video_id.replace('video-', '') if video_id.startswith('video-') else video_id
    prefixed_id = f"video-{clean_id}"
    
    # List of possible video directories
    video_dirs = [
        timestamps_dir / video_id,
        timestamps_dir / clean_id,
        timestamps_dir / prefixed_id
    ]
    
    # Check each video directory
    for video_dir in video_dirs:
        if video_dir.exists():
            logger.info(f"Found video dir: {video_dir}")
            
            # Try common timestamp files
            timestamp_files = [
                video_dir / f"{video_id}_timestamps.npy",
                video_dir / f"{clean_id}_timestamps.npy",
                video_dir / "timestamps.npy",
                video_dir / f"{video_id}_frames.h5",
                video_dir / f"{clean_id}_frames.h5",
                video_dir / "frames.h5"
            ]
            
            # Also check H5 files directly
            h5_files = list(video_dir.glob("*.h5"))
            for h5_file in h5_files:
                if h5_file not in timestamp_files:
                    timestamp_files.append(h5_file)
            
            for file_path in timestamp_files:
                if file_path.exists():
                    if file_path.suffix == '.npy':
                        try:
                            timestamps = np.load(file_path)
                            logger.info(f"Loaded {len(timestamps)} timestamps from {file_path}")
                            return timestamps
                        except Exception as e:
                            logger.error(f"Error loading {file_path}: {e}")
                    elif file_path.suffix == '.h5':
                        try:
                            with h5py.File(file_path, 'r') as f:
                                # Show keys in H5 file
                                keys = list(f.keys())
                                logger.info(f"H5 file keys: {keys}")
                                
                                if 'timestamps' in f:
                                    timestamps = f['timestamps'][()]
                                    logger.info(f"Loaded {len(timestamps)} timestamps from {file_path}")
                                    return timestamps
                                elif 'frames' in f:
                                    frames = f['frames']
                                    num_frames = len(frames)
                                    logger.info(f"Generating synthetic timestamps for {num_frames} frames from H5")
                                    return np.arange(num_frames, dtype=float)
                        except Exception as e:
                            logger.error(f"Error loading {file_path}: {e}")
            
            # Check for jpg directory
            jpg_dir = video_dir / "jpg"
            if jpg_dir.exists():
                frame_files = list(jpg_dir.glob("*.jpg"))
                if frame_files:
                    logger.info(f"Generating synthetic timestamps from {len(frame_files)} JPG files")
                    return np.arange(len(frame_files), dtype=float)
    
    # Final fallback - generate from number of frames
    logger.warning(f"No timestamps found for {video_id}, generating synthetic ones")
    return None

def generate_cluster_embeddings(
    clustered_frames_dir: Path,
    original_embeddings_dir: Path,
    timestamps_dir: Path,
    output_cluster_embeddings_dir: Path,
    embed_dim: int,
    num_attention_heads: int = 1,
    device: torch.device = torch.device("mps")
):
    """
    Generates a single aggregated embedding for each cluster within each video
    using a cross-attention mechanism on the original frame embeddings.
    Also maintains timestamps for each cluster to enable time-based caption mapping.

    Args:
        clustered_frames_dir (Path): Directory containing video/cluster subfolders.
        original_embeddings_dir (Path): Directory containing the original per-frame embeddings.
        timestamps_dir (Path): Directory containing frame timestamps.
        output_cluster_embeddings_dir (Path): Directory to save the generated cluster embeddings.
        embed_dim (int): The dimensionality of the frame embeddings.
        num_attention_heads (int): Number of heads for the attention mechanism.
        device (torch.device): Device to perform computations on.
    """
    logger.info(f"Starting cluster embedding generation with timestamp tracking.")
    logger.info(f"Clustered frames directory: {clustered_frames_dir}")
    logger.info(f"Original embeddings directory: {original_embeddings_dir}")
    logger.info(f"Timestamps directory: {timestamps_dir}")
    logger.info(f"Output cluster embeddings directory: {output_cluster_embeddings_dir}")
    logger.info(f"Embedding dimension: {embed_dim}")
    logger.info(f"Attention heads: {num_attention_heads}")
    logger.info(f"Using device: {device}")

    output_cluster_embeddings_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize the Aggregation Model ---
    try:
        aggregator = CrossAttentionAggregator(embed_dim=embed_dim, num_heads=num_attention_heads).to(device)
        aggregator.eval() # Set model to evaluation mode
        # Save the model state dict for visualization purposes
        model_state_path = output_cluster_embeddings_dir / "aggregator_state_dict.pth"
        torch.save(aggregator.state_dict(), model_state_path)
        logger.info(f"Aggregator model state dict saved to {model_state_path}")
    except Exception as e:
        logger.error(f"Failed to initialize or save CrossAttentionAggregator: {e}")
        return

    # Find all video subdirectories in the clustered frames output
    video_dirs = [d for d in clustered_frames_dir.iterdir() if d.is_dir()]

    if not video_dirs:
        logger.warning(f"No video subdirectories found in {clustered_frames_dir}. Exiting.")
        return

    logger.info(f"Found {len(video_dirs)} videos to process.")

    for video_dir in tqdm(video_dirs, desc="Processing Videos"):
        video_id = video_dir.name
        logger.info(f"Processing video: {video_id}")

        output_video_embedding_dir = output_cluster_embeddings_dir / video_id
        output_video_embedding_dir.mkdir(parents=True, exist_ok=True)

        # --- Load Original Frame Embeddings for this Video ---
        original_embedding_path = original_embeddings_dir / f"{video_id}_embeddings.npy"
        if not original_embedding_path.exists():
            logger.error(f"Original embedding file not found for {video_id}: {original_embedding_path}. Skipping.")
            continue

        try:
            original_embeddings_np = np.load(original_embedding_path)
            original_embeddings_torch = torch.from_numpy(original_embeddings_np).float().to(device)
            logger.info(f"Loaded original frame embeddings for {video_id}. Shape: {original_embeddings_torch.shape}")

            # Verify embedding dimension
            if original_embeddings_torch.shape[-1] != embed_dim:
                 logger.error(f"Mismatch in embedding dimension for {video_id}. Expected {embed_dim}, got {original_embeddings_torch.shape[-1]}. Skipping.")
                 continue

        except Exception as e:
            logger.error(f"Error loading original embeddings for {video_id} from {original_embedding_path}: {e}. Skipping.")
            continue

        # --- Load Timestamps for this Video ---
        timestamps = load_frame_timestamps(timestamps_dir, video_id)
        if timestamps is None:
            # Generate synthetic timestamps if needed
            max_frame_idx = original_embeddings_torch.shape[0] - 1
            logger.warning(f"Generating synthetic timestamps for {video_id} with {max_frame_idx+1} frames")
            timestamps = np.arange(max_frame_idx+1, dtype=float)
        
        if len(timestamps) != original_embeddings_torch.shape[0]:
            logger.error(f"Mismatch in timestamps length ({len(timestamps)}) and embeddings count ({original_embeddings_torch.shape[0]}) for {video_id}. Skipping.")
            continue

        # Find all cluster subdirectories for this video
        cluster_dirs = [d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('cluster_')]

        if not cluster_dirs:
            logger.warning(f"No cluster subdirectories found for {video_id} in {video_dir}. Skipping.")
            continue

        logger.info(f"Found {len(cluster_dirs)} clusters for {video_id}.")

        for cluster_dir in tqdm(cluster_dirs, desc=f"Aggregating Clusters for {video_id}", leave=False):
            cluster_name = cluster_dir.name # e.g., 'cluster_0', 'cluster_-1'
            cluster_label_str = cluster_name.replace('cluster_', '')

            # Optionally skip noise cluster (-1) if you don't want an embedding for it
            if cluster_label_str == '-1':
                 logger.info(f"Skipping noise cluster ({cluster_name}) for {video_id}.")
                 continue

            output_cluster_embedding_path = output_video_embedding_dir / f"{cluster_name}_embedding.npy"
            output_cluster_timestamps_path = output_video_embedding_dir / f"{cluster_name}_timestamps.npy"
            output_cluster_indices_path = output_video_embedding_dir / f"{cluster_name}_indices.npy"

            if output_cluster_embedding_path.exists() and output_cluster_timestamps_path.exists():
                logger.debug(f"Cluster embedding and timestamps already exist for {video_id}/{cluster_name}, skipping.")
                continue

            # Get the list of frame files in this cluster
            frame_files = sorted(list(cluster_dir.glob("frame_*.png")))

            if not frame_files:
                logger.warning(f"Cluster folder {video_id}/{cluster_name} is empty. Skipping.")
                continue

            # Extract original frame indices from filenames
            try:
                frame_indices = [parse_frame_index_from_filename(f) for f in frame_files]
                logger.debug(f"Found {len(frame_indices)} frames in cluster {cluster_name} for {video_id}.")
            except ValueError as e:
                logger.error(f"Error parsing frame index in {video_id}/{cluster_name}: {e}. Skipping cluster.")
                continue

            # Check if indices are within bounds of original embeddings
            max_index = original_embeddings_torch.shape[0] - 1
            if any(idx > max_index for idx in frame_indices):
                 logger.error(f"Frame index out of bounds in {video_id}/{cluster_name}. Max index in embeddings is {max_index}. Skipping cluster.")
                 continue

            # Select the original frame embeddings for this cluster
            cluster_frame_embeddings = original_embeddings_torch[frame_indices]
            
            # Select timestamps for this cluster
            cluster_timestamps = timestamps[frame_indices]
            
            # Get min and max timestamp for the cluster (for caption mapping)
            cluster_min_time = float(np.min(cluster_timestamps))
            cluster_max_time = float(np.max(cluster_timestamps))
            cluster_timespan = np.array([cluster_min_time, cluster_max_time])
            
            logger.debug(f"Cluster {cluster_name} timespan: {cluster_min_time:.2f}s - {cluster_max_time:.2f}s")

            # --- Generate Cluster Embedding using Aggregator ---
            with torch.no_grad(): # No gradients needed for inference
                try:
                    aggregated_embedding_torch, attention_weights = aggregator(cluster_frame_embeddings)
                    
                    # Convert back to NumPy and save
                    aggregated_embedding_np = aggregated_embedding_torch.cpu().numpy()
                    np.save(output_cluster_embedding_path, aggregated_embedding_np)
                    
                    # Save the frame indices
                    np.save(output_cluster_indices_path, np.array(frame_indices))
                    
                    # Save the cluster timespan
                    np.save(output_cluster_timestamps_path, cluster_timespan)
                    
                    logger.info(f"Saved cluster data for {video_id}/{cluster_name}")

                except Exception as e:
                     logger.error(f"Error during aggregation for {video_id}/{cluster_name}: {e}. Skipping cluster.")
                     # Optionally remove partially created files if save failed
                     for path in [output_cluster_embedding_path, output_cluster_timestamps_path, output_cluster_indices_path]:
                         if path.exists():
                             path.unlink()

    logger.info("=== Cluster Embedding Generation Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate single embeddings per cluster using cross-attention.")
    parser.add_argument("--clustered_frames_dir", type=Path, default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/clusters",
                        help="Directory containing video/cluster subfolders from the clustering script.")
    parser.add_argument("--original_embeddings_dir", type=Path, default="swin_embeddings",
                        help="Directory containing the original per-frame embedding files (.npy).")
    parser.add_argument("--timestamps_dir", type=Path, 
                       default="/Users/akshat/Developer/Vid_Attention/data_extraction/data/QuerYD/processed_frames",
                       help="Directory containing frame timestamps.")
    parser.add_argument("--output_cluster_embeddings_dir", type=Path, default="cluster_embeddings",
                        help="Directory to save the generated cluster embeddings.")
    parser.add_argument("--embed_dim", type=int, default=768,
                        help="The dimensionality of the frame embeddings generated by the first script.")
    parser.add_argument("--num_attention_heads", type=int, default=4,
                        help="Number of attention heads for the aggregation mechanism.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps",
                        help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()

    device = torch.device(args.device)

    generate_cluster_embeddings(
        clustered_frames_dir=args.clustered_frames_dir,
        original_embeddings_dir=args.original_embeddings_dir,
        timestamps_dir=args.timestamps_dir,
        output_cluster_embeddings_dir=args.output_cluster_embeddings_dir,
        embed_dim=args.embed_dim,
        num_attention_heads=args.num_attention_heads,
        device=device
    )