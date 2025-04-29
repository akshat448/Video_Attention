import argparse
import os
import logging
import numpy as np
import h5py
import hdbscan # Import HDBSCAN
from PIL import Image # Needed to save frames as images
from tqdm import tqdm
from pathlib import Path
import shutil # To copy files

# Configure logging
logging.basicConfig(level=logging.INFO, # Use INFO for general progress, DEBUG for detailed frame/batch info
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_frames(video_frames_path_base):
    """Loads frames from either a .npy or .h5 file for a given base path."""
    npy_path = video_frames_path_base.with_suffix('.npy')
    h5_path = video_frames_path_base.with_suffix('.h5')

    if npy_path.exists():
        try:
            logger.debug(f"Attempting to load frames from NPY: {npy_path}")
            # Use allow_pickle=True if your NPY might contain pickled objects (less common for frame arrays)
            frames = np.load(npy_path)
            # NPY stores frames usually as (num_frames, H, W, C)
            logger.debug(f"Successfully loaded frames from {npy_path}. Shape: {frames.shape}")
            return frames
        except Exception as e:
            logger.error(f"Error loading NPY file {npy_path}: {e}")
            return None
    elif h5_path.exists():
        try:
            logger.debug(f"Attempting to load frames from HDF5: {h5_path}")
            with h5py.File(h5_path, 'r') as f:
                # HDF5 dataset might be named 'frames' as per the creation script
                if 'frames' in f:
                    frames = f['frames'][()]
                    # HDF5 might store as (num_frames, H, W, C)
                    logger.debug(f"Successfully loaded frames from {h5_path}. Shape: {frames.shape}")
                    return frames
                else:
                    logger.error(f"Dataset 'frames' not found in HDF5 file: {h5_path}")
                    return None
        except Exception as e:
            logger.error(f"Error loading HDF5 file {h5_path}: {e}")
            return None
    else:
        logger.warning(f"No .npy or .h5 file found for base path: {video_frames_path_base}")
        return None

def organize_frames_by_cluster(
    input_frames_dir: Path,
    input_embeddings_dir: Path,
    output_dir: Path,
    min_cluster_size: int = 5
):
    """
    Loads embeddings and frames, performs HDBSCAN clustering, and saves
    frames into folders based on cluster assignments.

    Args:
        input_frames_dir (Path): Directory containing video frame folders (.npy or .h5).
                                 Expected structure: input_frames_dir / video_id / video_id_frames.(npy/h5)
        input_embeddings_dir (Path): Directory containing the generated embedding files (.npy).
                                     Expected structure: input_embeddings_dir / video_id_embeddings.npy
        output_dir (Path): Directory where cluster folders will be created.
                           Structure created: output_dir / video_id / cluster_X / frame_YYYYY.png
        min_cluster_size (int): The minimum size of clusters for HDBSCAN.
    """
    logger.info(f"Starting frame organization by cluster.")
    logger.info(f"Input frames directory: {input_frames_dir}")
    logger.info(f"Input embeddings directory: {input_embeddings_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"HDBSCAN min_cluster_size: {min_cluster_size}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all embedding files
    embedding_files = list(input_embeddings_dir.glob("*_embeddings.npy"))

    if not embedding_files:
        logger.warning(f"No *_embeddings.npy files found in {input_embeddings_dir}. Exiting.")
        return

    logger.info(f"Found {len(embedding_files)} embedding files to process.")

    for embedding_file_path in tqdm(embedding_files, desc="Processing Videos"):
        video_id = embedding_file_path.stem.replace("_embeddings", "")
        logger.info(f"Processing video: {video_id}")

        # --- Construct Paths ---
        # Path to the potential frame file base name (without extension)
        # Assuming frames are stored like input_frames_dir / video_id / video_id_frames.(npy/h5)
        video_frame_dir = input_frames_dir / video_id
        frame_file_base = video_frame_dir / f"{video_id}_frames"
        output_video_dir = output_dir / video_id
        output_video_dir.mkdir(parents=True, exist_ok=True) # Create video-specific output directory

        # --- Load Frames ---
        frames = load_frames(frame_file_base)
        if frames is None or frames.shape[0] == 0:
            logger.warning(f"Skipping video {video_id}: Could not load frames or frames array is empty.")
            continue
        num_frames = frames.shape[0]
        logger.info(f"Loaded {num_frames} frames for {video_id}.")

        # --- Load Embeddings ---
        try:
            embeddings = np.load(embedding_file_path)
            logger.info(f"Loaded embeddings for {video_id}. Shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error loading embeddings for {video_id} from {embedding_file_path}: {e}")
            continue

        # --- Validate Data Consistency ---
        if embeddings.shape[0] != num_frames:
            logger.error(f"Skipping video {video_id}: Mismatch between number of frames ({num_frames}) and embeddings ({embeddings.shape[0]}).")
            continue

        # --- HDBSCAN Clustering ---
        logger.info(f"Performing HDBSCAN clustering for {video_id}...")
        cluster_labels = np.array([]) # Initialize as empty
        try:
            # Check if there are enough samples for clustering
            if num_frames < max(2, min_cluster_size):
                logger.warning(f"Not enough samples ({num_frames}) for HDBSCAN with min_cluster_size={min_cluster_size}. Skipping clustering for {video_id}.")
                # Assign all to a 'single_group' or similar if needed, or just skip saving by cluster
                # For now, we'll just note it and proceed without clustering labels
                cluster_labels = np.zeros(num_frames, dtype=int) # Assign all to cluster 0 if clustering skipped
                logger.info(f"Assigned all frames to cluster 0 due to insufficient samples for clustering.")
            else:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2)  # Set min_samples to 2
                cluster_labels = clusterer.fit_predict(embeddings)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                logger.info(f"HDBSCAN completed for {video_id}. Found {n_clusters} clusters (including noise points labeled -1).")
                unique_labels = sorted(list(set(cluster_labels)))
                logger.info(f"Unique cluster labels found: {unique_labels}")

        except Exception as e:
            logger.error(f"Error during HDBSCAN clustering for {video_id}: {e}")
            logger.warning(f"Clustering failed for {video_id}. Assigning all frames to a default group.")
            cluster_labels = np.zeros(num_frames, dtype=int) # Assign all to cluster 0 if clustering fails
            logger.info(f"Assigned all frames to cluster 0.")

        # --- Save Frames by Cluster ---
        logger.info(f"Saving frames for {video_id} into cluster folders...")

        for i in tqdm(range(num_frames), desc=f"Saving Frames for {video_id}"):
            frame = frames[i] # Get the i-th frame
            label = cluster_labels[i] # Get the i-th cluster label

            # Determine the target cluster folder name
            cluster_folder_name = f"cluster_{label}"

            # Create the full path for the cluster folder within the video directory
            cluster_output_path = output_video_dir / cluster_folder_name
            cluster_output_path.mkdir(parents=True, exist_ok=True)

            # Define the filename for the frame (using zero-padding for sorting)
            frame_filename = f"frame_{i:05d}.png" # Using PNG for lossless saving
            frame_output_path = cluster_output_path / frame_filename

            # Convert numpy frame (H, W, C) to PIL Image and save
            try:
                # Ensure the frame data type is suitable for PIL (e.g., uint8)
                if frame.dtype != np.uint8:
                    logger.debug(f"Converting frame {i} from {frame.dtype} to uint8.")
                    # Scale if necessary, assuming original might be float [0, 1] or similar
                    if np.max(frame) <= 1.0 and np.min(frame) >= 0.0:
                         frame = (frame * 255).astype(np.uint8)
                    else: # Assume it's already in 0-255 range or needs clipping
                         frame = np.clip(frame, 0, 255).astype(np.uint8)

                img = Image.fromarray(frame)
                img.save(frame_output_path)
                # logger.debug(f"Saved frame {i} to {frame_output_path}") # Too verbose, use INFO level for video progress

            except Exception as e:
                logger.error(f"Error saving frame {i} for video {video_id} to {frame_output_path}: {e}")
                # Optionally, continue or break depending on how critical saving each frame is

        logger.info(f"Finished saving frames for {video_id}.")

    logger.info("=== Frame Organization Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize video frames into folders based on HDBSCAN clustering of embeddings.")
    parser.add_argument("--input_frames_dir", type=Path, default="processed_frames",
                        help="Directory containing original processed frame folders (e.g., 'processed_frames').")
    parser.add_argument("--input_embeddings_dir", type=Path, default="swin_embeddings",
                        help="Directory containing the generated embedding files (.npy).")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save the organized cluster folders.")
    parser.add_argument("--min_cluster_size", type=int, default=2,
                        help="Minimum cluster size parameter for HDBSCAN.")

    args = parser.parse_args()

    organize_frames_by_cluster(
        input_frames_dir=args.input_frames_dir,
        input_embeddings_dir=args.input_embeddings_dir,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size
    )