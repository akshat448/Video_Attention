import os
import json
import logging
import argparse
import numpy as np
import h5py
import hdbscan
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pickle_metadata(pickle_path):
    """Load caption data from pickle file."""
    try:
        with open(pickle_path, 'rb') as f:
            video_caption_dict = pickle.load(f)
        logger.info(f"Successfully loaded pickle data from {pickle_path}")
        return video_caption_dict
    except Exception as e:
        logger.error(f"Error loading pickle data from {pickle_path}: {e}")
        return None

def load_json_metadata(json_dir, video_id):
    """
    Load and parse JSON metadata for a specific video.
    Returns the parsed JSON data.
    """
    # Extract YouTube ID from video_id (assuming format: video-{youtube_id})
    if video_id.startswith("video-"):
        youtube_id = video_id[6:]  # Remove 'video-' prefix
    else:
        youtube_id = video_id
    
    # Construct the JSON file path
    json_path = os.path.join(json_dir, f"assets-{youtube_id}.json")
    
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded JSON metadata from {json_path}")
            return data
        else:
            logger.warning(f"JSON metadata file not found: {json_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading JSON metadata from {json_path}: {e}")
        return None

def extract_caption_segments_from_json(json_data):
    """
    Extract caption segments from JSON metadata.
    Returns list of (start_time, end_time, caption_text) tuples.
    """
    caption_segments = []
    
    if not json_data or 'result' not in json_data:
        logger.warning("Invalid or empty JSON data")
        return caption_segments
    
    try:
        # Navigate to the audio descriptions
        result = json_data['result']
        audio_descriptions = result.get('audio_descriptions', [])
        
        for audio_desc in audio_descriptions:
            audio_clips = audio_desc.get('audio_clips', [])
            
            for clip in audio_clips:
                transcript = clip.get('transcript', [])
                clip_start_time = float(clip.get('start_time', 0))
                
                for segment in transcript:
                    # Calculate absolute timestamp by adding clip's start time to segment's relative time
                    segment_start_time = float(segment.get('start_time', 0)) + clip_start_time
                    segment_end_time = float(segment.get('end_time', 0)) + clip_start_time
                    sentence = segment.get('sentence', '')
                    
                    if sentence and segment_end_time > segment_start_time:
                        caption_segments.append((segment_start_time, segment_end_time, sentence))
        
        logger.info(f"Extracted {len(caption_segments)} caption segments from JSON metadata")
    except Exception as e:
        logger.error(f"Error extracting caption segments from JSON metadata: {e}")
    
    return caption_segments

def load_frames_and_timestamps(frames_dir, video_id):
    """
    Load frames and timestamps for a given video.
    Returns frames array and timestamps array.
    """
    video_dir = Path(frames_dir) / video_id
    npy_path = video_dir / f"{video_id}_frames.npy"
    h5_path = video_dir / f"{video_id}_frames.h5"
    timestamp_path = video_dir / f"{video_id}_timestamps.npy"
    
    frames = None
    timestamps = None
    
    # Try to load frames
    if npy_path.exists():
        try:
            frames = np.load(npy_path)
            logger.info(f"Loaded frames from NPY: {npy_path}, shape: {frames.shape}")
        except Exception as e:
            logger.error(f"Error loading NPY file {npy_path}: {e}")
    elif h5_path.exists():
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'frames' in f:
                    frames = f['frames'][()]
                    logger.info(f"Loaded frames from HDF5: {h5_path}, shape: {frames.shape}")
                    
                    # Check if timestamps are stored in the same H5 file
                    if 'timestamps' in f:
                        timestamps = f['timestamps'][()]
                        logger.info(f"Loaded timestamps from HDF5, shape: {timestamps.shape}")
        except Exception as e:
            logger.error(f"Error loading HDF5 file {h5_path}: {e}")
    
    # Try to load timestamps if not already loaded
    if timestamps is None and timestamp_path.exists():
        try:
            timestamps = np.load(timestamp_path)
            logger.info(f"Loaded timestamps from NPY: {timestamp_path}, shape: {timestamps.shape}")
        except Exception as e:
            logger.error(f"Error loading timestamps from {timestamp_path}: {e}")
    
    # If frames loaded but no timestamps, need to generate them
    if frames is not None and timestamps is None:
        logger.warning(f"No timestamp data found for {video_id}. Will need to generate timestamps.")
        timestamps = None
    
    return frames, timestamps

def load_or_generate_embeddings(frames, embeddings_dir, video_id, device="cpu"):
    """
    Load pre-computed embeddings or generate new ones if needed.
    """
    embedding_path = Path(embeddings_dir) / f"{video_id}_embeddings.npy"
    
    if embedding_path.exists():
        try:
            embeddings = np.load(embedding_path)
            logger.info(f"Loaded pre-computed embeddings from {embedding_path}, shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings from {embedding_path}: {e}")
    
    # If embeddings don't exist or failed to load, we need to generate them
    logger.info(f"No pre-computed embeddings found for {video_id}. Need to generate them.")
    
    # Import needed only if generating embeddings
    try:
        import torch
        import sys
        sys.path.append("/Users/akshat/Developer/Vid_Attention/clustering")
        from Embedding_generator import get_preprocessor, generate_embeddings
        import timm
        
        # Load model
        model_name = "swin_tiny_patch4_window7_224"
        logger.info(f"Loading {model_name} model for embedding generation...")
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model = model.to(device)
        model.eval()
        
        # Get preprocessor
        preprocess_fn = get_preprocessor(model)
        
        # Generate embeddings
        batch_size = 16
        embeddings = generate_embeddings(frames, model, preprocess_fn, batch_size, device)
        
        # Save embeddings
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path, embeddings)
        logger.info(f"Generated and saved embeddings to {embedding_path}, shape: {embeddings.shape}")
        
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

def extract_caption_segments_from_pickle(pickle_data, video_id):
    """
    Extract caption segments from the pickle data for a specific video.
    Returns list of (start_time, end_time, caption_text) tuples.
    """
    caption_segments = []
    
    try:
        if video_id not in pickle_data:
            logger.warning(f"Video ID {video_id} not found in pickle data")
            return caption_segments
        
        # Get the audio segments for this video
        audio_segments = pickle_data[video_id]
        
        # Check pickle structure and print debugging information
        logger.debug(f"Audio segments type: {type(audio_segments)}")
        if isinstance(audio_segments, (list, tuple)) and len(audio_segments) > 0:
            logger.debug(f"First segment type: {type(audio_segments[0])}")
            logger.debug(f"First segment content: {audio_segments[0]}")
        
        # Handle list structure with time-stamped captions
        if isinstance(audio_segments, list):
            for segment in audio_segments:
                # Case 1: Handle dictionary format
                if isinstance(segment, dict):
                    start_time = segment.get('start_time', None)
                    end_time = segment.get('end_time', None)
                    text = segment.get('text', segment.get('caption', ''))
                    
                    # Make sure we have valid time values
                    if text and start_time is not None and end_time is not None:
                        try:
                            start_time = float(start_time)
                            end_time = float(end_time)
                            if end_time > start_time:
                                caption_segments.append((start_time, end_time, text))
                        except (ValueError, TypeError):
                            logger.debug(f"Could not convert time values to float: {start_time}, {end_time}")
                
                # Case 2: Handle tuple/list format that contains timestamps
                elif isinstance(segment, (tuple, list)) and len(segment) >= 3:
                    try:
                        # Try to extract as (start_time, end_time, text)
                        if isinstance(segment[0], (int, float)) and isinstance(segment[1], (int, float)):
                            start_time = float(segment[0])
                            end_time = float(segment[1])
                            text = segment[2]
                            
                            if text and end_time > start_time:
                                caption_segments.append((start_time, end_time, text))
                    except (ValueError, TypeError):
                        # If we can't convert to float, this might be a word tuple
                        # We'll skip it since it doesn't have timestamp information
                        logger.debug(f"Skipping word tuple: {segment}")
        
        # Handle dictionary structure
        elif isinstance(audio_segments, dict):
            for segment_id, segment_info in audio_segments.items():
                # Process dictionary format
                if isinstance(segment_info, dict):
                    start_time = segment_info.get('start_time', None)
                    end_time = segment_info.get('end_time', None)
                    text = segment_info.get('text', segment_info.get('caption', ''))
                    
                    # Make sure we have valid time values
                    if text and start_time is not None and end_time is not None:
                        try:
                            start_time = float(start_time)
                            end_time = float(end_time)
                            if end_time > start_time:
                                caption_segments.append((start_time, end_time, text))
                        except (ValueError, TypeError):
                            logger.debug(f"Could not convert time values to float: {start_time}, {end_time}")
                
                # Process tuple/list format
                elif isinstance(segment_info, (tuple, list)) and len(segment_info) >= 3:
                    try:
                        start_time = float(segment_info[0])
                        end_time = float(segment_info[1])
                        text = segment_info[2]
                        
                        if text and end_time > start_time:
                            caption_segments.append((start_time, end_time, text))
                    except (ValueError, TypeError):
                        logger.debug(f"Skipping tuple that can't be interpreted as timings: {segment_info}")
                
                # Process segment ID with timestamp format (e.g., "5.2-10.5")
                elif isinstance(segment_info, str) and '-' in segment_id:
                    try:
                        time_parts = segment_id.split('-')
                        start_time = float(time_parts[0])
                        end_time = float(time_parts[1])
                        text = segment_info
                        
                        if text and end_time > start_time:
                            caption_segments.append((start_time, end_time, text))
                    except (ValueError, IndexError):
                        logger.debug(f"Could not extract timestamps from segment_id: {segment_id}")
        
        logger.info(f"Extracted {len(caption_segments)} caption segments for video {video_id}")
    except Exception as e:
        logger.error(f"Error extracting caption segments for {video_id}: {e}")
    
    return caption_segments

def generate_timestamps_for_frames(num_frames, target_fps=1):
    """
    Generate timestamps for frames based on a target FPS.
    """
    try:
        # Generate equally spaced timestamps based on target_fps
        frame_interval = 1.0 / target_fps
        timestamps = np.arange(0, num_frames * frame_interval, frame_interval)
        
        # Ensure we have exactly num_frames timestamps
        if len(timestamps) > num_frames:
            timestamps = timestamps[:num_frames]
        elif len(timestamps) < num_frames:
            # Pad with extrapolated values if needed
            last_time = timestamps[-1]
            additional_times = np.arange(1, num_frames - len(timestamps) + 1) * frame_interval + last_time
            timestamps = np.concatenate([timestamps, additional_times])
        
        logger.info(f"Generated {len(timestamps)} timestamps at {target_fps} FPS")
        return timestamps
    except Exception as e:
        logger.error(f"Error generating timestamps: {e}")
        return None

def map_frames_to_captions_by_timestamp(timestamps, caption_segments):
    """
    Map frame indices to captions based on timestamp overlap.
    Returns a dictionary mapping frame indices to caption text, and a set of mapped indices.
    """
    direct_frame_to_caption = {}
    directly_mapped_indices = set()
    
    # Debug the first few timestamp values
    logger.debug(f"First few timestamps: {timestamps[:3]}, type: {type(timestamps[0])}")
    if caption_segments:
        logger.debug(f"First caption segment: {caption_segments[0]}")
    
    # Ensure timestamps are floats
    try:
        timestamps_array = np.array(timestamps, dtype=float)
    except ValueError as e:
        logger.error(f"Error converting timestamps to float: {e}")
        return {}, set()
    
    for i, timestamp in enumerate(timestamps_array):
        for segment in caption_segments:
            try:
                start_time, end_time, caption_text = segment
                
                # Convert times to float if they aren't already
                start_time = float(start_time)
                end_time = float(end_time)
                
                if start_time <= timestamp < end_time:
                    # Map this frame to the caption
                    direct_frame_to_caption[i] = caption_text
                    directly_mapped_indices.add(i)
                    break  # Assign to the first matching caption segment
            except (ValueError, TypeError) as e:
                logger.debug(f"Error processing caption segment {segment}: {e}")
                continue
    
    logger.info(f"Directly mapped {len(directly_mapped_indices)} frames to captions by timestamp")
    return direct_frame_to_caption, directly_mapped_indices

def apply_hdbscan_clustering(embeddings, min_cluster_size=2, min_samples=2):
    """
    Apply HDBSCAN clustering to the embeddings.
    Returns cluster labels for each embedding.
    """
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        logger.info(f"HDBSCAN clustering found {n_clusters} clusters and {n_noise} noise points")
        return cluster_labels
    except Exception as e:
        logger.error(f"Error during HDBSCAN clustering: {e}")
        return None

def propagate_captions_via_clusters(direct_frame_to_caption, directly_mapped_indices, 
                                   cluster_labels, num_frames):
    """
    Propagate captions to unmapped frames within the same cluster as mapped frames.
    Returns final mapping and set of all mapped indices.
    """
    final_frame_to_caption = direct_frame_to_caption.copy()
    all_mapped_indices = directly_mapped_indices.copy()
    
    # Create a mapping from cluster ID to captions from directly mapped frames
    cluster_to_captions = defaultdict(list)
    for i in directly_mapped_indices:
        cluster_id = cluster_labels[i]
        if cluster_id != -1:  # Skip noise points
            cluster_to_captions[cluster_id].append((i, direct_frame_to_caption[i]))
    
    # For each unmapped frame, check if we can propagate a caption
    propagated_count = 0
    for j in range(num_frames):
        if j not in all_mapped_indices:
            cluster_id = cluster_labels[j]
            if cluster_id != -1 and cluster_id in cluster_to_captions:
                # Propagate caption from the first directly mapped frame in this cluster
                source_idx, caption = cluster_to_captions[cluster_id][0]
                final_frame_to_caption[j] = caption
                all_mapped_indices.add(j)
                propagated_count += 1
    
    logger.info(f"Propagated captions to {propagated_count} additional frames via clusters")
    logger.info(f"Total mapped frames: {len(all_mapped_indices)} out of {num_frames}")
    
    return final_frame_to_caption, all_mapped_indices

def prepare_clip_training_data(embeddings, final_frame_to_caption, output_dir, video_id):
    """
    Prepare and save training data for CLIP in the format (embedding, caption).
    """
    clip_data = []
    
    for idx, caption in final_frame_to_caption.items():
        clip_data.append((embeddings[idx], caption))
    
    # Save the data
    os.makedirs(output_dir, exist_ok=True)
    clip_data_path = os.path.join(output_dir, f"{video_id}_clip_data.npz")
    
    # Extract embeddings and captions into separate arrays
    clip_embeddings = np.array([item[0] for item in clip_data])
    clip_captions = np.array([item[1] for item in clip_data])
    
    np.savez(clip_data_path, 
             embeddings=clip_embeddings, 
             captions=clip_captions,
             frame_indices=np.array(list(final_frame_to_caption.keys())))
    
    logger.info(f"Saved CLIP training data for {len(clip_data)} frame-caption pairs to {clip_data_path}")
    
    return clip_data_path

def visualize_caption_mapping(embeddings, timestamps, all_mapped_indices, cluster_labels, output_dir, video_id):
    """
    Visualize the mapping of frames to captions, colored by clusters.
    """
    try:
        from sklearn.manifold import TSNE
        
        # Only do t-SNE if we have a reasonable number of frames
        if len(embeddings) > 2000:
            logger.info(f"Too many frames ({len(embeddings)}) for t-SNE visualization. Sampling 2000.")
            indices = np.random.choice(len(embeddings), 2000, replace=False)
            embeddings_subset = embeddings[indices]
            mapped_subset = np.array([i in all_mapped_indices for i in indices])
            cluster_labels_subset = cluster_labels[indices]
        else:
            embeddings_subset = embeddings
            mapped_subset = np.array([i in all_mapped_indices for i in range(len(embeddings))])
            cluster_labels_subset = cluster_labels
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_subset)
        
        # Create visualization directory
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot 1: Mapped vs Unmapped Points
        plt.figure(figsize=(12, 10))
        plt.scatter(embeddings_2d[~mapped_subset, 0], embeddings_2d[~mapped_subset, 1], 
                   c='lightgray', label='Unmapped', alpha=0.6, s=10)
        plt.scatter(embeddings_2d[mapped_subset, 0], embeddings_2d[mapped_subset, 1], 
                   c='blue', label='Mapped', alpha=0.8, s=10)
        plt.title(f'Caption Mapping for {video_id} (Blue: Mapped, Gray: Unmapped)')
        plt.legend()
        plt.savefig(os.path.join(viz_dir, f"{video_id}_caption_mapping.png"), dpi=150)
        plt.close()
        
        # Plot 2: Cluster Visualization
        plt.figure(figsize=(12, 10))
        # Adjust cluster labels for visualization (shift by +1 so noise is 0)
        adjusted_labels = cluster_labels_subset + 1
        max_label = np.max(adjusted_labels) if adjusted_labels.size > 0 else 0
        cmap = plt.get_cmap('tab20', max(1, max_label + 1))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   c=adjusted_labels, cmap=cmap, alpha=0.7, s=10)
        plt.title(f'HDBSCAN Clusters for {video_id}')
        plt.savefig(os.path.join(viz_dir, f"{video_id}_clusters.png"), dpi=150)
        plt.close()
        
        # Plot 3: Temporal distribution of mapped frames
        plt.figure(figsize=(15, 5))
        mapped_timestamps = timestamps[list(all_mapped_indices)]
        plt.hist(mapped_timestamps, bins=50, alpha=0.7, color='blue', label='Mapped Frames')
        plt.hist(timestamps, bins=50, alpha=0.3, color='gray', label='All Frames')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frame Count')
        plt.title(f'Temporal Distribution of Mapped Frames for {video_id}')
        plt.legend()
        plt.savefig(os.path.join(viz_dir, f"{video_id}_temporal_distribution.png"), dpi=150)
        plt.close()
        
        logger.info(f"Saved visualizations to {viz_dir}")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

def process_video(video_id, frames_dir, json_dir, pickle_data, embeddings_dir, output_dir, 
                 min_cluster_size=2, min_samples=2, target_fps=1, device="cpu"):
    """
    Process a single video with the hybrid approach.
    Tries JSON metadata first, falls back to pickle data if needed.
    """
    logger.info(f"=== Processing video: {video_id} ===")
    
    # Step 1: Load frames and timestamps
    frames, timestamps = load_frames_and_timestamps(frames_dir, video_id)
    if frames is None:
        return None
    
    # Generate timestamps if not available
    if timestamps is None:
        timestamps = generate_timestamps_for_frames(len(frames), target_fps)
        if timestamps is None:
            return None
        
        # Optionally save the generated timestamps
        timestamp_path = os.path.join(frames_dir, video_id, f"{video_id}_timestamps.npy")
        np.save(timestamp_path, timestamps)
        logger.info(f"Saved generated timestamps to {timestamp_path}")
    
    # Step 2: Load or generate embeddings
    embeddings = load_or_generate_embeddings(frames, embeddings_dir, video_id, device)
    if embeddings is None or len(embeddings) != len(frames):
        return None
    
    # Step 3: Try to extract caption segments from JSON first
    json_data = load_json_metadata(json_dir, video_id)
    caption_segments = []
    data_source = None
    
    if json_data:
        caption_segments = extract_caption_segments_from_json(json_data)
        if caption_segments:
            data_source = "json"
    
    # Fall back to pickle data if no JSON segments found
    if not caption_segments and pickle_data and video_id in pickle_data:
        logger.info(f"Falling back to pickle data for {video_id}")
        caption_segments = extract_caption_segments_from_pickle(pickle_data, video_id)
        if caption_segments:
            data_source = "pickle"
    
    if not caption_segments:
        logger.warning(f"No caption segments found for {video_id}")
        return None
    
    # Step 4: Map frames to captions based on timestamps
    direct_frame_to_caption, directly_mapped_indices = map_frames_to_captions_by_timestamp(
        timestamps, caption_segments)
    
    # Skip if no direct mappings
    if not directly_mapped_indices:
        logger.warning(f"No frames could be directly mapped to captions for {video_id}")
        return None
    
    # Step 5: Apply HDBSCAN clustering
    cluster_labels = apply_hdbscan_clustering(embeddings, min_cluster_size, min_samples)
    if cluster_labels is None:
        return None
    
    # Step 6: Propagate captions via clusters
    final_frame_to_caption, all_mapped_indices = propagate_captions_via_clusters(
        direct_frame_to_caption, directly_mapped_indices, cluster_labels, len(frames))
    
    # Step 7: Prepare CLIP training data
    clip_data_path = prepare_clip_training_data(embeddings, final_frame_to_caption, output_dir, video_id)
    
    # Step 8: Create visualizations
    visualize_caption_mapping(embeddings, timestamps, all_mapped_indices, cluster_labels, output_dir, video_id)
    
    # Return statistics
    result = {
        "video_id": video_id,
        "total_frames": len(frames),
        "directly_mapped": len(directly_mapped_indices),
        "total_mapped": len(all_mapped_indices),
        "mapping_percentage": 100 * len(all_mapped_indices) / len(frames),
        "clip_data_path": clip_data_path,
        "data_source": data_source
    }
    
    logger.info(f"Processed {video_id}: {result['mapping_percentage']:.2f}% of frames mapped using {data_source} data")
    return result

def main():
    parser = argparse.ArgumentParser(description="Hybrid approach for timestamped caption mapping and HDBSCAN clustering")
    
    # Directory arguments
    parser.add_argument("--frames_dir", type=str, default="processed_frames",
                       help="Directory containing processed frames")
    parser.add_argument("--json_dir", type=str, default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/data/JSON_Metadata",
                       help="Directory containing JSON metadata files")
    parser.add_argument("--pickle_path", type=str, default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/data/QuerYD Captions Combined.pkl",
                       help="Path to the caption pickle file (fallback)")
    parser.add_argument("--embeddings_dir", type=str, default="swin_embeddings",
                       help="Directory for storing/loading frame embeddings")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory for output files (CLIP training data, visualizations)")
    
    # Processing arguments
    parser.add_argument("--min_cluster_size", type=int, default=2,
                       help="Minimum cluster size for HDBSCAN (default: 2)")
    parser.add_argument("--min_samples", type=int, default=2,
                       help="Minimum samples parameter for HDBSCAN (default: 2)")
    parser.add_argument("--target_fps", type=float, default=1.0,
                       help="Target FPS used when extracting frames (default: 1.0)")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device for embedding generation ('cpu', 'cuda', or 'mps')")
    parser.add_argument("--video_ids", type=str, nargs="+",
                       help="Specific video IDs to process (default: process all)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pickle data once for all videos (as fallback)
    pickle_data = load_pickle_metadata(args.pickle_path)
    
    # Find videos to process
    if args.video_ids:
        video_ids = args.video_ids
    else:
        # Find all video directories in frames_dir
        frame_video_ids = [d for d in os.listdir(args.frames_dir) 
                         if os.path.isdir(os.path.join(args.frames_dir, d))]
        
        logger.info(f"Found {len(frame_video_ids)} videos in frames directory")
        video_ids = frame_video_ids
    
    logger.info(f"Processing {len(video_ids)} videos")
    
    # Process each video
    results = []
    for video_id in tqdm(video_ids):
        result = process_video(
            video_id=video_id,
            frames_dir=args.frames_dir,
            json_dir=args.json_dir,
            pickle_data=pickle_data,
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            target_fps=args.target_fps,
            device=args.device
        )
        if result:
            results.append(result)
    
    # Save overall results
    summary_path = os.path.join(args.output_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "total_videos": len(video_ids),
            "successful_videos": len(results),
            "results": results
        }, f, indent=2)
    
    logger.info(f"Saved processing summary to {summary_path}")
    logger.info(f"Successfully processed {len(results)} out of {len(video_ids)} videos")
    
    if results:
        avg_mapping = sum(r["mapping_percentage"] for r in results) / len(results)
        logger.info(f"Average mapping percentage: {avg_mapping:.2f}%")
        
        # Count videos by data source
        json_count = sum(1 for r in results if r["data_source"] == "json")
        pickle_count = sum(1 for r in results if r["data_source"] == "pickle")
        logger.info(f"Videos processed using JSON: {json_count}, using pickle: {pickle_count}")

if __name__ == "__main__":
    main()