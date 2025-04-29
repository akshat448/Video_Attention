import os
import json
import logging
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    # Try multiple possible JSON naming patterns
    json_paths = [
        os.path.join(json_dir, f"assets-{youtube_id}.json")
    ]
    
    for json_path in json_paths:
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded JSON metadata from {json_path}")
                return data
        except Exception as e:
            logger.error(f"Error loading JSON metadata from {json_path}: {e}")
    
    # Log all the paths we tried
    logger.warning(f"Could not find JSON metadata for {video_id}. Tried: {', '.join(json_paths)}")
    return None

def extract_caption_segments_from_json(json_data):
    """Extract caption segments from JSON metadata."""
    caption_segments = []
    
    if not json_data or 'result' not in json_data:
        logger.warning("Invalid or empty JSON data")
        return caption_segments
    
    try:
        result = json_data['result']
        audio_descriptions = result.get('audio_descriptions', [])
        
        for audio_desc in audio_descriptions:
            audio_clips = audio_desc.get('audio_clips', [])
            
            for clip in audio_clips:
                transcript = clip.get('transcript', [])
                clip_start_time = float(clip.get('start_time', 0))
                
                for segment in transcript:
                    segment_start_time = float(segment.get('start_time', 0)) + clip_start_time
                    segment_end_time = float(segment.get('end_time', 0)) + clip_start_time
                    sentence = segment.get('sentence', '')
                    
                    if sentence and segment_end_time > segment_start_time:
                        caption_segments.append((segment_start_time, segment_end_time, sentence))
        
        logger.info(f"Extracted {len(caption_segments)} caption segments from JSON metadata")
    except Exception as e:
        logger.error(f"Error extracting caption segments from JSON metadata: {e}")
    
    return caption_segments

def load_cluster_data(cluster_embeddings_dir, video_id):
    """
    Load cluster embeddings and timestamps for a given video.
    Returns dictionaries of embeddings and timespan data for each cluster.
    """
    video_embed_dir = Path(cluster_embeddings_dir) / video_id
    if not video_embed_dir.exists():
        logger.warning(f"No cluster embeddings directory found for {video_id}")
        return {}, {}
    
    cluster_embeddings = {}
    cluster_timespans = {}
    cluster_indices = {}
    
    # Find all cluster embedding files
    for embed_file in video_embed_dir.glob("cluster_*_embedding.npy"):
        try:
            cluster_id = embed_file.stem.split("_")[1]
            embedding = np.load(embed_file)
            
            # Also load the corresponding timestamp file for this cluster
            timestamp_file = video_embed_dir / f"cluster_{cluster_id}_timestamps.npy"
            indices_file = video_embed_dir / f"cluster_{cluster_id}_indices.npy"
            
            if timestamp_file.exists():
                timespan = np.load(timestamp_file)
                cluster_timespans[cluster_id] = timespan
            else:
                logger.warning(f"No timestamp file found for cluster {cluster_id} in {video_id}")
            
            if indices_file.exists():
                indices = np.load(indices_file)
                cluster_indices[cluster_id] = indices
            
            # Only add embeddings if we also have timestamp data
            if cluster_id in cluster_timespans:
                cluster_embeddings[cluster_id] = embedding
            
        except Exception as e:
            logger.error(f"Error loading data for cluster {cluster_id} in {video_id}: {e}")
    
    logger.info(f"Loaded {len(cluster_embeddings)} clusters with embeddings and timestamps for {video_id}")
    return cluster_embeddings, cluster_timespans

def map_clusters_to_captions_direct(cluster_timespans, caption_segments):
    """
    Map clusters to caption segments based on direct timespan overlap.
    Uses pre-calculated timespan data from Cluster_attention.py.
    """
    if not cluster_timespans or not caption_segments:
        return {}
    
    cluster_to_captions = defaultdict(list)
    
    for cluster_id, timespan in cluster_timespans.items():
        try:
            min_time = timespan[0]
            max_time = timespan[1]
            
            for start_time, end_time, caption_text in caption_segments:
                if (start_time <= max_time and end_time >= min_time):
                    cluster_to_captions[cluster_id].append(caption_text)
        except Exception as e:
            logger.error(f"Error mapping cluster {cluster_id} to captions: {e}")
    
    # Combine multiple captions for each cluster
    combined_mapping = {}
    for cluster_id, captions in cluster_to_captions.items():
        if captions:  # Only include clusters with captions
            combined_mapping[cluster_id] = " ".join(captions)
    
    return combined_mapping

def prepare_clip_training_data(cluster_embeddings, cluster_to_captions, output_dir, video_id):
    """Prepare and save training data for CLIP."""
    clip_data = []
    
    for cluster_id, caption in cluster_to_captions.items():
        if cluster_id in cluster_embeddings:
            clip_data.append((cluster_embeddings[cluster_id], caption))
    
    # Save the data
    os.makedirs(output_dir, exist_ok=True)
    clip_data_path = os.path.join(output_dir, f"{video_id}_cluster_clip_data.npz")
    
    if clip_data:
        clip_embeddings = np.array([item[0] for item in clip_data])
        clip_captions = np.array([item[1] for item in clip_data])
        cluster_ids = np.array([cid for cid in cluster_to_captions.keys() if cid in cluster_embeddings])
        
        np.savez(clip_data_path, 
                embeddings=clip_embeddings, 
                captions=clip_captions,
                cluster_ids=cluster_ids)
        
        logger.info(f"Saved CLIP training data for {len(clip_data)} cluster-caption pairs to {clip_data_path}")
        return clip_data_path
    else:
        logger.warning(f"No valid cluster-caption pairs for {video_id}")
        return None

def process_video(video_id, json_dir, cluster_embeddings_dir, output_dir):
    """Process a single video: map cluster embeddings to caption segments."""
    logger.info(f"=== Processing video: {video_id} ===")
    
    # Step 1: Load JSON metadata and extract caption segments
    json_data = load_json_metadata(json_dir, video_id)
    if not json_data:
        return None
    
    caption_segments = extract_caption_segments_from_json(json_data)
    if not caption_segments:
        logger.warning(f"No caption segments found for {video_id}")
        return None
    
    # Step 2: Load cluster embeddings and timestamps
    cluster_embeddings, cluster_timespans = load_cluster_data(cluster_embeddings_dir, video_id)
    if not cluster_embeddings or not cluster_timespans:
        logger.warning(f"No cluster data found for {video_id}")
        return None
    
    # Step 3: Map clusters to caption segments using direct timespan data
    cluster_to_captions = map_clusters_to_captions_direct(cluster_timespans, caption_segments)
    if not cluster_to_captions:
        logger.warning(f"No clusters could be mapped to captions for {video_id}")
        return None
    
    # Step 4: Prepare CLIP training data
    clip_data_path = prepare_clip_training_data(cluster_embeddings, cluster_to_captions, output_dir, video_id)
    
    # Return statistics
    result = {
        "video_id": video_id,
        "total_clusters": len(cluster_embeddings),
        "mapped_clusters": len(cluster_to_captions),
        "mapping_percentage": 100 * len(cluster_to_captions) / len(cluster_embeddings) if cluster_embeddings else 0,
        "clip_data_path": clip_data_path
    }
    
    logger.info(f"Processed {video_id}: {result['mapping_percentage']:.2f}% of clusters mapped")
    return result

def main():
    parser = argparse.ArgumentParser(description="Map cluster embeddings to caption segments for CLIP training")
    
    # Directory arguments with updated paths
    parser.add_argument("--json_dir", type=str, 
                       default="/Users/akshat/Developer/Vid_Attention/data_extraction/data/QuerYD/JSON_Metadata",
                       help="Directory containing JSON metadata files")
    parser.add_argument("--cluster_embeddings_dir", type=str, 
                       default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/cluster_embeddings",
                       help="Directory containing cluster embeddings with timestamps")
    parser.add_argument("--output_dir", type=str, default="cluster_clip_data",
                       help="Directory for output files (CLIP training data)")
    parser.add_argument("--video_ids", type=str, nargs="+",
                       help="Specific video IDs to process (default: process all)")
    parser.add_argument("--debug", action="store_true",
                        help="Print additional debug information")
    
    args = parser.parse_args()
    
    # Set DEBUG level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Validate input directories exist
    logger.info("Checking input directories...")
    for path_arg, path_val in [
        ("json_dir", args.json_dir),
        ("cluster_embeddings_dir", args.cluster_embeddings_dir)
    ]:
        path = Path(path_val)
        if path.exists():
            logger.info(f"✓ {path_arg}: {path} (exists)")
        else:
            logger.warning(f"✗ {path_arg}: {path} (does not exist)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find videos to process
    if args.video_ids:
        video_ids = args.video_ids
    else:
        # Find all video directories in cluster embeddings directory
        cluster_embeddings_dir = Path(args.cluster_embeddings_dir)
        if not cluster_embeddings_dir.exists():
            logger.error(f"Cluster embeddings directory not found: {cluster_embeddings_dir}")
            return
            
        cluster_video_dirs = [d.name for d in cluster_embeddings_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(cluster_video_dirs)} videos in cluster embeddings directory")
        
        # Show sample video IDs
        if cluster_video_dirs:
            logger.info(f"Sample video IDs: {', '.join(cluster_video_dirs[:5])}... (first 5)")
            
        video_ids = cluster_video_dirs
    
    logger.info(f"Processing {len(video_ids)} videos")
    
    # Process each video
    results = []
    for video_id in tqdm(video_ids):
        result = process_video(
            video_id=video_id,
            json_dir=args.json_dir,
            cluster_embeddings_dir=args.cluster_embeddings_dir,
            output_dir=args.output_dir
        )
        if result:
            results.append(result)
    
    # Save overall results
    summary_path = os.path.join(args.output_dir, "cluster_mapping_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "total_videos": len(video_ids),
            "successful_videos": len(results),
            "results": results
        }, f, indent=2)
    
    logger.info(f"Saved mapping summary to {summary_path}")
    logger.info(f"Successfully processed {len(results)} out of {len(video_ids)} videos")
    
    if results:
        avg_mapping = sum(r["mapping_percentage"] for r in results) / len(results)
        logger.info(f"Average mapping percentage: {avg_mapping:.2f}%")

if __name__ == "__main__":
    main()