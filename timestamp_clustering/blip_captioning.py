import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_cluster_embeddings(cluster_dir):
    """Load cluster embeddings from individual .npy files in video-specific directories."""
    cluster_dir = Path(cluster_dir)
    
    # Check if directory exists
    if not cluster_dir.exists():
        logger.error(f"Cluster directory not found: {cluster_dir}")
        return {}
    
    # Find all video directories in the cluster directory
    video_dirs = [d for d in cluster_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(video_dirs)} video directories in {cluster_dir}")
    
    all_embeddings = {}
    
    for video_dir in video_dirs:
        video_id = video_dir.name
        
        # Find all cluster embedding files in this directory
        embedding_files = list(video_dir.glob("cluster_*_embedding.npy"))
        
        if not embedding_files:
            logger.warning(f"No cluster embedding files found for {video_id}")
            continue
        
        # Load each embedding file
        video_clusters = {}
        for embed_file in embedding_files:
            try:
                # Extract cluster ID from filename (cluster_X_embedding.npy)
                cluster_id = embed_file.stem.split("_")[1]
                embedding = np.load(embed_file)
                video_clusters[cluster_id] = embedding
            except Exception as e:
                logger.error(f"Error loading {embed_file}: {e}")
        
        if video_clusters:
            all_embeddings[video_id] = video_clusters
            logger.info(f"Loaded {len(video_clusters)} cluster embeddings for {video_id}")
    
    return all_embeddings

def load_frames_for_cluster(video_id, cluster_id, frames_dir):
    """Load representative frames for a specific cluster to use for captioning."""
    # Updated path structure to match the new directory layout
    cluster_dir = Path(frames_dir) / video_id / f"cluster_{cluster_id}"
    
    if not cluster_dir.exists():
        logger.warning(f"No cluster folder found for {video_id}, cluster {cluster_id}")
        return None
    
    # Get a few sample frames from this cluster
    frame_files = list(cluster_dir.glob("*.png")) or list(cluster_dir.glob("*.jpg"))
    
    if not frame_files:
        logger.warning(f"No frames found for {video_id}, cluster {cluster_id}")
        return None
    
    # Sort frames and take up to 3 samples (beginning, middle, end)
    frame_files.sort()
    sample_indices = [0]
    if len(frame_files) > 2:
        sample_indices.append(len(frame_files) // 2)
    if len(frame_files) > 1:
        sample_indices.append(len(frame_files) - 1)
    
    sample_frames = [frame_files[i] for i in sample_indices]
    logger.info(f"Selected {len(sample_frames)} sample frames for {video_id}, cluster {cluster_id}")
    
    return sample_frames

def generate_captions_with_blip(frames_dir, embeddings_dict, device="cpu"):
    """Generate captions using BLIP model with actual frames."""
    # Load BLIP model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    
    results = {}
    
    for video_id, clusters in tqdm(embeddings_dict.items(), desc="Generating captions"):
        video_results = {}
        
        for cluster_id in clusters.keys():
            # Instead of using embeddings, load actual frames from this cluster
            sample_frames = load_frames_for_cluster(video_id, cluster_id, frames_dir)
            
            if not sample_frames:
                video_results[cluster_id] = "No frames available for this cluster"
                continue
            
            # Generate captions for each sample frame and combine them
            captions = []
            for frame_path in sample_frames:
                try:
                    # Load and process image
                    from PIL import Image
                    image = Image.open(frame_path).convert('RGB')
                    
                    # Generate caption
                    inputs = processor(image, return_tensors="pt").to(device)
                    out = model.generate(**inputs, max_length=75)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    captions.append(caption)
                except Exception as e:
                    logger.error(f"Error generating caption for {frame_path}: {e}")
            
            # Combine captions into a summary
            if captions:
                summary = "This cluster shows " + "; also ".join(captions)
                video_results[cluster_id] = summary
            else:
                video_results[cluster_id] = "Failed to generate captions for this cluster"
        
        results[video_id] = video_results
    
    return results

def save_results(results, output_dir):
    """Save generated captions to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_results_path = output_dir / "all_cluster_captions.json"
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved all captions to {all_results_path}")
    
    for video_id, captions in results.items():
        video_path = output_dir / f"{video_id}_captions.json"
        with open(video_path, 'w') as f:
            json.dump(captions, f, indent=2)
        
        logger.info(f"Saved captions for {video_id} to {video_path}")
    
    summary_path = output_dir / "caption_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("CLUSTER CAPTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for video_id, captions in results.items():
            f.write(f"VIDEO: {video_id}\n")
            f.write("-" * 80 + "\n\n")
            
            for cluster_id, caption in sorted(captions.items()):
                f.write(f"Cluster {cluster_id}: {caption}\n\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    logger.info(f"Saved summary to {summary_path}")
    return all_results_path


def main():
    parser = argparse.ArgumentParser(description="Generate captions for cluster frames using BLIP")
    
    parser.add_argument("--cluster_dir", type=str, default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/cluster_embeddings",
                       help="Directory containing video-specific cluster embedding folders")
    # Updated default path to point to the new clusters directory
    parser.add_argument("--frames_dir", type=str, default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/clusters",
                       help="Directory containing organized frames by cluster")
    parser.add_argument("--output_dir", type=str, default="cluster_captions_blip",
                       help="Directory to save generated captions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else 
                                                    "mps" if torch.backends.mps.is_available() else "cpu",
                       help="Device for inference (cuda, mps, or cpu)")
    parser.add_argument("--video_ids", type=str, nargs="+", default=None,
                       help="Optional: Process only specific video IDs")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load cluster embeddings (we need this to know which clusters exist)
    cluster_embeddings = load_cluster_embeddings(args.cluster_dir)
    if not cluster_embeddings:
        logger.error("No cluster embeddings found. Exiting.")
        return
    
    # Filter videos if requested
    if args.video_ids:
        filtered_embeddings = {vid: emb for vid, emb in cluster_embeddings.items() 
                              if vid in args.video_ids or vid.replace("video-", "") in args.video_ids}
        
        if not filtered_embeddings:
            logger.warning(f"None of the specified video IDs were found. Available videos: {list(cluster_embeddings.keys())}")
            return
        
        cluster_embeddings = filtered_embeddings
        logger.info(f"Filtered to {len(cluster_embeddings)} videos")
    
    # Generate captions using BLIP with actual frames (not embeddings)
    captions = generate_captions_with_blip(
        frames_dir=args.frames_dir,
        embeddings_dict=cluster_embeddings,
        device=device
    )
    
    # Save results
    output_path = save_results(captions, args.output_dir)
    logger.info(f"Caption generation complete! Results saved to {output_path}")

if __name__ == "__main__":
    main()