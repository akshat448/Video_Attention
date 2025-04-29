import torch
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_cluster_data(npz_path):
    """Load cluster data from the pre-generated NPZ file."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        embeddings = data['embeddings']
        true_captions = data['captions']
        cluster_ids = data['cluster_ids']
        
        # Create a mapping from cluster_id to embedding and caption
        cluster_data = {}
        for i, cluster_id in enumerate(cluster_ids):
            cluster_data[str(cluster_id)] = {
                'embedding': embeddings[i],
                'true_caption': true_captions[i]
            }
        
        logger.info(f"Loaded {len(cluster_data)} clusters from {npz_path}")
        return cluster_data
    except Exception as e:
        logger.error(f"Error loading cluster data from {npz_path}: {e}")
        return None

def load_representative_frames(video_id, cluster_id, frames_dir):
    """Load representative frames for visualization (optional)."""
    cluster_dir = Path(frames_dir) / video_id / f"cluster_{cluster_id}"
    
    if not cluster_dir.exists():
        logger.warning(f"No cluster folder found for {video_id}, cluster {cluster_id}")
        return None
    
    # Get a sample of frames from this cluster
    frame_files = sorted(list(cluster_dir.glob("*.png")) or list(cluster_dir.glob("*.jpg")))
    
    if not frame_files:
        return None
    
    # Take up to 3 frames (beginning, middle, end)
    sample_indices = [0]
    if len(frame_files) > 2:
        sample_indices.append(len(frame_files) // 2)
    if len(frame_files) > 1:
        sample_indices.append(len(frame_files) - 1)
    
    return [frame_files[i] for i in sample_indices]

def generate_captions_with_blip(video_id, cluster_data, frames_dir, device="cpu"):
    """Generate captions for clusters using BLIP."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    for cluster_id, data in tqdm(cluster_data.items(), desc=f"Generating captions for {video_id}"):
        # Load representative frames for this cluster
        frames = load_representative_frames(video_id, cluster_id, frames_dir)
        
        if not frames:
            data['generated_caption'] = "No frames available"
            continue
        
        # Generate captions for each frame
        frame_captions = []
        for frame_path in frames:
            try:
                image = Image.open(frame_path).convert('RGB')
                inputs = processor(image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_length=50)
                caption = processor.decode(out[0], skip_special_tokens=True)
                frame_captions.append(caption)
            except Exception as e:
                logger.error(f"Error generating caption for {frame_path}: {e}")
        
        # Combine frame captions
        if frame_captions:
            data['generated_caption'] = "This cluster shows " + "; also ".join(frame_captions)
        else:
            data['generated_caption'] = "Failed to generate captions"
    
    return cluster_data

def calculate_semantic_similarity(cluster_data):
    """Calculate semantic similarity between generated and ground truth captions."""
    # Load a sentence transformer model for semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    results = {}
    
    for cluster_id, data in cluster_data.items():
        if 'generated_caption' not in data or 'true_caption' not in data:
            continue
        
        generated = data['generated_caption']
        true_caption = data['true_caption']
        
        # Skip if either caption is empty
        if not generated or not true_caption:
            continue
        
        # Calculate embeddings
        gen_embedding = model.encode(generated, convert_to_tensor=True)
        true_embedding = model.encode(true_caption, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(gen_embedding, true_embedding).item()
        
        results[cluster_id] = {
            'generated': generated,
            'ground_truth': true_caption,
            'similarity': similarity
        }
    
    return results

def visualize_results(similarity_results, output_dir, video_id):
    """Visualize similarity results as heatmap and histogram."""
    if not similarity_results:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract similarities for visualization
    similarities = [data['similarity'] for data in similarity_results.values()]
    cluster_ids = list(similarity_results.keys())
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=10, alpha=0.7, color='skyblue')
    plt.axvline(x=np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
    plt.title(f'Caption Similarity Distribution for {video_id}')
    plt.xlabel('Semantic Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    hist_path = output_dir / f"{video_id}_similarity_histogram.png"
    plt.savefig(hist_path)
    plt.close()
    
    # Create heatmap
    similarity_matrix = np.zeros((len(cluster_ids), 1))
    for i, cluster_id in enumerate(cluster_ids):
        similarity_matrix[i, 0] = similarity_results[cluster_id]['similarity']
    
    plt.figure(figsize=(8, len(cluster_ids) * 0.4 + 2))
    sns.heatmap(similarity_matrix, annot=True, fmt=".3f", 
                yticklabels=cluster_ids, xticklabels=['Similarity'],
                cmap='viridis', vmin=0, vmax=1)
    plt.title(f'Caption Similarity per Cluster for {video_id}')
    plt.tight_layout()
    
    heatmap_path = output_dir / f"{video_id}_similarity_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    
    logger.info(f"Saved visualizations to {output_dir}")

def save_results(video_id, similarity_results, output_dir):
    """Save evaluation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save detailed results as JSON
    json_path = output_dir / f"{video_id}_caption_evaluation.json"
    with open(json_path, 'w') as f:
        json.dump(similarity_results, f, indent=2)
    
    # Create a summary report
    summary_path = output_dir / f"{video_id}_evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"CAPTION EVALUATION SUMMARY FOR {video_id}\n")
        f.write("=" * 80 + "\n\n")
        
        similarities = [data['similarity'] for data in similarity_results.values()]
        avg_similarity = np.mean(similarities) if similarities else 0
        
        f.write(f"Number of clusters evaluated: {len(similarity_results)}\n")
        f.write(f"Average semantic similarity: {avg_similarity:.4f}\n\n")
        
        f.write("DETAILED RESULTS BY CLUSTER\n")
        f.write("-" * 80 + "\n\n")
        
        for cluster_id, data in sorted(similarity_results.items()):
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"  Similarity: {data['similarity']:.4f}\n")
            f.write(f"  Generated: {data['generated']}\n")
            f.write(f"  Ground Truth: {data['ground_truth']}\n\n")
    
    # Create a CSV summary for easier analysis
    df_data = []
    for cluster_id, data in similarity_results.items():
        df_data.append({
            'video_id': video_id,
            'cluster_id': cluster_id,
            'similarity': data['similarity'],
            'generated_caption': data['generated'],
            'ground_truth_caption': data['ground_truth']
        })
    
    df = pd.DataFrame(df_data)
    csv_path = output_dir / f"{video_id}_evaluation_data.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved evaluation results to {output_dir}")
    
    return {
        'video_id': video_id,
        'avg_similarity': avg_similarity,
        'num_clusters': len(similarity_results),
        'json_path': json_path,
        'summary_path': summary_path,
        'csv_path': csv_path
    }

def process_video(video_id, clip_data_dir, frames_dir, output_dir, device):
    """Process a single video for caption evaluation."""
    logger.info(f"Processing video: {video_id}")
    
    # 1. Load pre-mapped cluster data from NPZ file
    npz_path = Path(clip_data_dir) / f"{video_id}_cluster_clip_data.npz"
    if not npz_path.exists():
        logger.error(f"No cluster data file found for {video_id} at {npz_path}")
        return None
    
    cluster_data = load_cluster_data(npz_path)
    if not cluster_data:
        logger.error(f"Failed to load cluster data for {video_id}")
        return None
    
    # 2. Generate captions with BLIP
    cluster_data = generate_captions_with_blip(video_id, cluster_data, frames_dir, device)
    
    # 3. Calculate semantic similarity
    similarity_results = calculate_semantic_similarity(cluster_data)
    
    if not similarity_results:
        logger.error(f"No valid similarity results for {video_id}")
        return None
    
    # 4. Visualize results
    visualize_results(similarity_results, output_dir, video_id)
    
    # 5. Save results
    return save_results(video_id, similarity_results, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Evaluate BLIP-generated captions against ground truth captions")
    
    parser.add_argument("--clip_data_dir", type=str, 
                       default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/cluster_clip_data",
                       help="Directory containing pre-mapped cluster data (.npz files)")
    parser.add_argument("--frames_dir", type=str, 
                       default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/clusters",
                       help="Directory containing organized frames by cluster")
    parser.add_argument("--output_dir", type=str, 
                       default="caption_evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else 
                                "mps" if torch.backends.mps.is_available() else "cpu",
                       help="Device for inference (cuda, mps, or cpu)")
    parser.add_argument("--video_ids", type=str, nargs="+", required=False,
                       help="Specific video IDs to process")
    parser.add_argument("--num_videos", type=int, default=None,
                       help="Number of videos to process (ignored if --video_ids is provided)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine video IDs to process
    if args.video_ids:
        video_ids = args.video_ids
    else:
        # Get video IDs from NPZ files
        clip_data_dir = Path(args.clip_data_dir)
        npz_files = list(clip_data_dir.glob("*_cluster_clip_data.npz"))
        video_ids = [f.stem.replace("_cluster_clip_data", "") for f in npz_files]
        
        if args.num_videos:
            video_ids = video_ids[:args.num_videos]
    
    if not video_ids:
        logger.error("No videos found to process.")
        return
    
    logger.info(f"Will process {len(video_ids)} videos: {', '.join(video_ids[:5])}..." if len(video_ids) > 5 else ', '.join(video_ids))
    
    # Process each video
    all_results = []
    for video_id in video_ids:
        result = process_video(video_id, args.clip_data_dir, args.frames_dir, args.output_dir, args.device)
        if result:
            all_results.append(result)
    
    # Create overall summary
    if all_results:
        # Compute average similarity across all videos
        avg_similarities = [r['avg_similarity'] for r in all_results]
        overall_avg = np.mean(avg_similarities)
        
        summary_path = Path(args.output_dir) / "overall_evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("OVERALL CAPTION EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Videos evaluated: {len(all_results)}\n")
            f.write(f"Overall average similarity: {overall_avg:.4f}\n\n")
            
            f.write("RESULTS BY VIDEO\n")
            f.write("-" * 80 + "\n\n")
            
            for result in sorted(all_results, key=lambda x: x['video_id']):
                f.write(f"Video: {result['video_id']}\n")
                f.write(f"  Average similarity: {result['avg_similarity']:.4f}\n")
                f.write(f"  Clusters evaluated: {result['num_clusters']}\n")
                f.write(f"  Detailed results: {result['summary_path'].name}\n\n")
        
        logger.info(f"Overall evaluation complete. Results saved to {args.output_dir}")
    else:
        logger.warning("No videos were successfully evaluated.")

if __name__ == "__main__":
    main()