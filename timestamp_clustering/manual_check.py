import os
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
import h5py

# Import your models
from train_clip import CLIPModel as FrameCLIPModel
from cluster_clip import CLIPModel as ClusterCLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_clip_model(model_path, embedding_dim, projection_dim, text_model_name, model_type="frame", device="cpu"):
    """Load a trained CLIP model."""
    if model_type == "frame":
        model = FrameCLIPModel(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            text_model_name=text_model_name
        )
    else:
        model = ClusterCLIPModel(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            text_model_name=text_model_name
        )
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        logger.info(f"Loaded {model_type} model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None

def load_video_frames(video_id, input_dir):
    """Load video frames from NPY or H5 file."""
    # Convert input_dir to Path object
    input_dir = Path(input_dir)
    
    # Clean video ID if needed
    clean_id = video_id.replace("video-", "") if video_id.startswith("video-") else video_id
    video_id_with_prefix = f"video-{clean_id}"
    
    # Try various potential file paths
    potential_paths = [
        # Standard paths
        input_dir / video_id_with_prefix / f"{video_id_with_prefix}_frames.npy",
        input_dir / video_id_with_prefix / f"{video_id_with_prefix}.npy",
        input_dir / video_id_with_prefix / f"{clean_id}_frames.npy",
        
        # Alternate paths
        input_dir / f"{video_id_with_prefix}_frames.npy",
        input_dir / f"{video_id_with_prefix}.npy",
        
        # H5 variants
        input_dir / video_id_with_prefix / f"{video_id_with_prefix}_frames.h5",
        input_dir / video_id_with_prefix / f"{video_id_with_prefix}.h5",
        input_dir / f"{video_id_with_prefix}_frames.h5",
        input_dir / f"{video_id_with_prefix}.h5"
    ]
    
    for path in potential_paths:
        if path.exists():
            logger.info(f"Loading frames from {path}")
            try:
                if path.suffix == '.npy':
                    frames = np.load(path)
                    return frames
                elif path.suffix == '.h5':
                    with h5py.File(path, 'r') as f:
                        # Check different potential dataset names
                        for ds_name in ['frames', 'images', 'data', clean_id, video_id_with_prefix]:
                            if ds_name in f:
                                frames = f[ds_name][()]
                                return frames
            except Exception as e:
                logger.error(f"Error loading frames from {path}: {e}")
    
    logger.error(f"Could not find frames for {video_id}. Please check paths.")
    return None

def load_frame_embeddings(video_id, embeddings_dir):
    """Load pre-computed frame embeddings for a video."""
    # Convert embeddings_dir to Path object
    embeddings_dir = Path(embeddings_dir)
    
    # Clean video ID if needed
    clean_id = video_id.replace("video-", "") if video_id.startswith("video-") else video_id
    video_id_with_prefix = f"video-{clean_id}"
    
    # Try various potential file paths
    potential_paths = [
        embeddings_dir / f"{video_id_with_prefix}_embeddings.npy",
        embeddings_dir / f"{clean_id}_embeddings.npy",
        embeddings_dir / f"{video_id_with_prefix}.npy",
        embeddings_dir / f"{clean_id}.npy"
    ]
    
    for path in potential_paths:
        if path.exists():
            try:
                embeddings = np.load(path)
                logger.info(f"Loaded frame embeddings for {video_id} with shape {embeddings.shape}")
                return embeddings
            except Exception as e:
                logger.error(f"Error loading frame embeddings from {path}: {e}")
    
    logger.error(f"Could not find frame embeddings for {video_id}")
    return None

def find_cluster_embeddings(video_id, cluster_dir):
    """Try to find cluster embedding files with more lenient matching."""
    # Convert cluster_dir to Path object
    cluster_dir = Path(cluster_dir)
    
    # List all files in the directory
    logger.info(f"Checking all files in {cluster_dir}")
    
    # Clean video ID if needed
    clean_id = video_id.replace("video-", "") if video_id.startswith("video-") else video_id
    video_id_with_prefix = f"video-{clean_id}"
    
    # Try both with and without "video-" prefix
    for vid_id in [video_id_with_prefix, clean_id]:
        cluster_vid_dir = cluster_dir / vid_id
        if cluster_vid_dir.exists():
            logger.info(f"Found video directory: {cluster_vid_dir}")
            
            # Check for cluster data (NPZ file)
            cluster_data_file = cluster_dir / f"{vid_id}_cluster_clip_data.npz"
            if cluster_data_file.exists():
                logger.info(f"Found cluster data file: {cluster_data_file}")
                try:
                    data = np.load(cluster_data_file, allow_pickle=True)
                    if 'cluster_embeddings' in data:
                        cluster_embeddings = data['cluster_embeddings'].item()  # It's a dictionary
                        logger.info(f"Loaded {len(cluster_embeddings)} cluster embeddings from {cluster_data_file}")
                        return cluster_embeddings, None, None  # No frames loaded from this method
                except Exception as e:
                    logger.error(f"Error loading cluster data from {cluster_data_file}: {e}")
            
            # List all files in directory to aid debugging
            files = list(cluster_vid_dir.glob("*"))
            if files:
                logger.info(f"Found {len(files)} files in {cluster_vid_dir}")
                for file in files[:10]:  # Show first 10 to avoid excess logging
                    logger.info(f"  - {file.name}")
                
                # Try to find embedding files with any pattern
                embedding_files = list(cluster_vid_dir.glob("*.npy"))
                if embedding_files:
                    logger.info(f"Found {len(embedding_files)} .npy files that could be embeddings")
                    
                    # If there are .npy files, we'll try to load them as cluster embeddings
                    cluster_embeddings = {}
                    
                    for embed_file in embedding_files:
                        try:
                            # Extract cluster ID from filename (various patterns)
                            if "cluster_" in embed_file.stem:
                                # Try pattern "cluster_X_embedding.npy"
                                parts = embed_file.stem.split("_")
                                if len(parts) > 1:
                                    cluster_id = parts[1]
                                    embedding = np.load(embed_file)
                                    cluster_embeddings[cluster_id] = embedding
                                    logger.info(f"Loaded embedding for cluster {cluster_id} from {embed_file.name}")
                            elif embed_file.stem.isdigit():
                                # Try pattern "0.npy", "1.npy", etc.
                                cluster_id = embed_file.stem
                                embedding = np.load(embed_file)
                                cluster_embeddings[cluster_id] = embedding
                                logger.info(f"Loaded embedding for cluster {cluster_id} from {embed_file.name}")
                        except Exception as e:
                            logger.error(f"Error loading embedding from {embed_file}: {e}")
                    
                    if cluster_embeddings:
                        logger.info(f"Successfully loaded {len(cluster_embeddings)} cluster embeddings")
                        return cluster_embeddings, None, None
            else:
                logger.info(f"No files found in {cluster_vid_dir}")
    
    logger.error(f"Could not find any cluster embeddings for {video_id}")
    return {}, None, None

def load_candidate_descriptions():
    """Load a diverse set of candidate descriptions."""
    # These are generic descriptions that could apply to many videos
    return [
        "A person is talking to the camera about various topics.",
        "A group of people are having a conversation outdoors.",
        "Someone is demonstrating how to cook a meal in the kitchen.",
        "A cat is playing with a toy on the floor.",
        "Children are playing in a playground with slides and swings.",
        "A car is driving down a scenic road surrounded by trees.",
        "A sporting event with players competing on a field.",
        "People are walking in a busy city street with tall buildings.",
        "A person is explaining something using visual aids or diagrams.",
        "Someone is showing how to use a particular product or device.",
        "A dog is running and playing fetch in a park.",
        "A time-lapse of a natural landscape showing changes over time.",
        "People are dancing at a party or celebration.",
        "A concert with musicians performing on stage.",
        "Someone is working on a DIY project or craft.",
        "A tutorial showing step-by-step instructions for a process.",
        "A wild animal in its natural habitat.",
        "People shopping in a store or marketplace.",
        "Someone is exercising or demonstrating workout techniques.",
        "A family gathering or celebration with multiple generations.",
        "An educational video explaining a concept with animations.",
        "A travel video showcasing landmarks and scenery.",
        "Someone is unboxing and reviewing a new product.",
        "People enjoying activities at a beach or pool.",
        "A video game being played with commentary.",
        "A news reporter delivering information on an event.",
        "A close-up demonstration of a detailed task or hobby.",
        "Someone telling a story or performing a monologue.",
        "People participating in a challenge or competition.",
        "A webinar or presentation with slides and a speaker."
    ]

def generate_descriptions(model, tokenizer, embeddings, candidate_descriptions, model_type="frame", device="cpu"):
    """Generate descriptions for frame or cluster embeddings using the CLIP model."""
    # Prepare candidate descriptions
    tokenized_candidates = tokenizer(
        candidate_descriptions,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    results = {}
    
    # Process each embedding
    if isinstance(embeddings, dict):
        # For clusters (key-value pairs of cluster_id: embedding)
        for cluster_id, embedding in tqdm(embeddings.items(), desc=f"Generating descriptions for {model_type}"):
            # Normalize embedding
            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding, dtype=torch.float32).to(device)
            embedding = embedding.unsqueeze(0)  # Add batch dimension
            
            # Process this embedding
            with torch.no_grad():
                # Project image embedding
                image_embedding = model.image_projection(embedding)
                image_embedding = torch.nn.functional.normalize(image_embedding, dim=1)
                
                # Get text embeddings for all candidates
                text_outputs = model.text_encoder(
                    input_ids=tokenized_candidates.input_ids,
                    attention_mask=tokenized_candidates.attention_mask
                )
                text_embeddings = model.text_projection(text_outputs.last_hidden_state[:, 0])
                text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=1)
                
                # Calculate similarities
                similarities = (100.0 * image_embedding @ text_embeddings.T).squeeze()
                
                # Get top 3 matches
                top_matches = torch.topk(similarities, 3)
                top_indices = top_matches.indices.cpu().numpy()
                top_scores = top_matches.values.cpu().numpy()
                
                # Add to results
                results[cluster_id] = {
                    "top_descriptions": [
                        {"description": candidate_descriptions[idx], "score": float(score)}
                        for idx, score in zip(top_indices, top_scores)
                    ]
                }
    else:
        # For frame embeddings (sequential array)
        # Select a sample of frames (e.g., every 10th frame)
        sample_indices = list(range(0, len(embeddings), 10))[:5]  # Up to 5 frames
        
        for idx in tqdm(sample_indices, desc=f"Generating descriptions for {model_type}"):
            embedding = embeddings[idx]
            
            # Normalize embedding
            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding, dtype=torch.float32).to(device)
            embedding = embedding.unsqueeze(0)  # Add batch dimension
            
            # Process this embedding
            with torch.no_grad():
                # Project image embedding
                image_embedding = model.image_projection(embedding)
                image_embedding = torch.nn.functional.normalize(image_embedding, dim=1)
                
                # Get text embeddings for all candidates
                text_outputs = model.text_encoder(
                    input_ids=tokenized_candidates.input_ids,
                    attention_mask=tokenized_candidates.attention_mask
                )
                text_embeddings = model.text_projection(text_outputs.last_hidden_state[:, 0])
                text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=1)
                
                # Calculate similarities
                similarities = (100.0 * image_embedding @ text_embeddings.T).squeeze()
                
                # Get top 3 matches
                top_matches = torch.topk(similarities, 3)
                top_indices = top_matches.indices.cpu().numpy()
                top_scores = top_matches.values.cpu().numpy()
                
                # Add to results
                results[idx] = {
                    "top_descriptions": [
                        {"description": candidate_descriptions[idx], "score": float(score)}
                        for idx, score in zip(top_indices, top_scores)
                    ]
                }
    
    return results

def save_text_results(video_id, frame_results, cluster_results, output_dir):
    """Save results as simple text files instead of HTML."""
    # Convert output_dir to Path object if it's not already
    output_dir = Path(output_dir)
    frame_output_path = output_dir / f"{video_id}_frame_descriptions.txt"
    cluster_output_path = output_dir / f"{video_id}_cluster_descriptions.txt"
    
    # Save frame results
    if frame_results:
        with open(frame_output_path, 'w') as f:
            f.write(f"FRAME-BASED CLIP MODEL DESCRIPTIONS FOR {video_id}\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, result in sorted(frame_results.items()):
                f.write(f"Frame {idx}:\n")
                f.write("-" * 40 + "\n")
                for i, desc_info in enumerate(result["top_descriptions"]):
                    f.write(f"{i+1}. {desc_info['description']} (Score: {desc_info['score']:.2f})\n")
                f.write("\n")
        
        logger.info(f"Saved frame descriptions to {frame_output_path}")
    
    # Save cluster results
    if cluster_results:
        with open(cluster_output_path, 'w') as f:
            f.write(f"CLUSTER-BASED CLIP MODEL DESCRIPTIONS FOR {video_id}\n")
            f.write("=" * 80 + "\n\n")
            
            for cluster_id, result in sorted(cluster_results.items()):
                f.write(f"Cluster {cluster_id}:\n")
                f.write("-" * 40 + "\n")
                for i, desc_info in enumerate(result["top_descriptions"]):
                    f.write(f"{i+1}. {desc_info['description']} (Score: {desc_info['score']:.2f})\n")
                f.write("\n")
        
        logger.info(f"Saved cluster descriptions to {cluster_output_path}")
    
    return frame_output_path, cluster_output_path

def main():
    parser = argparse.ArgumentParser(description="Generate descriptions using trained CLIP models")
    
    # Input directories
    parser.add_argument("--frame_model_path", type=str, required=True,
                       help="Path to the trained frame-based CLIP model")
    parser.add_argument("--cluster_model_path", type=str, required=True,
                       help="Path to the trained cluster-based CLIP model")
    parser.add_argument("--frames_dir", type=str, default="/Users/akshat/Developer/Vid_Attention/data_extraction/data/QuerYD/processed_frames",
                       help="Directory containing video frames")
    parser.add_argument("--embeddings_dir", type=str, default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/swin_embeddings",
                       help="Directory containing pre-computed frame embeddings")
    parser.add_argument("--cluster_dir", type=str, default="/Users/akshat/Developer/Vid_Attention/timestamp_clustering/cluster_embeddings",
                       help="Directory containing cluster information")
    parser.add_argument("--output_dir", type=str, default="clip_description_results",
                       help="Directory to save results")
    
    # Model parameters
    parser.add_argument("--embedding_dim", type=int, default=768,
                       help="Dimension of the input embeddings")
    parser.add_argument("--projection_dim", type=int, default=256,
                       help="Dimension of the joint projection space")
    parser.add_argument("--text_model", type=str, default="distilbert-base-uncased",
                       help="Hugging Face transformer model for text")
    
    # Video selection
    parser.add_argument("--video_ids", type=str, nargs="+",
                       help="Specific video IDs to process")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else 
                                                    "mps" if torch.backends.mps.is_available() else "cpu",
                       help="Device for inference (cuda, mps, or cpu)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    frame_model = load_clip_model(
        model_path=args.frame_model_path,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        text_model_name=args.text_model,
        model_type="frame",
        device=device
    )
    
    cluster_model = load_clip_model(
        model_path=args.cluster_model_path,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        text_model_name=args.text_model,
        model_type="cluster",
        device=device
    )
    
    if frame_model is None or cluster_model is None:
        logger.error("Failed to load models. Exiting.")
        return
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    
    # Get candidate descriptions
    candidate_descriptions = load_candidate_descriptions()
    logger.info(f"Loaded {len(candidate_descriptions)} candidate descriptions")
    
    # If no video IDs provided, suggest some examples
    if not args.video_ids:
        logger.error("No video IDs provided. Please specify videos using --video_ids argument.")
        logger.info("Example format: --video_ids video-G4Sn91t1V4g video-JZIerGNMtnk")
        return
    
    # Process each video
    for video_id in args.video_ids:
        logger.info(f"Processing video: {video_id}")
        
        # 1. Load frame data
        frames = load_video_frames(video_id, args.frames_dir)
        frame_embeddings = load_frame_embeddings(video_id, args.embeddings_dir)
        
        # 2. Load cluster data - using more robust method
        cluster_embeddings, _, _ = find_cluster_embeddings(video_id, args.cluster_dir)
        
        # Check if we have enough data to proceed
        if frame_embeddings is None and not cluster_embeddings:
            logger.warning(f"No frame or cluster data found for {video_id}, skipping.")
            continue
        
        # 3. Generate descriptions with frame model
        frame_results = {}
        if frame_embeddings is not None:
            # Select sample frames (every 10th frame, up to 5)
            frame_results = generate_descriptions(
                model=frame_model,
                tokenizer=tokenizer,
                embeddings=frame_embeddings,
                candidate_descriptions=candidate_descriptions,
                model_type="frame",
                device=device
            )
        
        # 4. Generate descriptions with cluster model
        cluster_results = {}
        if cluster_embeddings:
            cluster_results = generate_descriptions(
                model=cluster_model,
                tokenizer=tokenizer,
                embeddings=cluster_embeddings,
                candidate_descriptions=candidate_descriptions,
                model_type="cluster",
                device=device
            )
        
        # 5. Save results as text files
        if frame_results or cluster_results:
            frame_path, cluster_path = save_text_results(
                video_id=video_id,
                frame_results=frame_results,
                cluster_results=cluster_results,
                output_dir=output_dir
            )
        else:
            logger.warning(f"No descriptions generated for {video_id}")
    
    logger.info("Description generation complete!")

if __name__ == "__main__":
    main()