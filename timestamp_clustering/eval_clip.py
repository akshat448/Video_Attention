import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import your model and dataset classes
from train_clip import CLIPModel, CLIPDataset

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, embedding_dim, projection_dim, text_model_name, device):
    """Load a trained CLIP model."""
    model = CLIPModel(
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        text_model_name=text_model_name
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    logger.info(f"Final train loss: {checkpoint.get('train_loss', 'unknown')}")
    logger.info(f"Final validation loss: {checkpoint.get('val_loss', 'unknown')}")
    
    return model

def compute_embeddings(model, dataloader, device, max_samples=None):
    """Compute embeddings for all samples in the dataloader."""
    model.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []
    all_captions = []
    all_video_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            # Move batch to device
            image_embeddings = batch["embedding"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            image_emb, text_emb, _ = model(
                image_embeddings, input_ids, attention_mask
            )
            
            all_image_embeddings.append(image_emb.cpu())
            all_text_embeddings.append(text_emb.cpu())
            all_captions.extend(batch["caption"])
            all_video_ids.extend(batch["video_id"])
            
            if max_samples and len(all_captions) >= max_samples:
                all_image_embeddings[0] = all_image_embeddings[0][:max_samples - len(all_captions) + len(batch["caption"])]
                all_text_embeddings[0] = all_text_embeddings[0][:max_samples - len(all_captions) + len(batch["caption"])]
                all_captions = all_captions[:max_samples]
                all_video_ids = all_video_ids[:max_samples]
                break
    
    # Concatenate all embeddings
    image_embeddings = torch.cat(all_image_embeddings)
    text_embeddings = torch.cat(all_text_embeddings)
    
    return {
        "image_embeddings": image_embeddings,
        "text_embeddings": text_embeddings,
        "captions": all_captions,
        "video_ids": all_video_ids
    }

def calculate_retrieval_metrics(image_embeddings, text_embeddings):
    """Calculate comprehensive retrieval metrics."""
    # Compute similarity matrix
    similarity = image_embeddings @ text_embeddings.T
    
    # Calculate retrieval metrics
    image_to_text_ranks = []
    text_to_image_ranks = []
    
    for i in range(similarity.shape[0]):
        # Image to text retrieval
        sim_i2t = similarity[i]
        sorted_indices = torch.argsort(sim_i2t, descending=True)
        rank = torch.where(sorted_indices == i)[0].item() + 1
        image_to_text_ranks.append(rank)
        
        # Text to image retrieval
        sim_t2i = similarity[:, i]
        sorted_indices = torch.argsort(sim_t2i, descending=True)
        rank = torch.where(sorted_indices == i)[0].item() + 1
        text_to_image_ranks.append(rank)
    
    # Calculate basic statistics
    metrics = {
        "i2t": {
            "mean_rank": np.mean(image_to_text_ranks),
            "median_rank": np.median(image_to_text_ranks),
            "mrr": np.mean([1.0/rank for rank in image_to_text_ranks]),  # Mean Reciprocal Rank
        },
        "t2i": {
            "mean_rank": np.mean(text_to_image_ranks),
            "median_rank": np.median(text_to_image_ranks),
            "mrr": np.mean([1.0/rank for rank in text_to_image_ranks]),
        }
    }
    
    # Calculate recall@k metrics
    for k in [1, 5, 10, 50]:
        metrics["i2t"][f"R@{k}"] = 100 * sum(r <= k for r in image_to_text_ranks) / len(image_to_text_ranks)
        metrics["t2i"][f"R@{k}"] = 100 * sum(r <= k for r in text_to_image_ranks) / len(text_to_image_ranks)
    
    # Calculate average metrics
    average = {}
    for k in metrics["i2t"]:
        average[k] = (metrics["i2t"][k] + metrics["t2i"][k]) / 2
    metrics["average"] = average
    
    return metrics, image_to_text_ranks, text_to_image_ranks, similarity

def plot_similarity_matrix(similarity, captions, output_dir, k=10):
    """Plot a heatmap of the k x k similarity matrix for random samples."""
    n = similarity.shape[0]
    indices = np.random.choice(n, min(k, n), replace=False)
    
    plt.figure(figsize=(12, 10))
    sub_similarity = similarity[indices][:, indices].cpu().numpy()
    
    shortened_captions = [cap[:30] + "..." if len(cap) > 30 else cap for cap in np.array(captions)[indices]]
    
    sns.heatmap(sub_similarity, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=shortened_captions, yticklabels=shortened_captions)
    
    plt.title("Cosine Similarity Matrix")
    plt.xlabel("Captions")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_matrix.png"), dpi=150)
    plt.close()

def plot_embedding_visualization(image_embeddings, text_embeddings, captions, video_ids, output_dir, max_points=500):
    """Create t-SNE visualization of the embedding space."""
    # Sample points if too many
    n = image_embeddings.shape[0]
    indices = np.random.choice(n, min(max_points, n), replace=False)
    
    # Combine image and text embeddings
    combined_embeddings = torch.cat([
        image_embeddings[indices], 
        text_embeddings[indices]
    ]).numpy()
    
    # Create labels
    labels = ["Image"] * len(indices) + ["Text"] * len(indices)
    
    # Apply t-SNE
    logger.info("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(indices)//5))
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    
    # Split back to image and text
    image_2d = embeddings_2d[:len(indices)]
    text_2d = embeddings_2d[len(indices):]
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot image and text points
    plt.scatter(image_2d[:, 0], image_2d[:, 1], c='blue', alpha=0.5, label="Images")
    plt.scatter(text_2d[:, 0], text_2d[:, 1], c='red', alpha=0.5, label="Texts")
    
    # Draw lines between matched pairs
    for i in range(len(indices)):
        plt.plot([image_2d[i, 0], text_2d[i, 0]], 
                 [image_2d[i, 1], text_2d[i, 1]], 
                 'k-', alpha=0.1)
    
    plt.legend()
    plt.title("t-SNE Visualization of Image and Text Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embeddings_tsne.png"), dpi=150)
    plt.close()
    
    # Also create a visualization colored by video
    unique_videos = sorted(set(video_ids))
    video_to_color = {vid: i for i, vid in enumerate(unique_videos)}
    
    plt.figure(figsize=(12, 10))
    for vid in unique_videos:
        vid_indices = [i for i, v in enumerate(np.array(video_ids)[indices]) if v == vid]
        
        if not vid_indices:
            continue
            
        color = plt.cm.tab20(video_to_color[vid] % 20)
        plt.scatter(image_2d[vid_indices, 0], image_2d[vid_indices, 1], color=color, alpha=0.7, label=f"{vid}")
        plt.scatter(text_2d[vid_indices, 0], text_2d[vid_indices, 1], color=color, alpha=0.7, marker='x')
    
    plt.title("t-SNE Visualization by Video")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embeddings_by_video.png"), dpi=150)
    plt.close()

def plot_retrieval_metrics(metrics, output_dir):
    """Plot retrieval metrics for visualization."""
    # Plot Recall@K
    plt.figure(figsize=(10, 6))
    k_values = [1, 5, 10, 50]
    plt.plot(k_values, [metrics["i2t"][f"R@{k}"] for k in k_values], 'o-', label="Image to Text")
    plt.plot(k_values, [metrics["t2i"][f"R@{k}"] for k in k_values], 'o-', label="Text to Image")
    plt.plot(k_values, [metrics["average"][f"R@{k}"] for k in k_values], 'o-', label="Average")
    
    plt.xlabel("k")
    plt.ylabel("Recall@k (%)")
    plt.title("Recall@k Performance")
    plt.legend()
    plt.grid(True)
    plt.xticks(k_values)
    plt.savefig(os.path.join(output_dir, "recall_at_k.png"), dpi=150)
    plt.close()
    
    # Bar plot of overall metrics
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ["mean_rank", "median_rank", "mrr"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    plt.bar(x - width, [metrics["i2t"][m] for m in metrics_to_plot], width, label="Image to Text")
    plt.bar(x, [metrics["t2i"][m] for m in metrics_to_plot], width, label="Text to Image")
    plt.bar(x + width, [metrics["average"][m] for m in metrics_to_plot], width, label="Average")
    
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title("Retrieval Performance Metrics")
    plt.xticks(x, metrics_to_plot)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "retrieval_metrics.png"), dpi=150)
    plt.close()

def create_confusion_matrix(similarity, k=5, threshold=0.9):
    """Create a retrieval confusion matrix at Recall@k."""
    n = similarity.shape[0]
    
    # Get top-k indices for each query
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    for i in range(n):
        # Image to text
        sim_i2t = similarity[i]
        sorted_indices = torch.argsort(sim_i2t, descending=True)
        top_k_indices = sorted_indices[:k].cpu().numpy()
        
        # Count metrics
        if i in top_k_indices:
            true_positives += 1
        else:
            false_negatives += 1
        
        false_positives += sum(1 for idx in top_k_indices if idx != i)
        true_negatives += n - k - (0 if i in top_k_indices else 1)
    
    cm = np.array([
        [true_positives, false_positives],
        [false_negatives, true_negatives]
    ])
    
    return cm

def plot_confusion_matrix(cm, output_dir, k=5):
    """Plot the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Positive", "Negative"],
                yticklabels=["Positive", "Negative"])
    
    plt.title(f"Confusion Matrix for Retrieval@{k}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_k{k}.png"), dpi=150)
    plt.close()

def analyze_video_retrieval_by_distribution(video_ids, ranks, output_dir):
    """Analyze retrieval performance by video distribution."""
    video_ranks = {}
    for vid, rank in zip(video_ids, ranks):
        if vid not in video_ranks:
            video_ranks[vid] = []
        video_ranks[vid].append(rank)
    
    # Calculate mean rank for each video
    video_mean_ranks = {vid: np.mean(ranks) for vid, ranks in video_ranks.items()}
    
    # Sort videos by mean rank
    sorted_videos = sorted(video_mean_ranks.items(), key=lambda x: x[1])
    
    # Plot distribution of mean ranks
    plt.figure(figsize=(14, 6))
    x = np.arange(len(sorted_videos))
    plt.bar(x, [rank for _, rank in sorted_videos])
    plt.xticks(x, [vid for vid, _ in sorted_videos], rotation=90)
    plt.xlabel("Video ID")
    plt.ylabel("Mean Rank")
    plt.title("Mean Rank Distribution by Video")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_rank_by_video.png"), dpi=150)
    plt.close()
    
    # Plot histogram of all ranks
    plt.figure(figsize=(10, 6))
    plt.hist(ranks, bins=20, alpha=0.7)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Distribution of Retrieval Ranks")
    plt.axvline(np.mean(ranks), color='r', linestyle='dashed', linewidth=1, label=f"Mean: {np.mean(ranks):.2f}")
    plt.axvline(np.median(ranks), color='g', linestyle='dashed', linewidth=1, label=f"Median: {np.median(ranks):.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rank_distribution.png"), dpi=150)
    plt.close()
    
    return video_mean_ranks

def find_hardest_examples(similarity, captions, video_ids, k=5):
    """Find the hardest examples (worst ranking) for analysis."""
    n = similarity.shape[0]
    
    # Compute ranks
    image_to_text_ranks = []
    text_to_image_ranks = []
    
    for i in range(n):
        # Image to text
        sim_i2t = similarity[i]
        sorted_indices = torch.argsort(sim_i2t, descending=True)
        rank = torch.where(sorted_indices == i)[0].item() + 1
        image_to_text_ranks.append(rank)
        
        # Text to image
        sim_t2i = similarity[:, i]
        sorted_indices = torch.argsort(sim_t2i, descending=True)
        rank = torch.where(sorted_indices == i)[0].item() + 1
        text_to_image_ranks.append(rank)
    
    # Find worst examples
    worst_i2t = np.argsort(image_to_text_ranks)[-k:][::-1]
    worst_t2i = np.argsort(text_to_image_ranks)[-k:][::-1]
    
    # Create worst examples list
    worst_examples = {
        "image_to_text": [
            {
                "index": int(idx),
                "caption": captions[idx],
                "video_id": video_ids[idx],
                "rank": int(image_to_text_ranks[idx])
            }
            for idx in worst_i2t
        ],
        "text_to_image": [
            {
                "index": int(idx),
                "caption": captions[idx],
                "video_id": video_ids[idx],
                "rank": int(text_to_image_ranks[idx])
            }
            for idx in worst_t2i
        ]
    }
    
    return worst_examples

def find_best_examples(similarity, captions, video_ids, k=5):
    """Find the best examples (best ranking) for analysis."""
    n = similarity.shape[0]
    
    # Compute ranks
    image_to_text_ranks = []
    text_to_image_ranks = []
    
    for i in range(n):
        # Image to text
        sim_i2t = similarity[i]
        sorted_indices = torch.argsort(sim_i2t, descending=True)
        rank = torch.where(sorted_indices == i)[0].item() + 1
        image_to_text_ranks.append(rank)
        
        # Text to image
        sim_t2i = similarity[:, i]
        sorted_indices = torch.argsort(sim_t2i, descending=True)
        rank = torch.where(sorted_indices == i)[0].item() + 1
        text_to_image_ranks.append(rank)
    
    # Find best examples (rank = 1)
    best_i2t = np.where(np.array(image_to_text_ranks) == 1)[0][:k]
    best_t2i = np.where(np.array(text_to_image_ranks) == 1)[0][:k]
    
    # Create best examples list
    best_examples = {
        "image_to_text": [
            {
                "index": int(idx),
                "caption": captions[idx],
                "video_id": video_ids[idx],
                "rank": 1
            }
            for idx in best_i2t
        ],
        "text_to_image": [
            {
                "index": int(idx),
                "caption": captions[idx],
                "video_id": video_ids[idx],
                "rank": 1
            }
            for idx in best_t2i
        ]
    }
    
    return best_examples

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP model on video caption pairs")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="cluster_clip_data",
                       help="Directory containing *_clip_data.npz files")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="cluster_clip_evaluation",
                       help="Directory to save evaluation results")
    
    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=768,
                       help="Dimension of the SWIN image embeddings")
    parser.add_argument("--projection_dim", type=int, default=256,
                       help="Dimension of the joint projection space")
    parser.add_argument("--text_model", type=str, default="distilbert-base-uncased",
                       help="Hugging Face transformer model for text")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else 
                                                    "mps" if torch.backends.mps.is_available() else "cpu",
                       help="Device for inference (cuda, mps, or cpu)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    
    # Create dataset
    dataset = CLIPDataset(args.data_dir, tokenizer)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Load model
    model = load_model(
        model_path=args.model_path,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        text_model_name=args.text_model,
        device=device
    )
    
    # Compute embeddings
    logger.info("Computing embeddings...")
    embeddings = compute_embeddings(model, dataloader, device, args.max_samples)
    
    # Calculate metrics
    logger.info("Calculating retrieval metrics...")
    metrics, i2t_ranks, t2i_ranks, similarity = calculate_retrieval_metrics(
        embeddings["image_embeddings"], 
        embeddings["text_embeddings"]
    )
    
    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary of metrics
    logger.info("=== Retrieval Metrics ===")
    logger.info(f"Image to Text - R@1: {metrics['i2t']['R@1']:.2f}%, R@5: {metrics['i2t']['R@5']:.2f}%, "
               f"R@10: {metrics['i2t']['R@10']:.2f}%, MRR: {metrics['i2t']['mrr']:.4f}")
    logger.info(f"Text to Image - R@1: {metrics['t2i']['R@1']:.2f}%, R@5: {metrics['t2i']['R@5']:.2f}%, "
               f"R@10: {metrics['t2i']['R@10']:.2f}%, MRR: {metrics['t2i']['mrr']:.4f}")
    logger.info(f"Average - R@1: {metrics['average']['R@1']:.2f}%, R@5: {metrics['average']['R@5']:.2f}%, "
               f"R@10: {metrics['average']['R@10']:.2f}%, MRR: {metrics['average']['mrr']:.4f}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Plot retrieval metrics
    plot_retrieval_metrics(metrics, args.output_dir)
    
    # 2. Plot similarity matrix
    plot_similarity_matrix(similarity, embeddings["captions"], args.output_dir)
    
    # 3. Create t-SNE visualization
    plot_embedding_visualization(
        embeddings["image_embeddings"],
        embeddings["text_embeddings"],
        embeddings["captions"],
        embeddings["video_ids"],
        args.output_dir
    )
    
    # 4. Create confusion matrix
    for k in [1, 5, 10]:
        cm = create_confusion_matrix(similarity, k=k)
        plot_confusion_matrix(cm, args.output_dir, k=k)
    
    # 5. Analyze by video
    logger.info("Analyzing performance by video...")
    i2t_by_video = analyze_video_retrieval_by_distribution(
        embeddings["video_ids"], i2t_ranks, args.output_dir)
    
    # 6. Find best and worst examples
    logger.info("Finding best and worst examples...")
    worst_examples = find_hardest_examples(
        similarity, embeddings["captions"], embeddings["video_ids"])
    
    best_examples = find_best_examples(
        similarity, embeddings["captions"], embeddings["video_ids"], k=10)
    
    # Save examples
    with open(os.path.join(args.output_dir, "worst_examples.json"), 'w') as f:
        json.dump(worst_examples, f, indent=2)
        
    with open(os.path.join(args.output_dir, "best_examples.json"), 'w') as f:
        json.dump(best_examples, f, indent=2)
    
    logger.info(f"All evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    main()