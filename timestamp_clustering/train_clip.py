import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import transformers
from transformers import AutoTokenizer, AutoModel
import json
import random

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIPDataset(Dataset):
    """Dataset for loading and preprocessing CLIP training data from NPZ files."""
    
    def __init__(self, data_dir, tokenizer, max_length=77):
        """
        Args:
            data_dir: Directory containing video_clip_data.npz files
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length for tokenizer
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clip_files = list(self.data_dir.glob("*_clip_data.npz"))
        
        logger.info(f"Found {len(self.clip_files)} clip data files in {data_dir}")
        
        # Collect total samples and sample mapping for random access
        self.sample_map = []
        self.total_samples = 0
        
        for clip_file in self.clip_files:
            try:
                clip_data = np.load(clip_file)
                num_samples = len(clip_data["captions"])
                
                for i in range(num_samples):
                    self.sample_map.append((clip_file, i))
                
                self.total_samples += num_samples
                logger.debug(f"Added {num_samples} samples from {clip_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {clip_file}: {e}")
        
        logger.info(f"Dataset created with {self.total_samples} samples")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        clip_file, sample_idx = self.sample_map[idx]
        
        try:
            # Load the NPZ file
            clip_data = np.load(clip_file)
            
            # Get the embedding and caption
            embedding = clip_data["embeddings"][sample_idx]
            caption = clip_data["captions"][sample_idx]
            
            # Normalize embeddings (crucial for contrastive learning)
            embedding = embedding / np.linalg.norm(embedding, axis=0, keepdims=True)
            
            # Tokenize caption
            tokenized = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Convert embedding to tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            
            return {
                "embedding": embedding_tensor,
                "input_ids": tokenized.input_ids.squeeze(),
                "attention_mask": tokenized.attention_mask.squeeze(),
                "caption": caption,
                "video_id": clip_file.stem.split("_clip_data")[0]
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx} ({clip_file}, {sample_idx}): {e}")
            # Return a dummy sample in case of error
            return self.__getitem__(random.randint(0, len(self) - 1))

class CLIPModel(nn.Module):
    """Custom CLIP model for image-text contrastive learning.
    
    Since we already have embeddings, we'll use only a projection layer for images,
    and a transformer for text, followed by projection.
    """
    
    def __init__(self, embedding_dim, projection_dim, text_model_name="distilbert-base-uncased"):
        super().__init__()
        self.projection_dim = projection_dim
        
        # Image projection (from existing embeddings to projection space)
        self.image_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # Text projection
        text_hidden_size = self.text_encoder.config.hidden_size
        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Initialize weights
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, image_embeddings, input_ids, attention_mask):
        # Project image embeddings to projection space
        image_embeddings = self.image_projection(image_embeddings)
        
        # Get text embeddings from transformer
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeddings = text_outputs.last_hidden_state[:, 0]  # Use [CLS] token
        
        # Project text embeddings to projection space
        text_embeddings = self.text_projection(text_embeddings)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        return image_embeddings, text_embeddings, self.logit_scale.exp()

def contrastive_loss(image_embeddings, text_embeddings, logit_scale):
    """Compute contrastive loss between image and text embeddings."""
    # Compute cosine similarity
    logits = logit_scale * image_embeddings @ text_embeddings.T
    
    # Calculate loss in both directions
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    loss = (loss_i + loss_t) / 2
    
    return loss

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        # Move batch to device
        image_embeddings = batch["embedding"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass
        image_embeddings, text_embeddings, logit_scale = model(
            image_embeddings, input_ids, attention_mask
        )
        
        # Compute loss
        loss = contrastive_loss(image_embeddings, text_embeddings, logit_scale)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            image_embeddings = batch["embedding"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            image_embeddings, text_embeddings, logit_scale = model(
                image_embeddings, input_ids, attention_mask
            )
            
            # Compute loss
            loss = contrastive_loss(image_embeddings, text_embeddings, logit_scale)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)

def calculate_retrieval_metrics(model, dataloader, device):
    """Calculate I2T and T2I retrieval metrics."""
    model.eval()
    
    # Collect all embeddings and metadata
    all_image_embeddings = []
    all_text_embeddings = []
    all_videos = []
    
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
            all_videos.extend(batch["video_id"])
    
    # Concatenate all embeddings
    all_image_embeddings = torch.cat(all_image_embeddings)
    all_text_embeddings = torch.cat(all_text_embeddings)
    
    # Compute similarity matrix
    similarity = all_image_embeddings @ all_text_embeddings.T
    
    # Calculate retrieval metrics
    # For each query, get the rank of the correct match
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
    
    # Calculate recall@k metrics
    recalls = {}
    for k in [1, 5, 10]:
        i2t_recall = 100 * sum(r <= k for r in image_to_text_ranks) / len(image_to_text_ranks)
        t2i_recall = 100 * sum(r <= k for r in text_to_image_ranks) / len(text_to_image_ranks)
        recalls[f"R@{k}"] = {
            "image_to_text": i2t_recall,
            "text_to_image": t2i_recall,
            "average": (i2t_recall + t2i_recall) / 2
        }
    
    return recalls

def main():
    parser = argparse.ArgumentParser(description="Train CLIP model on video caption pairs")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="output",
                        help="Directory containing *_clip_data.npz files")
    parser.add_argument("--output_dir", type=str, default="clip_model",
                        help="Directory to save model checkpoints")
    
    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=768,
                        help="Dimension of the SWIN image embeddings")
    parser.add_argument("--projection_dim", type=int, default=256,
                        help="Dimension of the joint projection space")
    parser.add_argument("--text_model", type=str, default="distilbert-base-uncased",
                        help="Hugging Face transformer model for text")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Validation split ratio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else 
                                                       "mps" if torch.backends.mps.is_available() else "cpu",
                        help="Device for training (cuda, mps, or cpu)")
    
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
    
    # Split into train and validation
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    model = CLIPModel(
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        text_model_name=args.text_model
    )
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, "best_model.pt"))
            
            logger.info(f"Saved new best model with val_loss: {val_loss:.4f}")
        
        # Calculate retrieval metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            recalls = calculate_retrieval_metrics(model, val_loader, device)
            
            for k, values in recalls.items():
                logger.info(f"{k} - I2T: {values['image_to_text']:.2f}%, " +
                           f"T2I: {values['text_to_image']:.2f}%, " +
                           f"Avg: {values['average']:.2f}%")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, os.path.join(args.output_dir, "final_model.pt"))
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Calculate final metrics
    logger.info("Calculating final metrics...")
    recalls = calculate_retrieval_metrics(model, val_loader, device)
    
    # Save metrics to JSON
    with open(os.path.join(args.output_dir, "final_metrics.json"), 'w') as f:
        json.dump(recalls, f, indent=2)
    
    for k, values in recalls.items():
        logger.info(f"{k} - I2T: {values['image_to_text']:.2f}%, " +
                   f"T2I: {values['text_to_image']:.2f}%, " +
                   f"Avg: {values['average']:.2f}%")

if __name__ == "__main__":
    main()