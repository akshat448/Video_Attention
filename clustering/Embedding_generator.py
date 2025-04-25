import argparse
import os
import logging
import torch
import timm
import numpy as np
import h5py
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
print("hello?")
def load_frames(video_frames_path):
    """Loads frames from either a .npy or .h5 file."""
    npy_path = video_frames_path.with_suffix('.npy')
    h5_path = video_frames_path.with_suffix('.h5')

    if npy_path.exists():
        try:
            logger.debug(f"Loading frames from NPY: {npy_path}")
            frames = np.load(npy_path)
            # NPY stores frames usually as (num_frames, H, W, C)
            return frames
        except Exception as e:
            logger.error(f"Error loading NPY file {npy_path}: {e}")
            return None
    elif h5_path.exists():
        try:
            logger.debug(f"Loading frames from HDF5: {h5_path}")
            with h5py.File(h5_path, 'r') as f:
                # HDF5 dataset might be named 'frames' as per the creation script
                if 'frames' in f:
                    frames = f['frames'][()]
                    # HDF5 might store as (num_frames, H, W, C)
                    return frames
                else:
                    logger.error(f"Dataset 'frames' not found in HDF5 file: {h5_path}")
                    return None
        except Exception as e:
            logger.error(f"Error loading HDF5 file {h5_path}: {e}")
            return None
    else:
        logger.warning(f"No .npy or .h5 file found for base path: {video_frames_path}")
        return None

def get_preprocessor(model):
    """Gets the appropriate torchvision transforms for the given timm model."""
    config = model.default_cfg
    img_size = config['input_size'][1:] # Get (height, width)
    mean = config['mean']
    std = config['std']

    # Note: Input frames from the database are likely RGB (H, W, C) NumPy arrays.
    # We need to convert them to PIL Images or Tensors (C, H, W) and then normalize.
    preprocess = transforms.Compose([
        transforms.ToPILImage(), # Convert NumPy HWC to PIL Image
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), # Convert PIL Image to Tensor CHW and scales to [0, 1]
        transforms.Normalize(mean=mean, std=std),
    ])
    return preprocess

def generate_embeddings(frames_array, model, preprocess_fn, batch_size, device):
    """Generates embeddings for a list/array of frames using the model."""
    model.eval() # Set model to evaluation mode
    embeddings = []
    num_frames = frames_array.shape[0]

    with torch.no_grad(): # Disable gradient calculations
        for i in range(0, num_frames, batch_size):
            batch_frames_np = frames_array[i:i+batch_size]
            
            # Preprocess each frame in the batch
            # Assumes frames_array is (N, H, W, C) NumPy array
            batch_tensors = torch.stack([preprocess_fn(frame) for frame in batch_frames_np])
            
            batch_tensors = batch_tensors.to(device)

            # Get features (embeddings) from the model
            # Using forward_features or num_classes=0 ensures we get embeddings
            batch_embeddings = model(batch_tensors)

            # Move embeddings to CPU and convert to NumPy
            embeddings.append(batch_embeddings.cpu().numpy())
            logger.debug(f"Processed batch {i//batch_size + 1}/{(num_frames + batch_size - 1)//batch_size}")

    if not embeddings:
        return np.array([]) # Return empty array if no frames

    # Concatenate embeddings from all batches
    full_embeddings = np.concatenate(embeddings, axis=0)
    logger.info(f"Generated embeddings of shape: {full_embeddings.shape}")
    return full_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate SWIN Transformer embeddings for processed video frames.")
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Directory containing processed frame folders (e.g., 'processed_frames')")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save the generated embeddings")
    parser.add_argument("--model_name", type=str, default="swin_tiny_patch4_window7_224",
                        help="Name of the pre-trained SWIN model from timm (e.g., swin_tiny_patch4_window7_224)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing frames")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps",
                        help="Device to use ('cuda' or 'cpu')")

    args = parser.parse_args()

    # --- Setup ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # --- Load Model ---
    try:
        logger.info(f"Loading pre-trained model: {args.model_name}")
        # Load the model without the final classification head to get features
        model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model '{args.model_name}': {e}")
        return

    # --- Get Preprocessor ---
    preprocess_fn = get_preprocessor(model)
    logger.info(f"Using model default preprocessing: Input size {model.default_cfg['input_size']}, Mean {model.default_cfg['mean']}, Std {model.default_cfg['std']}")


    # --- Process Videos ---
    logger.info(f"Starting embedding generation from: {args.input_dir}")
    video_ids = [d.name for d in args.input_dir.iterdir() if d.is_dir()]
    
    if not video_ids:
        logger.warning(f"No video subdirectories found in {args.input_dir}")
        return

    for video_id in tqdm(video_ids, desc="Generating Embeddings"):
        logger.info(f"Processing video: {video_id}")
        video_frame_dir = args.input_dir / video_id
        
        # Determine the base path for frame files (without extension)
        # Assumes files are named like '<video_id>_frames.npy' or '.h5'
        frame_file_base = video_frame_dir / f"{video_id}_frames"

        # Define output path for embeddings
        output_embedding_path = args.output_dir / f"{video_id}_embeddings.npy"

        if output_embedding_path.exists():
            logger.info(f"Embeddings already exist for {video_id}, skipping.")
            continue

        # Load frames
        frames = load_frames(frame_file_base)

        if frames is None or frames.shape[0] == 0:
            logger.warning(f"No frames loaded or empty frames for {video_id}, skipping.")
            continue
        
        logger.info(f"Loaded {frames.shape[0]} frames for {video_id} with shape {frames.shape[1:]}")

        # Generate embeddings
        try:
            embeddings_np = generate_embeddings(frames, model, preprocess_fn, args.batch_size, device)

            # Save embeddings
            if embeddings_np.size > 0:
                np.save(output_embedding_path, embeddings_np)
                logger.info(f"Saved embeddings for {video_id} to {output_embedding_path}")
            else:
                 logger.warning(f"No embeddings generated for {video_id}.")

        except Exception as e:
            logger.error(f"Error generating embeddings for {video_id}: {e}")
            # Optionally, continue to the next video or stop execution
            # continue

    logger.info("=== Embedding Generation Complete ===")

if __name__ == "__main__":
    main()