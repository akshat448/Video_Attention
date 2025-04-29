import argparse
import os
import logging
import subprocess
import multiprocessing as mp
import time
import shutil
import h5py
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import video processing function
from video_processing import extract_frames_cv2

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed parameters
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
WORKERS = min(8, mp.cpu_count())  # Use 8 workers or max available
VIDEO_QUALITY = "medium"  # Fixed medium quality

def download_one_video(url, video_dir, video_id, failed_folder):
    """Download a video using yt-dlp with optimized parameters."""
    output_path = os.path.join(video_dir, f"video-{video_id}")
    output_path_with_ext = output_path + ".mp4"
    
    try:
        logger.info(f"Downloading video {video_id}")
        cmd = [
            "yt-dlp", 
            url,
            "-o", f"{output_path}.%(ext)s",
            "--format", "mp4[height<=720]",  # Medium quality
            "--no-playlist",
            "--retries", "3",
            "--fragment-retries", "10",
            "--concurrent-fragments", "5",
            "--no-overwrites"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully downloaded video {video_id}")
            # Find the actual downloaded file (may have different extension)
            for ext in ['mp4', 'mkv', 'webm']:
                potential_path = f"{output_path}.{ext}"
                if os.path.exists(potential_path):
                    # Rename to standard format if not already mp4
                    if ext != 'mp4':
                        os.rename(potential_path, output_path_with_ext)
                    else:
                        output_path_with_ext = potential_path
                    break
            return True, output_path_with_ext
        else:
            logger.error(f"Error downloading {video_id}: {result.stderr}")
            with open(os.path.join(failed_folder, f'{video_id}.txt'), 'w') as f:
                f.write(f"Error downloading {video_id}: {result.stderr}")
            return False, None
            
    except Exception as e:
        logger.exception(f"Exception while downloading {video_id}: {e}")
        with open(os.path.join(failed_folder, f'{video_id}.txt'), 'w') as f:
            f.write(f"Exception: {e}")
        return False, None

def download_worker(args):
    """Helper function for parallel downloads"""
    url, video_dir, video_id, failed_folder = args
    return download_one_video(url, video_dir, video_id, failed_folder)

def download_videos_parallel(video_dir, txt_file, refresh):
    """Process videos from text file and download the first 20 in parallel."""
    os.makedirs(video_dir, exist_ok=True)
    failed_folder = 'failed_folder'
    os.makedirs(failed_folder, exist_ok=True)
    
    with open(txt_file, 'r') as f:
        video_links = f.read().splitlines()
    
    # Limit to the first 20 videos
    video_links = video_links[:50]
    
    existing_videos = os.listdir(video_dir) if os.path.exists(video_dir) else []
    existing_ids = [vid.split('video-')[1].split('.')[0] for vid in existing_videos if vid.startswith('video-')]
    
    logger.info(f"Found {len(video_links)} videos to process")
    
    # Prepare download tasks
    download_tasks = []
    for url in video_links:
        try:
            video_id = url.split('https://www.youtube.com/watch?v=')[1]
            
            if video_id in existing_ids and not refresh:
                logger.info(f"Skipping {video_id} - already downloaded")
                continue
                
            download_tasks.append((url, video_dir, video_id, failed_folder))
        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
    
    # Set up progress bar
    pbar = tqdm(total=len(download_tasks), desc="Downloading videos")
    results = {}
    failed_ids = set()
    
    # Execute tasks with multiprocessing
    with mp.Pool(processes=WORKERS) as pool:
        for i, (success, video_path) in enumerate(pool.imap_unordered(download_worker, download_tasks)):
            if success and video_path:
                video_id = os.path.basename(video_path).split('video-')[1].split('.')[0]
                results[video_id] = video_path
            else:
                failed_ids.add(download_tasks[i][2])  # Add failed video_id to the set
            pbar.update(1)
    
    pbar.close()
    logger.info(f"Downloaded {len(results)} videos successfully")
    
    # Remove failed URLs from the text file
    if failed_ids:
        with open(txt_file, 'w') as f:
            for url in video_links:
                video_id = url.split('https://www.youtube.com/watch?v=')[1]
                if video_id not in failed_ids:
                    f.write(url + '\n')
        logger.info(f"Removed {len(failed_ids)} failed video URLs from {txt_file}")
    
    return results

def save_frames_compressed(video_id, frames, timestamps, output_dir, compression='npy', compression_level=4):
    """Save extracted frames and timestamps with different compression formats"""
    result = {"video_id": video_id}
    frames_array = np.array(frames)
    timestamps_array = np.array(timestamps)
    
    # Create output directory
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Save as NPY
    if compression in ['npy', 'both']:
        npy_path = os.path.join(video_output_dir, f"{video_id}_frames.npy")
        np.save(npy_path, frames_array)
        result["npy_path"] = npy_path
        logger.info(f"Saved {len(frames)} frames as NPY: {npy_path}")
        
        # Save timestamps as NPY
        timestamps_path = os.path.join(video_output_dir, f"{video_id}_timestamps.npy")
        np.save(timestamps_path, timestamps_array)
        result["timestamps_path"] = timestamps_path
        logger.info(f"Saved {len(timestamps)} timestamps as NPY: {timestamps_path}")
    
    # Save as HDF5 with compression
    if compression in ['hdf5', 'both']:
        h5_path = os.path.join(video_output_dir, f"{video_id}_frames.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset(
                'frames',
                data=frames_array,
                compression='gzip',
                compression_opts=compression_level,
                chunks=True,
            )
            # Also save timestamps in the same H5 file
            f.create_dataset(
                'timestamps',
                data=timestamps_array,
                compression='gzip',
                compression_opts=compression_level,
                chunks=True,
            )
        result["h5_path"] = h5_path
        logger.info(f"Saved {len(frames)} frames and {len(timestamps)} timestamps as HDF5 with compression: {h5_path}")
    
    return result


def process_video(video_path, output_dir, target_fps=1, save_jpg=True, 
                  compression='npy', compression_level=4):
    """Process a video: extract frames and save them with timestamps."""
    try:
        # Extract frames and timestamps
        success, video_id, frame_count = extract_frames_cv2(
            video_path=video_path,
            output_dir=output_dir,
            target_fps=target_fps,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            save_jpg=True,  # Always save JPGs
            save_npy=False,
            save_timestamps=True  # Enable timestamp saving
        )
        
        if not success or frame_count == 0:
            logger.error(f"Frame extraction failed for {video_path}")
            return {"success": False, "video_id": video_id}
        
        # Get the frames and timestamps that were extracted
        video_output_dir = os.path.join(output_dir, video_id)
        jpg_dir = os.path.join(video_output_dir, "jpg")
        
        # Load frames from JPGs
        jpg_files = sorted([os.path.join(jpg_dir, f) for f in os.listdir(jpg_dir) if f.endswith('.jpg')])
        
        frames = []
        for jpg_file in jpg_files:
            frame = cv2.imread(jpg_file)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        logger.info(f"Loaded {len(frames)} frames from JPGs for {video_id}")
        
        # Generate timestamps if they weren't saved earlier
        timestamps = []
        timestamp_path = os.path.join(video_output_dir, f"{video_id}_timestamps.npy")
        
        if os.path.exists(timestamp_path):
            # Load saved timestamps
            timestamps = np.load(timestamp_path).tolist()
            logger.info(f"Loaded {len(timestamps)} timestamps from {timestamp_path}")
        else:
            # Generate timestamps based on target_fps
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if original_fps > 0:
                frame_interval = max(1, round(original_fps / target_fps))
                timestamps = [(i * frame_interval) / original_fps for i in range(len(frames))]
                logger.info(f"Generated {len(timestamps)} timestamps for {video_id}")
            else:
                # Fallback: just use frame index / target_fps
                timestamps = [i / target_fps for i in range(len(frames))]
                logger.warning(f"Using fallback timestamp generation for {video_id}")
        
        # Save with compression, now including timestamps
        result = save_frames_compressed(
            video_id=video_id,
            frames=frames,
            timestamps=timestamps,
            output_dir=output_dir,
            compression=compression,
            compression_level=compression_level
        )
        
        # Delete JPGs if requested
        if not save_jpg:
            shutil.rmtree(jpg_dir)
            logger.info(f"Removed original JPGs for {video_id}")
            
        result["success"] = True
        return result
            
    except Exception as e:
        logger.exception(f"Error processing video {video_path}: {e}")
        return {"success": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Download and process videos from QuerYD dataset")
    
    # General options
    parser.add_argument("--video_dir", type=Path, default="videos",
                        help="Directory to save downloaded videos")
    parser.add_argument("--output_dir", type=Path, default="processed_frames",
                        help="Directory to save processed frames")
    parser.add_argument("--txt_file", type=Path, default="filtered_links.txt",
                        help="File containing video URLs")
    
    # Processing options
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading, process existing videos only")
    parser.add_argument("--skip_processing", action="store_true", 
                        help="Skip processing, download videos only")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download existing videos")
    parser.add_argument("--fps", type=int, default=1,
                        help="Target frames per second to extract")
    
    # Storage options
    parser.add_argument("--keep_jpgs", action="store_true",
                        help="Keep individual JPG frames")
    parser.add_argument("--delete_videos", action="store_true",
                        help="Delete videos after processing")
    parser.add_argument("--compression", choices=["npy", "hdf5", "both"], default="hdf5",
                        help="Compression format for frames")
    parser.add_argument("--compression_level", type=int, default=4, choices=range(1, 10),
                        help="Compression level (1-9, higher=more compression)")
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/{datetime.now().strftime(r'%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Create directories
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=== QuerYD Dataset Processing ===")
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using {WORKERS} workers for parallel processing")
    logger.info(f"Frame resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    
    video_paths = {}
    
    # STEP 1: Download videos
    if not args.skip_download:
        logger.info("=== Downloading videos ===")
        
        # Check for yt-dlp
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        except FileNotFoundError:
            logger.error("yt-dlp is not installed. Please install it with: pip install yt-dlp")
            return
        
        # Download videos
        video_paths = download_videos_parallel(
            video_dir=args.video_dir,
            txt_file=args.txt_file,
            refresh=args.refresh
        )
    else:
        logger.info("Skipping download phase, using existing videos")
        # Find existing videos
        for video_file in os.listdir(args.video_dir):
            if video_file.startswith("video-") and os.path.isfile(os.path.join(args.video_dir, video_file)):
                video_id = video_file.split("video-")[1].split(".")[0]
                video_paths[video_id] = os.path.join(args.video_dir, video_file)
    
    # STEP 2: Process videos
    if not args.skip_processing:
        logger.info("=== Processing videos ===")
        
        processing_results = []
        for video_id, video_path in tqdm(video_paths.items(), desc="Processing videos"):
            result = process_video(
                video_path=video_path,
                output_dir=args.output_dir,
                target_fps=args.fps,
                save_jpg=args.keep_jpgs,
                compression=args.compression,
                compression_level=args.compression_level
            )
            processing_results.append(result)
            
            # Delete video if requested
            if args.delete_videos and result.get("success", False):
                try:
                    os.remove(video_path)
                    logger.info(f"Deleted video: {video_path}")
                except Exception as e:
                    logger.error(f"Failed to delete video {video_path}: {e}")
        
        # Save processing results
        results_file = os.path.join(args.output_dir, "processing_results.json")
        with open(results_file, 'w') as f:
            json.dump(processing_results, f, indent=2)
        logger.info(f"Saved processing results to {results_file}")
    
    logger.info("=== Processing complete ===")

if __name__ == "__main__":
    main()