import cv2
import os
import numpy as np
import h5py
import json
from pathlib import Path
import pickle
from tqdm import tqdm

def extract_frames_cv2(video_path, output_dir, target_fps=1, width=224, height=224, 
                      save_jpg=True, save_npy=True, save_timestamps=True):
    """
    Extracts frames from a video file using OpenCV (cv2), resizes them,
    converts them to RGB, and saves them as JPG and/or NPY files.
    Now also calculates and saves frame timestamps.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): The base directory for saving output.
        target_fps (int): Target frames per second to extract.
        width (int): Target width for extracted frames.
        height (int): Target height for extracted frames.
        save_jpg (bool): Whether to save individual frames as JPG files.
        save_npy (bool): Whether to save all frames as a single NPY file.
        save_timestamps (bool): Whether to save frame timestamps.

    Returns:
        tuple: (success (bool), video_id (str), frame_count (int))
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False, None, 0
    
    # Extract video ID from filename (assuming format video-{id})
    video_filename = os.path.basename(video_path)
    video_id = video_filename.split('.')[0]  # Remove file extension
    
    print(f"Processing video: {video_id}")
    
    # Create output directories
    video_output_dir = os.path.join(output_dir, video_id)
    jpg_output_dir = os.path.join(video_output_dir, 'jpg') if save_jpg else None
    
    if save_jpg:
        os.makedirs(jpg_output_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False, video_id, 0
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    print(f"Video: {video_id} | FPS: {original_fps:.2f} | Duration: {duration:.2f}s | Total frames: {total_frames}")
    
    if original_fps <= 0:
        print("Error: Could not determine video FPS.")
        cap.release()
        return False, video_id, 0
    
    # Calculate frame skip interval
    frame_interval = max(1, round(original_fps / target_fps))
    estimated_frames = total_frames // frame_interval
    print(f"Extracting at ~{target_fps} FPS (every {frame_interval} frames) | Estimated frames: {estimated_frames}")
    
    # Extract frames
    frame_count = 0
    saved_frames = []
    saved_timestamps = []  # New list to store timestamps
    
    with tqdm(total=estimated_frames, desc=f"Extracting frames from {video_id}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we should save this frame
            if frame_count % frame_interval == 0:
                # Calculate timestamp for this frame
                timestamp = frame_count / original_fps
                
                # Resize frame
                resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # Save as JPG
                if save_jpg:
                    jpg_filename = f"frame_{frame_count:06d}.jpg"
                    jpg_path = os.path.join(jpg_output_dir, jpg_filename)
                    cv2.imwrite(jpg_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))  # Save in BGR for OpenCV compatibility
                
                # Add to frames array (in RGB)
                saved_frames.append(rgb_frame)
                saved_timestamps.append(timestamp)  # Store the timestamp
                pbar.update(1)
            
            frame_count += 1
    
    cap.release()
    
    # Save frames and timestamps
    if saved_frames:
        os.makedirs(video_output_dir, exist_ok=True)
        
        if save_npy:
            frames_array = np.array(saved_frames)
            npy_path = os.path.join(video_output_dir, f"{video_id}_frames.npy")
            np.save(npy_path, frames_array)
            print(f"Saved {len(saved_frames)} frames as NPY: {npy_path}")
            
            # Save timestamps as NPY
            if save_timestamps:
                timestamps_array = np.array(saved_timestamps)
                timestamps_path = os.path.join(video_output_dir, f"{video_id}_timestamps.npy")
                np.save(timestamps_path, timestamps_array)
                print(f"Saved {len(saved_timestamps)} timestamps as NPY: {timestamps_path}")
    
    print(f"Frame extraction complete for {video_id}. Extracted {len(saved_frames)} frames.")
    return True, video_id, len(saved_frames)

def process_videos_in_directory(video_dir, output_dir, target_fps=1, save_jpg=True, limit=None):
    """
    Process all videos in a directory.
    
    Args:
        video_dir (str): Directory containing videos
        output_dir (str): Base directory for output
        target_fps (int): Target frames per second
        save_jpg (bool): Whether to save frames as JPG
        limit (int): Optional limit to number of videos to process
        
    Returns:
        dict: Summary of processing results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of video files
    video_paths = list(Path(video_dir).glob("video-*"))
    
    if limit:
        video_paths = video_paths[:limit]
        
    print(f"Found {len(video_paths)} videos to process" + 
          (f" (limited to {limit})" if limit else ""))
    
    # Process each video
    results = {
        "successful": 0,
        "failed": 0,
        "total_frames": 0,
        "videos": {}
    }
    
    for video_path in video_paths:
        success, video_id, frame_count = extract_frames_cv2(
            video_path=str(video_path),
            output_dir=output_dir,
            target_fps=target_fps,
            save_jpg=save_jpg
        )
        
        if success:
            results["successful"] += 1
            results["total_frames"] += frame_count
        else:
            results["failed"] += 1
            
        results["videos"][video_id] = {
            "success": success,
            "frames": frame_count
        }
    
    # Save processing summary
    with open(os.path.join(output_dir, "processing_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessing complete. {results['successful']}/{len(video_paths)} videos processed successfully.")
    print(f"Total frames extracted: {results['total_frames']}")
    
    return results

if __name__ == "__main__":
    # Configuration
    VIDEO_DIR = "/Users/akshat/Developer/Vid_Attention/data_extraction/data/QuerYD/videos"
    OUTPUT_DIR = "/Users/akshat/Developer/Vid_Attention/data_extraction/data/QuerYD/processed_frames"
    TARGET_FPS = 1
    SAVE_JPG = False  # Set to False if you don't want to save JPG files
    LIMIT = None  # Set to a number to limit processing, None for all videos
    
    results = process_videos_in_directory(
        video_dir=VIDEO_DIR,
        output_dir=OUTPUT_DIR,
        target_fps=TARGET_FPS,
        save_jpg=SAVE_JPG,
        limit=LIMIT
    )