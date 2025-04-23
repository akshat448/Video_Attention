import argparse
import os
import logging
import subprocess
from pathlib import Path
from datetime import datetime

def download_one_video(url, video_dir, video_id, failed_folder):
    """Download a video using yt-dlp instead of pytube"""
    output_path = os.path.join(video_dir, f"video-{video_id}")
    
    try:
        print(f"Downloading video {video_id}")
        cmd = [
            "yt-dlp", 
            url,
            "-o", f"{output_path}",
            "--format", "mp4",  # Use mp4 format
            "--no-playlist"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully downloaded video {video_id}")
            return True
        else:
            print(f"Error downloading {video_id}: {result.stderr}")
            with open(os.path.join(failed_folder, f'{video_id}.txt'), 'w') as f:
                f.write(f"Error downloading {video_id}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Exception while downloading {video_id}: {e}")
        with open(os.path.join(failed_folder, f'{video_id}.txt'), 'w') as f:
            f.write(f"Exception: {e}")
        return False

def download_videos(video_dir, txt_file, refresh, logging):
    """Process videos from text file and download them"""
    os.makedirs(video_dir, exist_ok=True)
    failed_folder = 'failed_folder'
    os.makedirs(failed_folder, exist_ok=True)
    
    with open(txt_file, 'r') as f:
        video_links = f.read().splitlines()
    
    existing_videos = os.listdir(video_dir) if os.path.exists(video_dir) else []
    existing_ids = [vid.split('video-')[1].split('.')[0] for vid in existing_videos if vid.startswith('video-')]
    
    print(f"Found {len(video_links)} videos to process")
    
    for i, url in enumerate(video_links):
        video_id = url.split('https://www.youtube.com/watch?v=')[1]
        
        if video_id in existing_ids and not refresh:
            print(f"Skipping {video_id} - already downloaded")
            continue
            
        print(f"Processing {i+1}/{len(video_links)}: {video_id}")
        download_one_video(url, video_dir, video_id, failed_folder)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=Path, default="videos")
    parser.add_argument("--txt_file", type=Path, default="relevant-video-links.txt")
    parser.add_argument("--refresh", action="store_true")
    
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=f"logs/{datetime.now().strftime(r'%m%d_%H%M%S')}.log",
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    
    # Check if yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        print("yt-dlp is not installed. Please install it with: pip install yt-dlp")
        return
        
    download_videos(args.video_dir, args.txt_file, args.refresh, logging)

if __name__ == "__main__":
    main()