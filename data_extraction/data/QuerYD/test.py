import os
import cv2
import subprocess
from pathlib import Path

def check_video_file(video_path):
    """Check if a video file is valid and extract basic info."""
    print(f"\n{'='*60}")
    print(f"Checking video: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # 1. Check file size
    size_bytes = os.path.getsize(video_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    
    if size_bytes < 10000:  # Less than 10KB is suspicious for a video
        print("⚠️ WARNING: File is very small for a video!")
        return False
    
    # 2. Try to get video info with ffprobe
    try:
        cmd = [
            "ffprobe", 
            "-v", "error",
            "-show_entries", "format=duration,size,bit_rate:stream=width,height,codec_name,avg_frame_rate",
            "-of", "default=noprint_wrappers=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\nVideo properties (ffprobe):")
            print(result.stdout)
            return True
        else:
            print(f"⚠️ FFprobe couldn't analyze the video: {result.stderr}")
    except Exception as e:
        print(f"Error running ffprobe: {e}")
    
    # 3. Try OpenCV as a backup
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("⚠️ ERROR: OpenCV couldn't open the video file!")
            return False
        
        # Get basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print("\nVideo properties (OpenCV):")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Frame count: {frame_count}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Try to extract first frame as a test
        ret, frame = cap.read()
        if ret:
            print("✅ Successfully read the first frame")
            
            # Save the frame as a thumbnail (optional)
            thumbnail_dir = Path("thumbnails")
            thumbnail_dir.mkdir(exist_ok=True)
            thumbnail_path = thumbnail_dir / f"{os.path.basename(video_path)}_thumbnail.jpg"
            cv2.imwrite(str(thumbnail_path), frame)
            print(f"✅ Saved thumbnail to {thumbnail_path}")
        else:
            print("⚠️ Could not read any frames!")
            
        cap.release()
        return ret  # Return True if we could read a frame
        
    except Exception as e:
        print(f"Error analyzing with OpenCV: {e}")
        return False

def main():
    video_dir = Path("/Users/akshat/Developer/Vid_Attention/data_extraction/data/QuerYD/videos")
    
    # Find all video files
    video_files = list(video_dir.glob("video-*"))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Check each video
    success_count = 0
    for video_path in video_files:
        if check_video_file(str(video_path)):
            success_count += 1
    
    print(f"\nSummary: {success_count}/{len(video_files)} videos appear to be valid")

if __name__ == "__main__":
    main()