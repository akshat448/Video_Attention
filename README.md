# Video Attention Project - QuerYD Dataset Processing

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/Vid_Attention.git
    cd Vid_Attention
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Steps

The `data_setup.py` script handles both downloading videos and processing them into frames.

### Basic Usage Command

```bash
python data_setup.py --txt_file relevant-video-links.txt
```

### Command Line Options

#### General Options:
- `--video_dir`: Directory to save downloaded videos (default: `videos`).
- `--output_dir`: Directory to save processed frames (default: `processed_frames`).
- `--txt_file`: File containing video URLs (default: `relevant-video-links.txt`).

#### Processing Options:
- `--skip_download`: Skip downloading, process existing videos only.
- `--skip_processing`: Skip processing, download videos only.
- `--refresh`: Re-download existing videos.
- `--fps`: Target frames per second to extract (default: `1`).

#### Storage Options:
- `--keep_jpgs`: Keep individual JPG frames.
- `--delete_videos`: Delete videos after processing.
- `--compression`: Compression format for frames (`npy`, `hdf5`, or `both`).
- `--compression_level`: Compression level (1-9, higher = more compression).

### Example Commands

- **Download and process videos**:
    ```bash
    python data_setup.py --txt_file relevant-video-links.txt
    ```

- **Process existing videos only**:
    ```bash
    python data_setup.py --skip_download
    ```

- **Download videos only**:
    ```bash
    python data_setup.py --skip_processing
    ```

- **Save space by deleting videos after processing**:
    ```bash
    python data_setup.py --delete_videos
    ```

## Processing Pipeline

1. **Video Downloading**:
    - Videos are downloaded using `yt-dlp` in parallel.

2. **Frame Extraction**:
    - Frames are extracted at `224x224` resolution at the specified FPS (default: `1fps`).

3. **Compression**:
    - Frames are stored in HDF5 format by default, with optional JPG retention.

## Integration with Metadata

The QuerYD dataset includes rich metadata in both JSON and pickle formats:

- **JSON Metadata**:
    - Located in `/data/JSON Metadata v2/`, containing detailed transcript and timing information.

- **Pickle Files**:
    - Contains processed captions, timestamps, and confidence scores.

These metadata files can be aligned with extracted frames using their timestamps for building multimodal models.