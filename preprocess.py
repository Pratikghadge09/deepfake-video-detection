"""
Deepfake Video Detection - Video Preprocessing
Extracts frames from videos and generates featured tensors using MobileNetV2
"""

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tqdm import tqdm

# ===== CONFIGURATION =====
DATASET_PATH = 'dataset'
REAL_PATH = os.path.join(DATASET_PATH, 'real')
FAKE_PATH = os.path.join(DATASET_PATH, 'fake')
IMG_SIZE = 224                           # Image resize dimension
SEQUENCE_LENGTH = 15                     # Number of frames to extract per video
OUTPUT_DIR = 'processed_data'

# ===== LOAD PRETRAINED MODEL =====
print("🔧 Loading MobileNetV2 feature extractor...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)
print("✓ MobileNetV2 loaded successfully")

def extract_frames(video_path, sequence_length=SEQUENCE_LENGTH):
    """
    Extract uniformly sampled frames from a video.
    
    Args:
        video_path: Path to video file
        sequence_length: Number of frames to extract
        
    Returns:
        numpy array of shape (sequence_length, 224, 224, 3) or None if invalid
    """
    try:
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Skip videos with too few frames
        if total_frames < sequence_length:
            cap.release()
            return None
        
        # Calculate uniform indices to sample
        indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
        indices_set = set(indices)
        
        sampled_frames = []
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at selected indices
            if frame_idx in indices_set:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to 224x224
                frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
                sampled_frames.append(frame_resized)
                
                if len(sampled_frames) == sequence_length:
                    break
        
        cap.release()
        
        # Verify we got all frames
        if len(sampled_frames) != sequence_length:
            return None
        
        return np.array(sampled_frames, dtype=np.uint8)
    
    except Exception as e:
        print(f"⚠ Error processing {video_path}: {e}")
        return None

def preprocess_videos(video_dir, label, label_name):
    """
    Process all videos in a directory and extract features.
    
    Args:
        video_dir: Directory containing videos
        label: Integer label (0 for real, 1 for fake)
        label_name: String name for display
        
    Returns:
        tuple(features_array, labels_array)
    """
    features = []
    labels = []
    
    # Get list of video files
    video_files = [
        f for f in os.listdir(video_dir) 
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    ]
    
    print(f"\n📹 Processing {len(video_files)} {label_name} videos...")
    print(f"📂 Source: {video_dir}")
    
    for video_file in tqdm(video_files, desc=f"Processing {label_name}"):
        video_path = os.path.join(video_dir, video_file)
        
        # Extract frames
        frames = extract_frames(video_path)
        if frames is None:
            continue
        
        # Normalize frames for MobileNetV2
        frames = preprocess_input(frames.astype(np.float32))
        
        # Extract features using MobileNetV2
        video_features = base_model.predict(frames, verbose=0)
        
        # Store
        features.append(video_features)
        labels.append(label)
    
    print(f"✓ Successfully processed {len(features)} {label_name} videos")
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    print("=" * 60)
    print("🎬 DEEPFAKE DETECTION - VIDEO PREPROCESSING")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process real videos (label: 0 = REAL)
    real_features, real_labels = preprocess_videos(REAL_PATH, 0, "REAL")
    
    # Process fake videos (label: 1 = FAKE)
    fake_features, fake_labels = preprocess_videos(FAKE_PATH, 1, "FAKE")
    
    # Combine datasets
    print("\n" + "=" * 60)
    print("🔀 COMBINING DATASETS")
    print("=" * 60)
    
    X = np.concatenate([real_features, fake_features], axis=0)
    y = np.concatenate([real_labels, fake_labels], axis=0)
    
    print(f"Total samples: {len(X)}")
    print(f"  - Real videos: {len(real_features)}")
    print(f"  - Fake videos: {len(fake_features)}")
    print(f"Feature shape: {X.shape}")
    print(f"  - Videos: {X.shape[0]}")
    print(f"  - Frames per video: {X.shape[1]}")
    print(f"  - Features per frame: {X.shape[2]}")
    
    # Save processed data
    print("\n💾 Saving processed data...")
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
    
    print(f"✓ Features saved to {OUTPUT_DIR}/X.npy")
    print(f"✓ Labels saved to {OUTPUT_DIR}/y.npy")
    print("\n✅ Preprocessing complete! Ready for training.")
    print("=" * 60)

