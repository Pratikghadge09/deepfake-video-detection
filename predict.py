"""
Deepfake Video Detection - Prediction Script
Loads trained model and predicts if a video is real or fake
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
import os
import sys

# ===== CONFIGURATION =====
IMG_SIZE = 224
SEQUENCE_LENGTH = 15
MODEL_PATH = 'model.h5'

# ===== LOAD MODELS =====
def load_models():
    """Load feature extractor and trained model"""
    print("🔧 Loading models...")
    
    # Load MobileNetV2 for feature extraction
    try:
        feature_extractor = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            pooling='avg'
        )
        print("✓ MobileNetV2 loaded")
    except Exception as e:
        print(f"❌ Error loading MobileNetV2: {e}")
        return None, None
    
    # Load trained classification model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Please run train.py first")
        return feature_extractor, None
    
    try:
        model = load_model(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return feature_extractor, None
    
    return feature_extractor, model

def extract_frames(video_path, sequence_length=SEQUENCE_LENGTH):
    """
    Extract uniformly sampled frames from video.
    
    Args:
        video_path: Path to video file
        sequence_length: Number of frames to extract
        
    Returns:
        tuple(numpy_array, error_message)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < sequence_length:
            cap.release()
            return None, f"Video has {total_frames} frames, need at least {sequence_length}"
        
        # Calculate uniform sampling indices
        indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
        indices_set = set(indices)
        
        sampled_frames = []
        frame_idx = 0
        
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in indices_set:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to 224x224
                frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
                sampled_frames.append(frame_resized)
                
                if len(sampled_frames) == sequence_length:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        if len(sampled_frames) != sequence_length:
            return None, "Could not extract required number of frames"
        
        return np.array(sampled_frames, dtype=np.uint8), None
    
    except Exception as e:
        return None, f"Error extracting frames: {str(e)}"

def predict_video(video_path, feature_extractor, model):
    """
    Predict if video is real or fake.
    
    Args:
        video_path: Path to video file
        feature_extractor: MobileNetV2 model
        model: Trained LSTM model
        
    Returns:
        tuple(label, confidence)
    """
    if model is None:
        return "ERROR", 0.0, "Model not loaded"
    
    # Extract frames
    frames, error = extract_frames(video_path)
    if error:
        return "ERROR", 0.0, error
    
    try:
        # Preprocess frames for MobileNetV2
        frames = preprocess_input(frames.astype(np.float32))
        
        # Extract features using MobileNetV2
        video_features = feature_extractor.predict(frames, verbose=0)
        
        # Add batch dimension
        video_features = np.expand_dims(video_features, axis=0)
        
        # Make prediction
        prediction = model.predict(video_features, verbose=0)[0][0]
        
        # Determine label and confidence
        if prediction > 0.5:
            label = "FAKE"
            confidence = float(prediction)
        else:
            label = "REAL"
            confidence = float(1 - prediction)
        
        return label, confidence * 100, None
    
    except Exception as e:
        return "ERROR", 0.0, f"Prediction error: {str(e)}"

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("=" * 60)
        print("🛡️  DEEPFAKE VIDEO DETECTION - PREDICTION")
        print("=" * 60)
        print("\nUsage: python predict.py <video_path>")
        print("\nSupported formats: MP4, AVI, MOV, MKV, WEBM")
        print("\nExample: python predict.py sample_video.mp4")
        print("=" * 60)
        return
    
    video_path = sys.argv[1]
    
    # Verify file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: File not found - {video_path}")
        return
    
    print("=" * 60)
    print("🛡️  DEEPFAKE VIDEO DETECTION - PREDICTION")
    print("=" * 60)
    print(f"\n📹 Video: {video_path}")
    print(f"📊 File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    # Load models
    feature_extractor, model = load_models()
    if feature_extractor is None:
        return
    
    # Predict
    print("\n⏳ Analyzing video...")
    label, confidence, error = predict_video(video_path, feature_extractor, model)
    
    if error:
        print(f"❌ {error}")
        return
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 PREDICTION RESULT")
    print("=" * 60)
    print(f"\n🎯 Classification: {label}")
    print(f"📈 Confidence: {confidence:.2f}%")
    
    if label == "REAL":
        print(f"\n✓ This video appears to be AUTHENTIC")
    elif label == "FAKE":
        print(f"\n⚠ This video shows DEEPFAKE characteristics")
    
    print("\n" + "=" * 60)
    print("Note: This analysis supports human judgment, not a definitive proof")
    print("=" * 60)

if __name__ == "__main__":
    main()

