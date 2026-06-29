"""
Deepfake Video Detection - Model Training
Trains a CNN (MobileNetV2) + LSTM model for deepfake detection
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

print("=" * 70)
print("🎬 DEEPFAKE DETECTION - MODEL TRAINING")
print("=" * 70)

# ===== LOAD PREPROCESSED DATA =====
print("\n📊 Loading preprocessed data...")
try:
    X = np.load('processed_data/X.npy')
    y = np.load('processed_data/y.npy')
    print(f"✓ Data loaded successfully")
except FileNotFoundError:
    print("❌ Error: Preprocessed data not found!")
    print("   Please run preprocess.py first")
    exit(1)

print(f"\n📈 Dataset Information:")
print(f"  - Total samples: {X.shape[0]}")
print(f"  - Real videos: {np.sum(y == 0)}")
print(f"  - Fake videos: {np.sum(y == 1)}")
print(f"  - Feature shape: {X.shape}")

# ===== TRAIN/TEST SPLIT =====
print("\n🔀 Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✓ Train samples: {X_train.shape[0]} ({X_train.shape[1]} frames, {X_train.shape[2]} features)")
print(f"✓ Test samples: {X_test.shape[0]}")
print(f"  - Train Real/Fake: {np.sum(y_train == 0)}/{np.sum(y_train == 1)}")
print(f"  - Test Real/Fake: {np.sum(y_test == 0)}/{np.sum(y_test == 1)}")

# ===== BUILD MODEL =====
print("\n🏗️  Building CNN+LSTM Model...")
print("   Architecture:")
print(f"   - LSTM layers: 1 (128 units)")
print(f"   - Dropout: 0.5")
print(f"   - Dense layers: 2")
print(f"   - Activation functions: relu, sigmoid")
print(f"   - Loss: binary_crossentropy")
print(f"   - Optimizer: adam")

model = Sequential([
    # LSTM layer for temporal modeling
    LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    BatchNormalization(),
    Dropout(0.5),
    
    # Dense layers for classification
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

print("✓ Model built successfully\n")
model.summary()

# ===== TRAINING =====
print("\n🚀 Training Model...")
print("   Hyperparameters:")
print(f"   - Epochs: 30 (with early stopping)")
print(f"   - Batch size: 16")
print(f"   - Validation split: test set")

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=0
)

# Train
epochs = 30
batch_size = 16

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# ===== EVALUATION =====
print("\n" + "=" * 70)
print("📊 MODEL EVALUATION")
print("=" * 70)

# Predictions
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Real', 'Fake'],
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🔲 Confusion Matrix:")
print(cm)

# ===== VISUALIZATIONS =====
print("\n🎨 Generating visualizations...")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'], 
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Deepfake Detection', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")
plt.close()

# 2. Training Curves
figure, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='#00f2fe')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#ff4b4b')
axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#00f2fe')
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#ff4b4b')
axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_curves.png")
plt.close()

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE")
print("=" * 70)
print(f"\n📁 Model saved as: model.h5")
print(f"📊 Performance: {accuracy*100:.2f}% accuracy")
print(f"📈 Training curves saved: training_curves.png")
print(f"🔲 Confusion matrix saved: confusion_matrix.png")
print(f"\n✓ Ready to run: python app.py")
print("=" * 70)

