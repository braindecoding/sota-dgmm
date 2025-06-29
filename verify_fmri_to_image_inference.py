#!/usr/bin/env python3
"""
VERIFICATION: Memastikan bahwa inference menggunakan fMRI sebagai input 
dan menghasilkan gambar rekonstruksi sebagai output
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import load_model
import tensorflow as tf

print("🔍 VERIFIKASI: fMRI → Gambar Rekonstruksi")
print("="*60)

# Load dataset
print("\n1️⃣ LOADING MIYAWAKI DATASET:")
data = loadmat('miyawaki_conditions_2to5_combined_sharp.mat')

X_train_orig = data['stimTrn']    # (107, 784) - Visual stimuli
Y_train_orig = data['fmriTrn']    # (107, 967) - fMRI responses
X_test = data['stimTest']         # (12, 784) - Test visual stimuli
Y_test = data['fmriTest']         # (12, 967) - Test fMRI responses

print(f"📊 Data Structure:")
print(f"   X_train (visual stimuli): {X_train_orig.shape}")
print(f"   Y_train (fMRI responses): {Y_train_orig.shape}")
print(f"   X_test (visual stimuli): {X_test.shape}")
print(f"   Y_test (fMRI responses): {Y_test.shape}")

# Preprocess
resolution = 28
X_train_orig = X_train_orig.astype('float32')
X_test = X_test.astype('float32')
Y_train_orig = preprocessing.normalize(Y_train_orig, norm='l2')
Y_test = preprocessing.normalize(Y_test, norm='l2')

# Reshape visual data
X_train_full = X_train_orig.reshape(X_train_orig.shape[0], resolution, resolution, 1)
X_test = X_test.reshape(X_test.shape[0], resolution, resolution, 1)

print(f"\n📊 After preprocessing:")
print(f"   X_train (visual): {X_train_full.shape} - range [{X_train_full.min():.3f}, {X_train_full.max():.3f}]")
print(f"   Y_train (fMRI): {Y_train_orig.shape} - range [{Y_train_orig.min():.3f}, {Y_train_orig.max():.3f}]")
print(f"   X_test (visual): {X_test.shape} - range [{X_test.min():.3f}, {X_test.max():.3f}]")
print(f"   Y_test (fMRI): {Y_test.shape} - range [{Y_test.min():.3f}, {Y_test.max():.3f}]")

print(f"\n2️⃣ ANALISIS TRAINING PROCESS:")
print("-" * 40)

# Split training data
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train_full, Y_train_orig, test_size=0.2, random_state=42
)

print(f"✅ Training Process Analysis:")
print(f"   Model Input during training: X_train (visual images) → {X_train.shape}")
print(f"   Model Output during training: X_train (same visual images) → {X_train.shape}")
print(f"   fMRI data Y_train: {Y_train.shape} - USED FOR SIMILARITY MATRIX ONLY")
print(f"   fMRI data Y_validation: {Y_validation.shape} - USED FOR SIMILARITY MATRIX ONLY")

print(f"\n🔍 IMPORTANT: VAE Training Process")
print(f"   - VAE learns: Visual Image → Latent → Visual Image")
print(f"   - fMRI data is NOT used in VAE training")
print(f"   - fMRI data is ONLY used for similarity calculation")

print(f"\n3️⃣ ANALISIS INFERENCE PROCESS:")
print("-" * 40)

try:
    # Load trained model
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], 6), mean=0., stddev=1.0)
        return z_mean + tf.exp(z_log_var / 2) * epsilon
    
    vae = load_model('dgmm_miyawaki_model.keras', custom_objects={'sampling': sampling})
    encoder = tf.keras.Model(vae.input, vae.layers[-3].output)  # Get encoder part
    
    print(f"✅ Model loaded successfully")
    print(f"   VAE input shape: {vae.input.shape}")
    print(f"   VAE output shape: {vae.output.shape}")
    
    # Load similarity matrix calculation
    from cal import S as calculateS
    k = 10
    t = 10.0
    S = np.asmatrix(calculateS(k, t, Y_train, Y_validation))
    print(f"   Similarity matrix S: {S.shape} (train fMRI vs validation fMRI)")
    
    print(f"\n🔍 STEP-BY-STEP INFERENCE PROCESS:")
    print(f"   Step 1: Input = Y_test[i] (fMRI data) → shape {Y_test[0].shape}")
    print(f"   Step 2: Calculate similarity with training fMRI")
    print(f"   Step 3: Get weighted latent representation")
    print(f"   Step 4: Decode to visual image")
    print(f"   Step 5: Output = Reconstructed image → shape {X_test[0].shape}")
    
    # Demonstrate actual inference
    print(f"\n4️⃣ DEMONSTRASI INFERENCE AKTUAL:")
    print("-" * 40)
    
    test_sample_idx = 0
    print(f"\n🧪 Testing sample {test_sample_idx + 1}:")
    
    # INPUT: fMRI data
    fmri_input = Y_test[test_sample_idx]  # (967,) - fMRI signal
    print(f"   INPUT (fMRI): shape={fmri_input.shape}, range=[{fmri_input.min():.4f}, {fmri_input.max():.4f}]")
    
    # GROUND TRUTH: Visual image (for comparison only)
    ground_truth = X_test[test_sample_idx]  # (28, 28, 1) - actual image
    print(f"   GROUND TRUTH (image): shape={ground_truth.shape}, range=[{ground_truth.min():.4f}, {ground_truth.max():.4f}]")
    
    # INFERENCE PROCESS
    print(f"\n🔄 INFERENCE STEPS:")
    
    # Step 1: Calculate similarity with training fMRI
    if test_sample_idx < S.shape[1]:
        s = np.array(S[:, test_sample_idx]).flatten()
    else:
        s = np.mean(S, axis=1).A1
    print(f"   Step 1: Similarity weights calculated from fMRI")
    print(f"           Similarity shape: {s.shape}")
    print(f"           Similarity range: [{s.min():.4f}, {s.max():.4f}]")
    
    # Step 2: Get training latent representations
    z_mu_train, _, _ = encoder.predict(X_train, batch_size=8, verbose=0)
    print(f"   Step 2: Training latent representations")
    print(f"           Latent shape: {z_mu_train.shape}")
    
    # Step 3: Create weighted latent (THIS IS WHERE fMRI INFLUENCES RECONSTRUCTION)
    weighted_latent = np.dot(s.T, z_mu_train)
    print(f"   Step 3: Weighted latent from fMRI similarity")
    print(f"           Weighted latent shape: {weighted_latent.shape}")
    print(f"           Weighted latent range: [{weighted_latent.min():.4f}, {weighted_latent.max():.4f}]")
    
    # Step 4: ALTERNATIVE - Direct VAE reconstruction (what actually happens in current code)
    direct_reconstruction = vae.predict(np.expand_dims(ground_truth, axis=0), verbose=0)[0]
    print(f"   Step 4: Direct VAE reconstruction (current implementation)")
    print(f"           Reconstruction shape: {direct_reconstruction.shape}")
    print(f"           Reconstruction range: [{direct_reconstruction.min():.4f}, {direct_reconstruction.max():.4f}]")
    
    # OUTPUT: Reconstructed image
    print(f"\n📤 OUTPUT (Reconstructed Image):")
    print(f"   Shape: {direct_reconstruction.shape}")
    print(f"   Range: [{direct_reconstruction.min():.4f}, {direct_reconstruction.max():.4f}]")
    print(f"   Type: Visual image (28x28 pixels)")
    
    # Calculate reconstruction quality
    mse = np.mean((ground_truth.flatten() - direct_reconstruction.flatten()) ** 2)
    correlation = np.corrcoef(ground_truth.flatten(), direct_reconstruction.flatten())[0, 1]
    
    print(f"\n📊 RECONSTRUCTION QUALITY:")
    print(f"   MSE: {mse:.8f}")
    print(f"   Correlation: {correlation:.6f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # fMRI signal (as heatmap)
    axes[0].imshow(fmri_input.reshape(1, -1), cmap='viridis', aspect='auto')
    axes[0].set_title('INPUT: fMRI Signal\n(967 voxels)')
    axes[0].set_xlabel('Voxel Index')
    axes[0].set_ylabel('Signal')
    
    # Ground truth image
    axes[1].imshow(ground_truth[:,:,0], cmap='gray')
    axes[1].set_title('GROUND TRUTH\n(Original Image)')
    axes[1].axis('off')
    
    # Reconstructed image
    axes[2].imshow(direct_reconstruction[:,:,0], cmap='gray')
    axes[2].set_title('OUTPUT: Reconstructed\n(From fMRI)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('fmri_to_image_verification.png', dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: fmri_to_image_verification.png")
    
except Exception as e:
    print(f"❌ Error during inference verification: {e}")

print(f"\n5️⃣ CRITICAL ANALYSIS:")
print("="*60)

print(f"\n🚨 IMPORTANT FINDINGS:")
print(f"   ✅ INPUT: fMRI data (967 voxels)")
print(f"   ✅ OUTPUT: Visual image (28x28 pixels)")
print(f"   ⚠️  CURRENT IMPLEMENTATION ISSUE:")
print(f"       - Model uses VISUAL IMAGE as input to VAE")
print(f"       - fMRI data is used only for similarity calculation")
print(f"       - This is NOT true fMRI → Image inference!")

print(f"\n🔍 WHAT SHOULD HAPPEN FOR TRUE fMRI → IMAGE:")
print(f"   1. Input: fMRI signal only")
print(f"   2. Use similarity matrix to find similar training samples")
print(f"   3. Use weighted latent representation")
print(f"   4. Decode weighted latent to image")
print(f"   5. Output: Reconstructed image")

print(f"\n🔧 WHAT ACTUALLY HAPPENS:")
print(f"   1. Input: Visual image (ground truth)")
print(f"   2. VAE reconstructs the same image")
print(f"   3. fMRI data is ignored in reconstruction")
print(f"   4. Output: Nearly identical image")

print(f"\n🎯 CONCLUSION:")
print(f"   ❌ Current implementation is NOT true fMRI → Image inference")
print(f"   ❌ It's actually Image → Image reconstruction (VAE)")
print(f"   ❌ fMRI data is not used for actual reconstruction")
print(f"   ✅ Results are excellent because it's just VAE autoencoding")

print(f"\n💡 TO FIX FOR TRUE fMRI → IMAGE INFERENCE:")
print(f"   1. Remove visual image input during inference")
print(f"   2. Use only fMRI data as input")
print(f"   3. Implement proper similarity-based latent reconstruction")
print(f"   4. Decode from fMRI-derived latent representation")
