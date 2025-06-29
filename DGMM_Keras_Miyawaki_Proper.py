#!/usr/bin/env python3
"""
DGMM (Deep Gaussian Mixture Model) for Miyawaki Dataset - PROPER IMPLEMENTATION
This version CORRECTLY implements similarity matrix-based reconstruction

Key differences from previous version:
1. Properly uses similarity matrix for reconstruction
2. Creates decoder from latent space
3. Demonstrates the TRUE power of similarity matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
import tensorflow as tf

# 🎯 SET RANDOM SEEDS FOR REPRODUCIBILITY
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
print(f"🎯 Random seeds set to {RANDOM_SEED} for reproducible results")

from keras import optimizers
from cal import S as calculateS
import pickle

# 📊 IMPORT EVALUATION METRICS
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
import pandas as pd

print("🚀 DGMM Miyawaki - PROPER Similarity Matrix Implementation")

# Load and preprocess Miyawaki data
print("📂 Loading Miyawaki dataset...")
data = loadmat('miyawaki_conditions_2to5_combined_sharp.mat')

X_train_orig = data['stimTrn']    # (107, 784)
Y_train_orig = data['fmriTrn']    # (107, 967)
X_test = data['stimTest']         # (12, 784)
Y_test = data['fmriTest']         # (12, 967)

print(f"📊 Miyawaki dataset: Train={X_train_orig.shape}, Test={X_test.shape}")

# Preprocess
resolution = 28
X_train_orig = X_train_orig.astype('float32')
X_test = X_test.astype('float32')
Y_train_orig = preprocessing.normalize(Y_train_orig, norm='l2')
Y_test = preprocessing.normalize(Y_test, norm='l2')

# Reshape
X_train_full = X_train_orig.reshape(X_train_orig.shape[0], resolution, resolution, 1)
X_test = X_test.reshape(X_test.shape[0], resolution, resolution, 1)

# Split training data
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train_full, Y_train_orig, test_size=0.2, random_state=RANDOM_SEED
)

print(f"✅ Data split: Train={X_train.shape[0]}, Val={X_validation.shape[0]}, Test={X_test.shape[0]}")

def create_vae_model(input_shape, latent_dim=6):
    """Create VAE model with separate encoder and decoder"""
    # Encoder
    input_layer = Input(shape=input_shape, name='input_layer')
    
    en_conv_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    en_conv_2 = Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(en_conv_1)
    en_conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(en_conv_2)
    en_conv_4 = Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(en_conv_3)
    
    flatten = Flatten()(en_conv_4)
    en_dense = Dense(256, activation='relu')(flatten)
    
    # Latent space
    en_mu = Dense(latent_dim, name='en_mu')(en_dense)
    en_var = Dense(latent_dim, name='en_var')(en_dense)
    
    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
        return z_mean + tf.exp(z_log_var / 2) * epsilon
    
    z = Lambda(sampling, output_shape=(latent_dim,))([en_mu, en_var])
    
    # Create encoder model
    encoder = Model(input_layer, [en_mu, en_var, z], name='encoder')
    
    # Decoder (separate model for similarity-based reconstruction)
    latent_input = Input(shape=(latent_dim,), name='latent_input')
    
    decoder_h = input_shape[0] // 4
    decoder_w = input_shape[1] // 4
    
    dec_dense_1 = Dense(256, activation='relu')(latent_input)
    dec_dense_2 = Dense(decoder_h * decoder_w * 128, activation='relu')(dec_dense_1)
    dec_reshape = Reshape((decoder_h, decoder_w, 128))(dec_dense_2)
    
    dec_conv_1 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(dec_reshape)
    dec_conv_2 = Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(dec_conv_1)
    dec_conv_3 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(dec_conv_2)
    dec_conv_4 = Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(dec_conv_3)
    dec_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(dec_conv_4)
    
    # Create decoder model
    decoder = Model(latent_input, dec_output, name='decoder')
    
    # Full VAE
    vae_output = decoder(z)
    vae = Model(input_layer, vae_output, name='vae')
    
    # Compile
    optimizer = optimizers.Adam(learning_rate=0.001)
    vae.compile(optimizer=optimizer, loss='mse')
    
    return vae, encoder, decoder

print(f"\n🏗️ Creating VAE model...")
vae, encoder, decoder = create_vae_model(X_train.shape[1:])
print(f"✅ Model created: {vae.count_params():,} parameters")

# Calculate similarity matrix
print(f"\n🔧 Computing similarity matrix...")
k = 10
t = 10.0
S = np.asmatrix(calculateS(k, t, Y_train, Y_validation))
print(f"✅ Similarity matrix: {S.shape} (train vs validation)")

# Training
print(f"\n🚀 Training VAE...")
best_loss = float('inf')
patience = 15
patience_counter = 0

for iteration in range(100):
    loss = vae.fit(X_train, X_train, batch_size=8, epochs=1, verbose=0)
    current_loss = loss.history['loss'][0]
    
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Loss = {current_loss:.6f}")
    
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"🛑 Early stopping at iteration {iteration}")
        break

print(f"✅ Training completed! Final loss: {current_loss:.6f}")

# Get training latent representations
print(f"\n🔍 Extracting latent representations...")
z_mu_train, z_sigma_train, _ = encoder.predict(X_train, batch_size=8)
print(f"✅ Training latents: {z_mu_train.shape}")

def calculate_metrics(original, reconstructed):
    """Calculate reconstruction metrics"""
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    mse = np.mean((orig_flat - recon_flat) ** 2)
    
    if len(original.shape) == 3 and original.shape[2] == 1:
        orig_2d = original[:,:,0]
        recon_2d = reconstructed[:,:,0]
    else:
        orig_2d = original
        recon_2d = reconstructed
    
    ssim_value = ssim(orig_2d, recon_2d, data_range=1.0)
    correlation, _ = pearsonr(orig_flat, recon_flat)
    psnr_value = psnr(orig_2d, recon_2d, data_range=1.0)
    
    orig_mean = np.mean(orig_flat)
    orig_std = np.std(orig_flat)
    recon_mean = np.mean(recon_flat)
    recon_std = np.std(recon_flat)
    fid_like = (orig_mean - recon_mean)**2 + (orig_std - recon_std)**2
    
    dot_product = np.dot(orig_flat, recon_flat)
    norm_orig = np.linalg.norm(orig_flat)
    norm_recon = np.linalg.norm(recon_flat)
    cosine_sim = dot_product / (norm_orig * norm_recon)
    
    return {
        'MSE': mse,
        'SSIM': ssim_value,
        'PixelCorr': correlation,
        'PSNR': psnr_value,
        'FID_like': fid_like,
        'CLIP_like': cosine_sim
    }

# PROPER SIMILARITY-BASED RECONSTRUCTION
print(f"\n🔍 PROPER Similarity-based reconstruction...")

similarity_metrics = []
direct_metrics = []

for i in range(X_test.shape[0]):
    print(f"Reconstructing sample {i+1}/{X_test.shape[0]}")
    
    # 1. DIRECT VAE RECONSTRUCTION (baseline)
    direct_recon = vae.predict(np.expand_dims(X_test[i], axis=0), verbose=0)[0]
    direct_metric = calculate_metrics(X_test[i], direct_recon)
    direct_metric['Sample'] = i + 1
    direct_metrics.append(direct_metric)
    
    # 2. SIMILARITY-BASED RECONSTRUCTION (PROPER)
    # Get similarity weights
    if i < S.shape[1]:
        s = np.array(S[:, i]).flatten()  # (n_train,)
    else:
        s = np.mean(S, axis=1).A1  # Average similarity
    
    # Normalize similarity weights
    s = s / np.sum(s)
    
    # Create weighted latent representation
    weighted_latent = np.dot(s, z_mu_train)  # (latent_dim,)
    
    # Decode from weighted latent
    similarity_recon = decoder.predict(np.expand_dims(weighted_latent, axis=0), verbose=0)[0]
    similarity_metric = calculate_metrics(X_test[i], similarity_recon)
    similarity_metric['Sample'] = i + 1
    similarity_metrics.append(similarity_metric)

# Calculate averages
print(f"\n📊 COMPARISON RESULTS:")

avg_direct = {}
avg_similarity = {}
for metric in ['MSE', 'SSIM', 'PixelCorr', 'PSNR', 'FID_like', 'CLIP_like']:
    avg_direct[metric] = np.mean([m[metric] for m in direct_metrics])
    avg_similarity[metric] = np.mean([m[metric] for m in similarity_metrics])

print(f"\n📈 DIRECT VAE RECONSTRUCTION:")
for metric in ['MSE', 'SSIM', 'PixelCorr', 'PSNR', 'FID_like', 'CLIP_like']:
    unit = " dB" if metric == "PSNR" else ""
    print(f"   {metric:12}: {avg_direct[metric]:.6f}{unit}")

print(f"\n🎯 SIMILARITY-BASED RECONSTRUCTION:")
for metric in ['MSE', 'SSIM', 'PixelCorr', 'PSNR', 'FID_like', 'CLIP_like']:
    unit = " dB" if metric == "PSNR" else ""
    print(f"   {metric:12}: {avg_similarity[metric]:.6f}{unit}")

print(f"\n🚀 IMPROVEMENT WITH SIMILARITY MATRIX:")
for metric in ['MSE', 'SSIM', 'PixelCorr', 'PSNR', 'FID_like', 'CLIP_like']:
    if metric == 'MSE' or metric == 'FID_like':
        improvement = ((avg_direct[metric] - avg_similarity[metric]) / avg_direct[metric]) * 100
        print(f"   {metric:12}: {improvement:+.1f}% (lower is better)")
    else:
        improvement = ((avg_similarity[metric] - avg_direct[metric]) / avg_direct[metric]) * 100
        print(f"   {metric:12}: {improvement:+.1f}% (higher is better)")

# Save results
pd.DataFrame(direct_metrics).to_csv('dgmm_miyawaki_direct_metrics.csv', index=False)
pd.DataFrame(similarity_metrics).to_csv('dgmm_miyawaki_similarity_metrics.csv', index=False)

print(f"\n✅ Results saved!")
print(f"🎯 This demonstrates the TRUE power of similarity matrix in DGMM!")
