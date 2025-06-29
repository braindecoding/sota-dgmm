#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DGMM Keras with 5-Fold Cross Validation
Created based on DGMM_Keras.py with CV enhancement

@author: duchangde (modified for CV)
"""

# GPU configuration for TensorFlow
import tensorflow as tf
# Enable GPU memory growth to avoid allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {len(gpus)} device(s)")
        print(f"GPU: {tf.config.experimental.get_device_details(gpus[0])['device_name']}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from sklearn import preprocessing
from sklearn.model_selection import KFold
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model

# 🎯 SET RANDOM SEEDS FOR REPRODUCIBILITY
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
print(f"🎯 Random seeds set to {RANDOM_SEED} for reproducible results")

from keras import backend
from keras import optimizers
from cal import S as calculateS
import pickle
import os
import pandas as pd

# 📊 IMPORT EVALUATION METRICS
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr

# Configuration
N_FOLDS = 5
MODEL_SAVE_PATH = 'dgmm_cv_model_checkpoint.pkl'
KERAS_MODEL_PATH = 'dgmm_cv_keras_model.keras'

# Load dataset
print("📂 Loading dataset...")
handwriten_69 = loadmat('digit69_28x28.mat')

# Original data
Y_full_train = handwriten_69['fmriTrn']  # 90 samples
Y_test = handwriten_69['fmriTest']       # 10 samples
X_full_train = handwriten_69['stimTrn']  # 90 samples
X_test = handwriten_69['stimTest']       # 10 samples

print(f"📊 Dataset loaded:")
print(f"   Training samples: {X_full_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")

# Prepare test data (same for all folds)
resolution = 28
X_test = X_test.astype('float32') / 255.
X_test = X_test.reshape([X_test.shape[0], resolution, resolution, 1])

# Normalize test data
min_max_scaler_test = preprocessing.MinMaxScaler(feature_range=(0, 1))   
Y_test = min_max_scaler_test.fit_transform(Y_test)

print(f"Test data prepared: X_test={X_test.shape}, Y_test={Y_test.shape}")

# Initialize 5-fold cross validation
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# Store results for all folds
cv_results = []
fold_metrics = []

# Model parameters (same for all folds)
maxiter = 50  # Reduced for faster CV
nb_epoch = 1
batch_size = 10
K = 6
C = 5
intermediate_dim = 128

# Hyper-parameters
tau_alpha = 1
tau_beta = 1
eta_alpha = 1
eta_beta = 1
gamma_alpha = 1
gamma_beta = 1
Beta = 1
rho = 0.1
k = 10
t = 10.0
L = 50  # Reduced for faster CV

# Input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
filters = 64
num_conv = 3

if backend.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

# Register custom function for Keras 3.x compatibility
@tf.keras.utils.register_keras_serializable()
def sampling(args):
    Z_mu, Z_lsgms = args
    import tensorflow as tf
    epsilon = tf.random.normal(shape=(tf.shape(Z_mu)[0], K), mean=0., stddev=1.0)
    return Z_mu + tf.exp(Z_lsgms) * epsilon

# Register custom loss function for Keras 3.x compatibility
@tf.keras.utils.register_keras_serializable()
def obj(y_true, y_pred):
    import tensorflow as tf
    from tensorflow.keras import backend, metrics
    
    # Flatten inputs for proper computation
    X = backend.flatten(y_true)
    X_mu = backend.flatten(y_pred)
    
    # Use binary crossentropy like in original implementation
    Lx = -metrics.binary_crossentropy(X, X_mu)
    
    # Return negative log likelihood (cost function)
    cost = -backend.mean(Lx)
    return cost

def build_model():
    """Build DGMM model architecture"""
    # Building the architecture
    X = Input(shape=original_img_size)
    
    conv_1 = Conv2D(img_chns, kernel_size=(2, 2), padding='same', activation='relu', name='en_conv_1')(X)
    conv_2 = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2), name='en_conv_2')(conv_1)
    conv_3 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1, name='en_conv_3')(conv_2)
    conv_4 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1, name='en_conv_4')(conv_3)
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu', name='en_dense_5')(flat)
    
    Z_mu = Dense(K, name='en_mu')(hidden)
    Z_lsgms = Dense(K, name='en_var')(hidden)
    
    Z = Lambda(sampling, output_shape=(K,))([Z_mu, Z_lsgms])
    
    # Decoder layers
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(filters * 14 * 14, activation='relu')
    
    if backend.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, 14, 14)
    else:
        output_shape = (batch_size, 14, 14, filters)
    
    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
    decoder_deconv_2 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
    decoder_deconv_3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
    decoder_mean_squash_mu = Conv2D(img_chns, kernel_size=2, padding='valid', activation='sigmoid')
    decoder_mean_squash_lsgms = Conv2D(img_chns, kernel_size=2, padding='valid', activation='tanh')
    
    hid_decoded = decoder_hid(Z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    X_mu = decoder_mean_squash_mu(x_decoded_relu)
    X_lsgms = decoder_mean_squash_lsgms(x_decoded_relu)
    
    # Create models
    DGMM = Model(inputs=X, outputs=X_mu)
    encoder = Model(inputs=X, outputs=[Z_mu, Z_lsgms])
    
    # Build reconstruction model
    Z_predict = Input(shape=(K,))
    _hid_decoded = decoder_hid(Z_predict)
    _up_decoded = decoder_upsample(_hid_decoded)
    _reshape_decoded = decoder_reshape(_up_decoded)
    _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
    _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
    _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
    X_mu_predict = decoder_mean_squash_mu(_x_decoded_relu)
    imagereconstruct = Model(inputs=Z_predict, outputs=X_mu_predict)
    
    return DGMM, encoder, imagereconstruct

def calculate_metrics(original, reconstructed):
    """Calculate comprehensive reconstruction metrics"""
    metrics_results = {}
    
    # Ensure both images are in the same format (0-1 range)
    if original.max() > 1.0:
        original = original.astype(np.float32) / 255.0
    if reconstructed.max() > 1.0:
        reconstructed = reconstructed.astype(np.float32) / 255.0
    
    # Flatten for some metrics
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # 1. MSE (Mean Squared Error)
    mse = np.mean((orig_flat - recon_flat) ** 2)
    metrics_results['MSE'] = mse
    
    # 2. SSIM (Structural Similarity Index)
    if len(original.shape) == 3 and original.shape[2] == 1:
        orig_2d = original[:,:,0]
        recon_2d = reconstructed[:,:,0]
    else:
        orig_2d = original
        recon_2d = reconstructed
    
    ssim_value = ssim(orig_2d, recon_2d, data_range=1.0)
    metrics_results['SSIM'] = ssim_value
    
    # 3. Pixel Correlation (Pearson correlation)
    correlation, _ = pearsonr(orig_flat, recon_flat)
    metrics_results['PixelCorr'] = correlation
    
    # 4. PSNR (Peak Signal-to-Noise Ratio)
    psnr_value = psnr(orig_2d, recon_2d, data_range=1.0)
    metrics_results['PSNR'] = psnr_value
    
    return metrics_results

print(f"\n🔄 Starting {N_FOLDS}-Fold Cross Validation...")
print("="*60)

# Start Cross Validation
for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(X_full_train)):
    print(f"\n📁 FOLD {fold_idx + 1}/{N_FOLDS}")
    print("-" * 40)

    # Split data for this fold
    X_train = X_full_train[train_indices]
    X_validation = X_full_train[val_indices]
    Y_train = Y_full_train[train_indices]
    Y_validation = Y_full_train[val_indices]

    # Preprocess training data for this fold
    X_train = X_train.astype('float32') / 255.
    X_validation = X_validation.astype('float32') / 255.

    # Reshape to channels_last format
    X_train = X_train.reshape([X_train.shape[0], resolution, resolution, 1])
    X_validation = X_validation.reshape([X_validation.shape[0], resolution, resolution, 1])

    # Normalize Y data for this fold
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    Y_train = min_max_scaler.fit_transform(Y_train)
    Y_validation = min_max_scaler.transform(Y_validation)

    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_validation.shape[0]} samples")

    numTrn = X_train.shape[0]
    numVal = X_validation.shape[0]
    D1 = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
    D2 = Y_train.shape[1]

    # Build model for this fold
    print("🏗️ Building model...")
    DGMM, encoder, imagereconstruct = build_model()

    # Compile model
    try:
        opt_method = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    except:
        opt_method = optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    DGMM.compile(optimizer=opt_method, loss=obj)
    print(f"✅ Model compiled for fold {fold_idx + 1}")

    # Initialize parameters for this fold
    Z_mu = np.asmatrix(np.random.random(size=(numTrn, K)))
    B_mu = np.asmatrix(np.random.random(size=(K, D2)))
    R_mu = np.asmatrix(np.random.random(size=(numTrn, C)))
    sigma_r = np.asmatrix(np.eye((C)))
    H_mu = np.asmatrix(np.random.random(size=(C, D2)))
    sigma_h = np.asmatrix(np.eye((C)))

    tau_mu = tau_alpha / tau_beta
    eta_mu = eta_alpha / eta_beta
    gamma_mu = gamma_alpha / gamma_beta

    Y_mu_param = np.array(Z_mu * B_mu + R_mu * H_mu)
    Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2)))

    # Calculate similarity matrix for this fold (NO DATA LEAKAGE)
    print("🔧 Computing similarity matrix (train-validation split)...")
    S = np.asmatrix(calculateS(k, t, Y_train, Y_validation))
    print(f"✅ Similarity matrix computed: {S.shape}")

    # Training loop for this fold
    print(f"🚀 Starting training for fold {fold_idx + 1}...")
    fold_losses = []

    for iteration in range(maxiter):
        if iteration % 10 == 0:
            print(f"   Iteration {iteration}/{maxiter}")

        # Update Z
        history = DGMM.fit(X_train, X_train, shuffle=True, verbose=0, epochs=nb_epoch, batch_size=batch_size)
        fold_losses.append(history.history['loss'][0])

        [Z_mu, Z_lsgms] = encoder.predict(X_train, verbose=0)
        Z_mu = np.asmatrix(Z_mu)

        # Update B
        temp1 = np.exp(Z_lsgms)
        temp2 = Z_mu.T * Z_mu + np.asmatrix(np.diag(temp1.sum(axis=0)))
        temp3 = tau_mu * np.asmatrix(np.eye(K))
        sigma_b = (gamma_mu * temp2 + temp3).I
        B_mu = sigma_b * gamma_mu * Z_mu.T * (np.asmatrix(Y_train) - R_mu * H_mu)

        # Update H
        RTR_mu = R_mu.T * R_mu + numTrn * sigma_r
        sigma_h = (eta_mu * np.asmatrix(np.eye(C)) + gamma_mu * RTR_mu).I
        H_mu = sigma_h * gamma_mu * R_mu.T * (np.asmatrix(Y_train) - Z_mu * B_mu)

        # Update R
        HHT_mu = H_mu * H_mu.T + D2 * sigma_h
        sigma_r = (np.asmatrix(np.eye(C)) + gamma_mu * HHT_mu).I
        R_mu = (sigma_r * gamma_mu * H_mu * (np.asmatrix(Y_train) - Z_mu * B_mu).T).T

        # Update tau
        tau_alpha_new = tau_alpha + 0.5 * K * D2
        tau_beta_new = tau_beta + 0.5 * ((np.diag(B_mu.T * B_mu)).sum() + D2 * sigma_b.trace())
        tau_mu = tau_alpha_new / tau_beta_new
        tau_mu = tau_mu[0,0]

        # Update eta
        eta_alpha_new = eta_alpha + 0.5 * C * D2
        eta_beta_new = eta_beta + 0.5 * ((np.diag(H_mu.T * H_mu)).sum() + D2 * sigma_h.trace())
        eta_mu = eta_alpha_new / eta_beta_new
        eta_mu = eta_mu[0,0]

        # Update gamma
        gamma_alpha_new = gamma_alpha + 0.5 * numTrn * D2
        gamma_temp = np.asmatrix(Y_train) - Z_mu * B_mu - R_mu * H_mu
        gamma_temp = np.multiply(gamma_temp, gamma_temp)
        gamma_temp = gamma_temp.sum(axis=0)
        gamma_temp = gamma_temp.sum(axis=1)
        gamma_beta_new = gamma_beta + 0.5 * gamma_temp
        gamma_mu = gamma_alpha_new / gamma_beta_new
        gamma_mu = gamma_mu[0,0]

        # Calculate Y_mu
        Y_mu_param = np.array(Z_mu * B_mu + R_mu * H_mu)
        Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2)))

    print(f"✅ Training completed for fold {fold_idx + 1}")
    print(f"   Final loss: {fold_losses[-1]:.6f}")

    # Reconstruction and evaluation for this fold
    print(f"🔍 Performing reconstruction for fold {fold_idx + 1}...")

    # Reconstruct test images
    numTest = X_test.shape[0]
    X_reconstructed_mu = np.zeros((numTest, img_chns, img_rows, img_cols))
    HHT = H_mu * H_mu.T + D2 * sigma_h
    Temp = gamma_mu * np.asmatrix(np.eye(D2)) - (gamma_mu**2) * (H_mu.T * (np.asmatrix(np.eye(C)) + gamma_mu * HHT).I * H_mu)

    for i in range(numTest):
        # Use similarity from validation set (CORRECT approach)
        s = S[:, i % S.shape[1]]  # Handle index mismatch

        # Compute latent variables
        z_sigma_test = (B_mu * Temp * B_mu.T + (1 + rho * s.sum(axis=0)[0,0]) * np.asmatrix(np.eye(K))).I
        z_mu_test = (z_sigma_test * (B_mu * Temp * (np.asmatrix(Y_test)[i,:]).T + rho * np.asmatrix(Z_mu).T * s)).T

        temp_mu = np.zeros((1, img_chns, img_rows, img_cols))
        epsilon_std = 1
        for l in range(L):
            epsilon = np.random.normal(0, epsilon_std, 1)
            z_test = z_mu_test + np.sqrt(np.diag(z_sigma_test)) * epsilon
            x_reconstructed_mu = imagereconstruct.predict(z_test, batch_size=1, verbose=0)
            temp_mu = temp_mu + x_reconstructed_mu
        x_reconstructed_mu = temp_mu / L

        # Handle shape mismatch
        x_reconstructed_squeezed = np.squeeze(x_reconstructed_mu)
        if x_reconstructed_squeezed.shape == (28, 28, 28):
            x_reconstructed_reshaped = x_reconstructed_squeezed[:, :, 0:1]
        elif len(x_reconstructed_squeezed.shape) == 1:
            expected_size = img_rows * img_cols * img_chns
            if x_reconstructed_squeezed.shape[0] != expected_size:
                if x_reconstructed_squeezed.shape[0] > expected_size:
                    x_reconstructed_squeezed = x_reconstructed_squeezed[:expected_size]
                else:
                    padding = expected_size - x_reconstructed_squeezed.shape[0]
                    x_reconstructed_squeezed = np.pad(x_reconstructed_squeezed, (0, padding), 'constant')
            x_reconstructed_reshaped = x_reconstructed_squeezed.reshape(img_rows, img_cols, img_chns)
        else:
            x_reconstructed_reshaped = x_reconstructed_squeezed

        x_reconstructed_transposed = x_reconstructed_reshaped.transpose(2, 0, 1)
        X_reconstructed_mu[i,:,:,:] = x_reconstructed_transposed

    # Calculate metrics for this fold
    fold_sample_metrics = []
    for i in range(numTest):
        original = X_test[i]
        reconstructed = X_reconstructed_mu[i].transpose(1, 2, 0)
        sample_metrics = calculate_metrics(original, reconstructed)
        sample_metrics['fold'] = fold_idx + 1
        sample_metrics['sample'] = i + 1
        fold_sample_metrics.append(sample_metrics)

    # Calculate average metrics for this fold
    fold_avg_metrics = {}
    for metric in ['MSE', 'SSIM', 'PixelCorr', 'PSNR']:
        fold_avg_metrics[metric] = np.mean([m[metric] for m in fold_sample_metrics])

    fold_avg_metrics['fold'] = fold_idx + 1
    fold_metrics.append(fold_avg_metrics)

    print(f"✅ Fold {fold_idx + 1} Results:")
    print(f"   MSE: {fold_avg_metrics['MSE']:.6f}")
    print(f"   SSIM: {fold_avg_metrics['SSIM']:.4f}")
    print(f"   Pixel Correlation: {fold_avg_metrics['PixelCorr']:.4f}")
    print(f"   PSNR: {fold_avg_metrics['PSNR']:.2f} dB")

    # Store fold results
    fold_result = {
        'fold': fold_idx + 1,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'final_loss': fold_losses[-1],
        'losses': fold_losses,
        'num_train': numTrn,
        'num_val': numVal,
        'metrics': fold_avg_metrics,
        'sample_metrics': fold_sample_metrics
    }
    cv_results.append(fold_result)

    # Save reconstruction visualization for this fold
    plt.figure(figsize=(12, 2))
    n = min(10, numTest)
    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(np.rot90(np.fliplr(X_test[i].reshape(resolution, resolution))), cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_ylabel('Original', rotation=90, size='large')

        # Display reconstructed images
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i].reshape(resolution, resolution))), cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_ylabel('Reconstructed', rotation=90, size='large')

    plt.suptitle(f'DGMM CV Fold {fold_idx + 1} - Reconstruction Results')
    plt.savefig(f'dgmm_cv_fold_{fold_idx + 1}_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Fold {fold_idx + 1} visualization saved")

print("\n🎉 Cross Validation Completed!")
print("="*60)

# Calculate overall CV statistics
print("\n📊 CROSS VALIDATION RESULTS SUMMARY:")
print("-" * 50)

# Calculate mean and std for each metric across folds
cv_summary = {}
for metric in ['MSE', 'SSIM', 'PixelCorr', 'PSNR']:
    values = [fold['metrics'][metric] for fold in cv_results]
    cv_summary[f'{metric}_mean'] = np.mean(values)
    cv_summary[f'{metric}_std'] = np.std(values)

    print(f"{metric}:")
    print(f"  Mean: {cv_summary[f'{metric}_mean']:.6f}")
    print(f"  Std:  {cv_summary[f'{metric}_std']:.6f}")
    print(f"  Values: {[f'{v:.4f}' for v in values]}")
    print()

# Calculate training loss statistics
final_losses = [fold['final_loss'] for fold in cv_results]
cv_summary['loss_mean'] = np.mean(final_losses)
cv_summary['loss_std'] = np.std(final_losses)

print(f"Training Loss:")
print(f"  Mean: {cv_summary['loss_mean']:.6f}")
print(f"  Std:  {cv_summary['loss_std']:.6f}")
print(f"  Values: {[f'{v:.4f}' for v in final_losses]}")

# Save detailed results
print(f"\n💾 Saving results...")

# Save fold metrics
fold_metrics_df = pd.DataFrame([fold['metrics'] for fold in cv_results])
fold_metrics_df.to_csv('dgmm_cv_fold_metrics.csv', index=False)
print(f"✅ Fold metrics saved: dgmm_cv_fold_metrics.csv")

# Save CV summary
cv_summary_df = pd.DataFrame([cv_summary])
cv_summary_df.to_csv('dgmm_cv_summary.csv', index=False)
print(f"✅ CV summary saved: dgmm_cv_summary.csv")

# Save all sample metrics
all_sample_metrics = []
for fold in cv_results:
    all_sample_metrics.extend(fold['sample_metrics'])

sample_metrics_df = pd.DataFrame(all_sample_metrics)
sample_metrics_df.to_csv('dgmm_cv_all_sample_metrics.csv', index=False)
print(f"✅ All sample metrics saved: dgmm_cv_all_sample_metrics.csv")

# Save training losses for each fold
losses_data = []
for fold in cv_results:
    for i, loss in enumerate(fold['losses']):
        losses_data.append({
            'fold': fold['fold'],
            'iteration': i,
            'loss': loss
        })

losses_df = pd.DataFrame(losses_data)
losses_df.to_csv('dgmm_cv_training_losses.csv', index=False)
print(f"✅ Training losses saved: dgmm_cv_training_losses.csv")

# Create summary plot
plt.figure(figsize=(15, 10))

# Plot 1: Training losses for all folds
plt.subplot(2, 3, 1)
for fold in cv_results:
    plt.plot(fold['losses'], label=f"Fold {fold['fold']}")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss per Fold')
plt.legend()
plt.grid(True)

# Plot 2: MSE across folds
plt.subplot(2, 3, 2)
mse_values = [fold['metrics']['MSE'] for fold in cv_results]
plt.bar(range(1, N_FOLDS+1), mse_values)
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.title('MSE per Fold')
plt.grid(True)

# Plot 3: SSIM across folds
plt.subplot(2, 3, 3)
ssim_values = [fold['metrics']['SSIM'] for fold in cv_results]
plt.bar(range(1, N_FOLDS+1), ssim_values)
plt.xlabel('Fold')
plt.ylabel('SSIM')
plt.title('SSIM per Fold')
plt.grid(True)

# Plot 4: Pixel Correlation across folds
plt.subplot(2, 3, 4)
corr_values = [fold['metrics']['PixelCorr'] for fold in cv_results]
plt.bar(range(1, N_FOLDS+1), corr_values)
plt.xlabel('Fold')
plt.ylabel('Pixel Correlation')
plt.title('Pixel Correlation per Fold')
plt.grid(True)

# Plot 5: PSNR across folds
plt.subplot(2, 3, 5)
psnr_values = [fold['metrics']['PSNR'] for fold in cv_results]
plt.bar(range(1, N_FOLDS+1), psnr_values)
plt.xlabel('Fold')
plt.ylabel('PSNR (dB)')
plt.title('PSNR per Fold')
plt.grid(True)

# Plot 6: Box plot of all metrics
plt.subplot(2, 3, 6)
metrics_data = [mse_values, ssim_values, corr_values, [p/10 for p in psnr_values]]  # Scale PSNR for visibility
plt.boxplot(metrics_data, labels=['MSE', 'SSIM', 'PixelCorr', 'PSNR/10'])
plt.ylabel('Metric Value')
plt.title('Metrics Distribution')
plt.grid(True)

plt.tight_layout()
plt.savefig('dgmm_cv_summary_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Summary plots saved: dgmm_cv_summary_plots.png")

print(f"\n🎯 FINAL CV RESULTS:")
print(f"   MSE: {cv_summary['MSE_mean']:.6f} ± {cv_summary['MSE_std']:.6f}")
print(f"   SSIM: {cv_summary['SSIM_mean']:.4f} ± {cv_summary['SSIM_std']:.4f}")
print(f"   Pixel Correlation: {cv_summary['PixelCorr_mean']:.4f} ± {cv_summary['PixelCorr_std']:.4f}")
print(f"   PSNR: {cv_summary['PSNR_mean']:.2f} ± {cv_summary['PSNR_std']:.2f} dB")
print(f"   Training Loss: {cv_summary['loss_mean']:.6f} ± {cv_summary['loss_std']:.6f}")

print(f"\n✅ Cross Validation Analysis Complete!")
print(f"📁 Generated files:")
print(f"   - dgmm_cv_fold_metrics.csv")
print(f"   - dgmm_cv_summary.csv")
print(f"   - dgmm_cv_all_sample_metrics.csv")
print(f"   - dgmm_cv_training_losses.csv")
print(f"   - dgmm_cv_summary_plots.png")
print(f"   - dgmm_cv_fold_X_reconstruction.png (for each fold)")
print(f"\n🎉 Done!")
