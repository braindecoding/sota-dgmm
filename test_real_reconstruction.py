#!/usr/bin/env python3
"""
Real reconstruction test using trained Keras model
This script properly loads and uses the trained autoencoder for reconstruction
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from sklearn import preprocessing
import tensorflow as tf

# GPU configuration for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPU available: {len(gpus)} device(s)')
    except RuntimeError as e:
        print(e)
else:
    print('No GPU found, using CPU')

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
import tensorflow as tf

print("🧪 DGMM Real Reconstruction Test")
print("=" * 50)

# Model parameters (must match training)
K = 6  # latent dimensions
C = 5  # components
D2 = 3092  # fMRI dimensions

# Define sampling function (must match training exactly)
@tf.keras.utils.register_keras_serializable()
def sampling(args):
    Z_mu, Z_lsgms = args
    epsilon = tf.random.normal(shape=(tf.shape(Z_mu)[0], K), mean=0., stddev=1.0)
    return Z_mu + tf.exp(Z_lsgms) * epsilon

def build_autoencoder():
    """Build the same autoencoder architecture as in training"""
    img_rows, img_cols, img_chns = 28, 28, 1
    
    # Input layer
    X = Input(shape=(img_rows, img_cols, img_chns), name='input_layer')
    
    # Encoder
    en_conv_1 = Conv2D(filters=1, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu', name='en_conv_1')(X)
    en_conv_2 = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu', name='en_conv_2')(en_conv_1)
    en_conv_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='en_conv_3')(en_conv_2)
    en_conv_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='en_conv_4')(en_conv_3)
    
    flatten = Flatten()(en_conv_4)
    en_dense_5 = Dense(128, activation='relu', name='en_dense_5')(flatten)
    
    Z_mu = Dense(K, name='en_mu')(en_dense_5)
    Z_lsgms = Dense(K, name='en_var')(en_dense_5)
    
    # Sampling layer
    Z = Lambda(sampling, output_shape=(K,))([Z_mu, Z_lsgms])
    
    # Decoder
    de_dense_1 = Dense(128, activation='relu')(Z)
    de_dense_2 = Dense(14*14*64, activation='relu')(de_dense_1)
    de_reshape = Reshape((14, 14, 64))(de_dense_2)
    
    de_conv_1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(de_reshape)
    de_conv_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(de_conv_1)
    de_conv_3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(de_conv_2)
    
    # Final output layer
    X_mu = Conv2D(filters=1, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='sigmoid')(de_conv_3)
    
    # Create autoencoder model
    autoencoder = Model(inputs=X, outputs=X_mu)
    
    return autoencoder

def load_and_test():
    """Load trained model and test reconstruction"""
    
    # Check if model files exist
    checkpoint_file = 'dgmm_model_checkpoint.pkl'
    keras_model_file = 'dgmm_keras_model.keras'
    
    if not os.path.exists(checkpoint_file):
        print(f"❌ Checkpoint file not found: {checkpoint_file}")
        return False
        
    if not os.path.exists(keras_model_file):
        print(f"❌ Keras model file not found: {keras_model_file}")
        return False
    
    print(f"✅ Found checkpoint: {checkpoint_file}")
    print(f"✅ Found Keras model: {keras_model_file}")
    
    # Load dataset
    print("\n📊 Loading dataset...")
    handwriten_69 = loadmat('digit69_28x28.mat')
    Y_train = handwriten_69['fmriTrn']
    Y_test = handwriten_69['fmriTest']
    X_train = handwriten_69['stimTrn']
    X_test = handwriten_69['stimTest']
    
    print(f"Data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Normalization (same as training)
    print("\n🔧 Normalizing data...")
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))   
    Y_train = min_max_scaler.fit_transform(Y_train)     
    Y_test = min_max_scaler.transform(Y_test)
    
    # Reshape data for CNN
    img_rows, img_cols, img_chns = 28, 28, 1
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_chns)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_chns)
    
    print(f"Reshaped data:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Build and load model
    print("\n🧠 Building and loading trained model...")
    try:
        autoencoder = build_autoencoder()
        autoencoder.load_weights(keras_model_file)
        print("✅ Model loaded successfully!")
        
        # Print model summary
        print("\n📋 Model Summary:")
        autoencoder.summary()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test reconstruction
    print("\n🔍 Testing reconstruction...")
    try:
        # Test on first few samples
        n_test_samples = min(10, X_test.shape[0])
        X_test_sample = X_test[:n_test_samples]
        
        print(f"Testing on {n_test_samples} samples...")
        X_reconstructed = autoencoder.predict(X_test_sample, verbose=1)
        
        print(f"✅ Reconstruction successful!")
        print(f"Original shape: {X_test_sample.shape}")
        print(f"Reconstructed shape: {X_reconstructed.shape}")
        
        # Calculate reconstruction error
        mse = np.mean((X_test_sample - X_reconstructed) ** 2)
        print(f"📊 Mean Squared Error: {mse:.6f}")
        
        # Create visualization
        print("\n🎨 Creating visualization...")
        fig, axes = plt.subplots(2, n_test_samples, figsize=(20, 6))
        fig.suptitle('DGMM Real Reconstruction Results', fontsize=16)
        
        for i in range(n_test_samples):
            # Original images
            axes[0, i].imshow(X_test_sample[i].reshape(28, 28), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructed images
            axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('real_reconstruction_results.png', dpi=150, bbox_inches='tight')
        print("✅ Saved visualization: real_reconstruction_results.png")
        
        # Save numerical results
        results = {
            'X_test_sample': X_test_sample,
            'X_reconstructed': X_reconstructed,
            'mse': mse,
            'n_samples': n_test_samples
        }
        
        with open('real_reconstruction_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("✅ Saved results: real_reconstruction_results.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = load_and_test()
    
    if success:
        print("\n🎉 Real reconstruction test completed successfully!")
        print("\n💡 Files generated:")
        print("   real_reconstruction_results.png  # Visual comparison with real model")
        print("   real_reconstruction_results.pkl  # Numerical results")
    else:
        print("\n❌ Real reconstruction test failed!")
        print("💡 Please check that the model was trained successfully.")
