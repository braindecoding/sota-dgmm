#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Inspired from
@author: duchangde 
and Miyawaki on Kamitami Labs
Modified and development by @awangga
"""

import os    
os.environ['THEANO_FLAGS'] = "device=gpu"  
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras import metrics

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from lib import prepro,ars,obj,init,train

# In[]: Load dataset X stimulus Y fMRI
resolution = 28
#X_train,X_test,Y_train,Y_test=prepro.getXY('digit69_28x28.mat',resolution)
X_train, X_test, X_validation, Y_train, Y_test, Y_validation=prepro.getXYVal('./data/digit69_28x28.mat',resolution)

# In[]: Set the model parameters and hyper-parameters
maxiter = 200
nb_epoch = 1
batch_size = 10
D1 = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
D2 = Y_train.shape[1]
K = 6 # dimensi latent space
C = 5
#intermediate_dim = 128 # origin
intermediate_dim = 256 # origin

import sys
 
# total arguments
n = len(sys.argv)
print("Total arguments passed:", n)
 
# Arguments passed K+intermediate_dim + K + maxiter + batch_size
print("\nName of Python script:", sys.argv[0])
 
print("\nArguments passed:", end = " ")
for i in range(1, n):
    print(sys.argv[i], end = " ")

K=int(sys.argv[1])
intermediate_dim=int(sys.argv[2])
batch_size=int(sys.argv[3])
maxiter=int(sys.argv[4])
experimentname=str(K)+"_"+str(intermediate_dim)+"_" + str(batch_size)+"_" + str(maxiter)


#hyper-parameters
tau_alpha = 1
tau_beta = 1
eta_alpha = 1
eta_beta = 1
gamma_alpha = 1
gamma_beta = 1

Beta = 1 # Beta-VAE for Learning Disentangled Representations
rho=0.1  # posterior regularization parameter


k=10     # k-nearest neighbors origin
t = 10.0 # kernel parameter in similarity measure

L = 100   # Monte-Carlo sampling

np.random.seed(1000)
numTrn=X_train.shape[0]#ada 90 data training
numTest=X_test.shape[0]#ada 10 data testing

# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1

# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

if backend.image_data_format() == 'channels_first': # atau 'channels_last'
    original_img_size = (img_chns, img_rows, img_cols)#1,28, 28
else:
    original_img_size = (img_rows, img_cols, img_chns)#28, 28, 1

# In[]: Building the architechture
#input arsitektur dimensi stimulus
X = Input(shape=original_img_size)
#input arsitektur dimensi fmri
Y = Input(shape=(D2,))
Y_mu = Input(shape=(D2,))
Y_lsgms = Input(shape=(D2,))

Z,Z_lsgms,Z_mu = ars.encoder(X, D2, img_chns, filters, num_conv, intermediate_dim, K)

# In[]: we instantiate these layers separately so as to reuse them later
decoder_hid,decoder_upsample,decoder_reshape,decoder_deconv_1,decoder_deconv_2,decoder_deconv_3_upsamp,decoder_mean_squash_mu,decoder_mean_squash_lsgms=ars.decoderars(intermediate_dim, filters, batch_size, num_conv, img_chns)
X_mu,X_lsgms=ars.decoders(Z, decoder_hid,decoder_upsample,decoder_reshape,decoder_deconv_1,decoder_deconv_2,decoder_deconv_3_upsamp,decoder_mean_squash_mu,decoder_mean_squash_lsgms)

# In[]:define custom loss objective function   
def custom_loss(X, X_mu):#stimulus asli dan hasil pembangkitan
    X = backend.flatten(X)
    X_mu = backend.flatten(X_mu) 
    Lp = 0.5 * backend.mean( 1 + Z_lsgms - backend.square(Z_mu) - backend.exp(Z_lsgms), axis=-1)     
    Lx =  - metrics.binary_crossentropy(X, X_mu) # Pixels have a Bernoulli distribution  
    Ly =  obj.Y_normal_logpdf(Y, Y_mu, Y_lsgms,backend) # Voxels have a Gaussian distribution
    lower_bound = backend.mean(Lp + 10000 * Lx + Ly)
    cost = - lower_bound
    return  cost 


#Jika Y, Y_mu, dan Y_lsgms hanya digunakan dalam perhitungan kerugian dan tidak berkontribusi langsung pada perhitungan output dari model, mereka mungkin berfungsi sebagai "label" tambahan yang membantu model belajar representasi yang lebih baik dari data dengan memberikan lebih banyak informasi tentang bagaimana kerugian harus dihitung.
DGMM = Model(inputs=[X, Y, Y_mu, Y_lsgms], outputs=X_mu)

try:
    opt_method = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
except:
    opt_method = optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    

DGMM.compile(optimizer = opt_method, loss = custom_loss)
DGMM.summary()

# build a model to project inputs on the latent space
encoder = Model(inputs=X, outputs=[Z_mu,Z_lsgms])
# build a model to project inputs on the output space
imagepredict = Model(inputs=X, outputs=[X_mu,X_lsgms])

# build a digit generator that can sample from the learned distribution
Z_predict = Input(shape=(K,))

X_mu_predict,X_lsgms_predict=ars.decoders(Z_predict, decoder_hid,decoder_upsample,decoder_reshape,decoder_deconv_1,decoder_deconv_2,decoder_deconv_3_upsamp,decoder_mean_squash_mu,decoder_mean_squash_lsgms)

imagereconstruct = Model(inputs=Z_predict, outputs=X_mu_predict)

# In[]: Initialization

Z_mu,B_mu,R_mu,H_mu=init.randombetween0and1withmatrixsize(numTrn, K, C, D2)
Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu).astype(np.float32)

sigma_r,sigma_h = init.matriksidentitasukuran(C)

tau_mu,eta_mu,gamma_mu=init.alphabagibeta(tau_alpha,tau_beta,eta_alpha,eta_beta,gamma_alpha,gamma_beta)
Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2))).astype(np.float32)

#S=np.mat(calculate.S(k, t, Y_train, Y_test))

from lib import siamese,calculate
#S=np.mat(siamese.S(k, t, Y_train, Y_validation))
S=np.mat(calculate.S(k, t, Y_train, Y_validation))
# In[]: Loop training

for l in range(maxiter):
    print ('**************************************     iter = ', l)
    # update Z
    Z_mu,Z_lsgms=train.updateZ(DGMM, X_train, Y_train, Y_mu, Y_lsgms, nb_epoch, batch_size, encoder)
    # update B
    B_mu,sigma_b=train.updateB(Z_lsgms, Z_mu, K, tau_mu, gamma_mu, Y_train, R_mu, H_mu)
    # update H
    H_mu,sigma_h=train.updateH(R_mu, numTrn, sigma_r, eta_mu, C, gamma_mu, Y_train, Z_mu, B_mu) 
    # update R
    R_mu,sigma_r=train.updateR(H_mu, D2, sigma_h, C, gamma_mu, Y_train, Z_mu, B_mu) 
    # calculate Y_mu   
    Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu) 
    
    # update tau
    tau_mu=train.updateTau(tau_alpha, K, D2, tau_beta, B_mu, sigma_b)
    # update eta
    eta_mu=train.updateEta(eta_alpha, C, D2, eta_beta, H_mu, sigma_h)
    # update gamma
    gamma_mu=train.updateGamma(gamma_alpha, numTrn, D2, Y_train, Z_mu, B_mu, R_mu, H_mu, gamma_beta)
    # calculate Y_lsgms
    Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2)))   

# In[]: save model and parameter results
# import pickle
# with open('dgmm.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([numTest, img_chns, img_rows, img_cols,H_mu,D2,sigma_h,gamma_mu,C,S,B_mu,rho,K,Y_test,Z_mu,L,resolution,X_test], f)

# imagereconstruct.save("dgmmmodel.h5")

# In[]: reconstruct X (image) from Y (fmri)
print("reconstruct X (image) from Y (fmri)")
X_reconstructed_mu = np.zeros((numTest, img_chns, img_rows, img_cols))#empty array
HHT = H_mu * H_mu.T + D2 * sigma_h
Temp = gamma_mu * np.mat(np.eye(D2)) - (gamma_mu**2) * (H_mu.T * (np.mat(np.eye(C)) + gamma_mu * HHT).I * H_mu)
for i in range(numTest):
    s=S[:,i]
    z_sigma_test = (B_mu * Temp * B_mu.T + (1 + rho * s.sum(axis=0)[0,0]) * np.mat(np.eye(K)) ).I#mencari variansi / kuadrat standar deviasi
    z_mu_test = (z_sigma_test * (B_mu * Temp * (np.mat(Y_test)[i,:]).T + rho * np.mat(Z_mu).T * s )).T#mencari nilai untuk inputan terbaik, biasanya dari rata2
    temp_mu = np.zeros((1,img_chns, img_rows, img_cols))#1,1,28,28
    epsilon_std = 1
    for l in range(L):#Looping untuk Monte Carlo Sampling
        epsilon=np.random.normal(0,epsilon_std,1)#ambil sampel acak atau noise
        z_test = z_mu_test + np.sqrt(np.diag(z_sigma_test))*epsilon#nilai utama ditambahkan dengan standar deviasi
        x_reconstructed_mu = imagereconstruct.predict(z_test, batch_size=1)#1,28,28,1
        #edit rolly move axis
        x_reconstructed_mu=np.moveaxis(x_reconstructed_mu,-1,1)
        temp_mu = temp_mu + x_reconstructed_mu # ati2 nih disini main tambahin aja
    x_reconstructed_mu = temp_mu / L # mendapatkan rata2 dari semua rekonstruksi citra monte carlo
    X_reconstructed_mu[i,:,:,:] = x_reconstructed_mu
    


# In[]:# visualization the reconstructed images, output in var X_reconstructed_mu
# =============================================================================
# n = 10
# for j in range(1):
#     plt.figure(figsize=(12, 2))    
#     for i in range(n):
#         # display original images
#         ax = plt.subplot(2, n, i +j*n*2 + 1)
#         plt.imshow(np.rot90(np.fliplr(X_test[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         # display reconstructed images
#         ax = plt.subplot(2, n, i + n + j*n*2 + 1)
#         plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()
# =============================================================================

# In[]: simpan ke dalam folder
# Existing code
stim = X_test[:, :, :, 0].reshape(10, 784)
rec = X_reconstructed_mu[:, 0, :, :].reshape(10, 784)


# Calculate IS
#is_score, is_std = calculate_inception_score(rec, batch_size=2, resize=True, splits=10)
#print('Inception Score:', is_score, '±', is_std)

from lib.fidis import save_array_as_image
# Save stim array as images
for i in range(len(stim)):
    save_array_as_image(np.rot90(np.fliplr(stim[i].reshape(28, 28))), f'stim/image_{i}.png')

# Save rec array as images
for i in range(len(rec)):
    save_array_as_image(np.rot90(np.fliplr(rec[i].reshape(28, 28))), f'rec/image_{i}.png')