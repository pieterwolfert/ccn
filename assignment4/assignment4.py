import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import inv
data = loadmat('69dataset.mat')
labels = data['labels']
X = data['X']
Y = data['Y']
# TODO: compute pixel mean and voxel mean
pixel_mean = np.zeros((1,784))
num_images = X.shape[0]
voxel_mean = np.zeros((1,3092))
for i in range(num_images):
    pixel_mean += X[i,:]/num_images
    voxel_mean += Y[i,:]/num_images
X_norm = (X - pixel_mean) / np.std(X)
Y_norm = (Y - voxel_mean) / np.std(Y)

X_train, X_test = np.concatenate((X_norm[10:50,:], X_norm[50:90,:])), np.concatenate((X_norm[0:10,:], X_norm[90:-1,:]))
Y_train, Y_test = np.concatenate((Y_norm[10:50,:], Y_norm[50:90,:])), np.concatenate((Y_norm[0:10,:], Y_norm[90:-1,:]))

# Excercise 1
I_l = np.mat(10**-6 * np.identity(3092))
B = inv(np.mat(Y_train).T * np.mat(Y_train) + I_l) * np.mat(Y_train).T * np.mat(X_train)

x_test = B.T * Y_test.T
fig = plt.figure(figsize=(6,6))
# reconstructed images
for i in range(19):
    image = np.reshape(x_test.T[i,:],(28,28)).T
    sub = fig.add_subplot(4,5,i+1)
    sub.imshow(image)
    
# TODO: invert normalisation
# acutal images
fig = plt.figure(figsize=(6,6))
for i in range(19):
    image = np.reshape(X_test[i,:],(28,28)).T
    sub = fig.add_subplot(4,5,i+1)
    sub.imshow(image)