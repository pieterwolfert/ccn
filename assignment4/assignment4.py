import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import inv
data = loadmat('69dataset.mat')
labels = data['labels']
X = data['X'].astype(float)
Y = data['Y']
X_prior = data['prior'].astype(float)

X_mean, X_std  = np.reshape(np.mean(X,axis=0),(1,784)), np.reshape(np.std(X,axis=0),(1,784))
Y_mean, Y_std = np.reshape(np.mean(Y,axis=0),(1,3092)), np.reshape(np.std(Y,axis=0),(1,3092))
Xp_mean, Xp_std  = np.reshape(np.mean(X_prior,axis=0),(1,784)), np.reshape(np.std(X_prior,axis=0),(1,784))
X_norm = (X - X_mean) / X_std
X_norm[np.isnan(X_norm)] = 0
Y_norm = (Y - Y_mean) / Y_std
X_normp = (X_prior - Xp_mean) / Xp_std
X_normp[np.isnan(X_normp)] = 0

X_train, X_test = np.concatenate((X_norm[10:50,:], X_norm[50:90,:])), np.concatenate((X_norm[0:10,:], X_norm[90:-1,:]))
Y_train, Y_test = np.concatenate((Y_norm[10:50,:], Y_norm[50:90,:])), np.concatenate((Y_norm[0:10,:], Y_norm[90:-1,:]))

# Excercise 1
I_l = np.mat(10**-6 * np.identity(3092))
B = inv(np.mat(Y_train).T * np.mat(Y_train) + I_l) * np.mat(Y_train).T * np.mat(X_train)

x_test = B.T * Y_test.T
# Reconstruct normalized images
x_test = np.array(x_test).T*X_std + X_mean
X_test = X_test * X_std + X_mean
# acutal images
fig = plt.figure(figsize=(6,6))
for i in range(19):
    image = np.reshape(X_test[i,:],(28,28)).T
    sub = fig.add_subplot(4,5,i+1)
    sub.imshow(image)

fig = plt.figure(figsize=(6,6))
# reconstructed images excercise 1
plt.title('Linear reconstruction')
for i in range(19):
    image = np.reshape(x_test[i,:],(28,28)).T
    sub = fig.add_subplot(4,5,i+1)
    sub.imshow(image)
    

# Excercise 2
sigma = np.mat(10**-3 *np.identity(784))
lI = np.mat(10**-6*np.identity(784))
sigma_prior = (np.mat(X_prior).T * np.mat(X_prior)) / (np.shape(X_prior)[0] - 1)
# add regularisation
sigma_prior += np.mat(10**-6*np.identity(784))
# visualize sigma prior
fig = plt.figure(figsize=(6,6))
plt.title('Sigma prior')
plt.imshow(sigma_prior)
B = inv(np.mat(X_train).T*X_train +lI)*np.mat(X_train).T*np.mat(Y_train)
p1 = inv(inv(sigma_prior) + (B.T*inv(sigma)).T*B.T)
mu_post = ((p1*B).T*inv(sigma)).T*np.mat(Y_test).T
mu_post = np.array(mu_post).T*X_std + X_mean

fig = plt.figure(figsize=(6,6))
plt.title('Baysian reconstruction')
for i in range(19):
    image = np.reshape(mu_post[i,:],(28,28)).T
    sub = fig.add_subplot(4,5,i+1)
    sub.imshow(image)