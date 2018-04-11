import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, uniform
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def likelihood_function(X, Y, var, w):
    d = X.shape[1]
    mean = (inv(1.0/var * np.eye(d) + X.T @ X) @ X.T @ Y).ravel()
    covar = inv(1.0/var * np.eye(d) + X.T @ X)
    return multivariate_normal.pdf(w, mean=mean, cov=covar)

def mean_function(X, Y, var):
    d = X.shape[1]
    return (inv(1.0/var * np.eye(d) + X.T @ X) @ X.T @ Y).ravel()

fig , ax = plt.subplots(3, 3)

for axi, n in enumerate([5, 25, 125]):
    # generate data 
    w_true = np.array([[1], [2]])
    X = uniform(size = 2*n).reshape((n, 2))
    Z = normal(size = n).reshape((n, 1))
    Y = X @ w_true + Z
    
    for axj, var in enumerate([1, 4, 9]):
        print(mean_function(X, Y, var))
        # compute likelihood 
        W0 = np.arange(0, 4, 0.1)
        W1 = np.arange(0, 4, 0.1)
        N = W0.shape[0]
        likelihood = np.ones([N,N]) # likelihood as a function of w_1 and w_0                  
        for i in range(N): 
            for j in range(N): 
                w = np.array([W0[i], W1[j]])
                likelihood[i, j] = likelihood_function(X, Y, var, w)
                

        # plotting the likelihood

        # for 2D likelihood using imshow
        ax[axi, axj].imshow(likelihood, cmap='hot', aspect='auto',extent=[W1.min(), W1.max(), W0.max(), W0.min()])
        ax[axi, axj].set_xlabel('w1')
        ax[axi, axj].set_ylabel('w0')
        ax[axi, axj].set_title('n = {}, sigma^2 = {}'.format(n, var), fontsize = 10)

plt.savefig('2i.jpg')
plt.tight_layout()
plt.show()
    