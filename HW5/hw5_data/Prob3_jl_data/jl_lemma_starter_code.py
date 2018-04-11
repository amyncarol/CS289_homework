import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from numpy.linalg import svd


######## PROJECTION FUNCTIONS ##########

## Random Projections ##
def random_matrix(d, k):
	'''
	d = original dimension
	k = projected dimension
	'''
	return 1./np.sqrt(k)*np.random.normal(0, 1, (d, k))

def random_proj(X, k):
	_, d= X.shape
	return X.dot(random_matrix(d, k))

## PCA and projections ##
def my_pca(X, k):
	'''
	compute PCA components
	X = data matrix (each row as a sample)
	k = #principal components
	'''
	n, d = X.shape
	assert(d>=k)
	_, _, Vh = np.linalg.svd(X)    
	V = Vh.T
	return V[:, :k]

def pca_proj(X, k):
	'''
	compute projection of matrix X
	along its first k principal components
	'''
	P = my_pca(X, k)
	# P = P.dot(P.T)
	return X.dot(P)


######### LINEAR MODEL FITTING ############

def rand_proj_accuracy_split(X, y, k):
	'''
	Fitting a k dimensional feature set obtained
	from random projection of X, versus y
	for binary classification for y in {-1, 1}
	'''
	
	# test train split
	_, d = X.shape
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
	
	# random projection
	J = np.random.normal(0., 1., (d, k))
	rand_proj_X = X_train.dot(J)
	
	# fit a linear model
	line = sklearn.linear_model.LinearRegression(fit_intercept=False)
	line.fit(rand_proj_X, y_train)
	
	# predict y
	y_pred=line.predict(X_test.dot(J))
	
	# return the test error
	return 1-np.mean(np.sign(y_pred)!= y_test)

def pca_proj_accuracy(X, y, k):
	'''
	Fitting a k dimensional feature set obtained
	from PCA projection of X, versus y
	for binary classification for y in {-1, 1}
	'''

	# test-train split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

	# pca projection
	P = my_pca(X_train, k)
	P = P.dot(P.T)
	pca_proj_X = X_train.dot(P)
				
	# fit a linear model
	line = sklearn.linear_model.LinearRegression(fit_intercept=False)
	line.fit(pca_proj_X, y_train)
	
	 # predict y
	y_pred=line.predict(X_test.dot(P))
	

	# return the test error
	return 1-np.mean(np.sign(y_pred)!= y_test)


######## LOADING THE DATASETS #########

# to load the data:

######### YOUR CODE GOES HERE ##########

# Using PCA and Random Projection for:
# Visualizing the datasets 

#fig, axs = plt.subplots(2, 3, figsize = (12, 8))
#for i in range(3):
#	data = np.load('data'+str(i+1)+'.npz')
#	X = data['X']
#	y = data['y']
#	n, d = X.shape

#	pos_index = y == +1
#	neg_index = y == -1

#	X_pca = pca_proj(X, 2)
#	axs[0, i].scatter(X_pca[pos_index, 0], X_pca[pos_index, 1], color = 'r', marker = '+')
#	axs[0, i].scatter(X_pca[neg_index, 0], X_pca[neg_index, 1], color = 'b', marker = 'o')
#	axs[0, i].set_title('dataset' + str(i+1) + ' PCA')

#	X_random = random_proj(X, 2)
#	axs[1, i].scatter(X_random[pos_index, 0], X_random[pos_index, 1], color = 'r', marker = '+')
#	axs[1, i].scatter(X_random[neg_index, 0], X_random[neg_index, 1], color = 'b', marker = 'o')
#	axs[1, i].set_title('dataset' + str(i+1) + ' random')

#plt.savefig('3h.jpg')
#plt.show()

# Computing the accuracies over different datasets.

#n_trials = 10

#for i in range(3):
#	data = np.load('data'+str(i+1)+'.npz')
#	X = data['X']
#	y = data['y']
#	n, d = X.shape

#	accuracies = np.zeros(d)
#	for k in range(1, d+1):
#		accuracies[k-1] = pca_proj_accuracy(X, y, k)
#	plt.plot(range(1, d+1), accuracies, label = 'PCA_' + 'dataset' + str(i+1))

#	accuracies = np.zeros((n_trials, d))
#	for j in range(n_trials):
#		for k in range(1, d+1):
#			accuracies[j, k-1] = rand_proj_accuracy_split(X, y, k)
#	plt.plot(range(1, d+1), np.average(accuracies, axis=0), label = 'random_' + 'dataset' + str(i+1))

#plt.legend()
#plt.savefig('3i.jpg')
#plt.show()


# And computing the SVD of the feature matrix

for i in range(3):
	data = np.load('data'+str(i+1)+'.npz')
	X = data['X']
	y = data['y']
	n, d = X.shape

	s =  svd(X, compute_uv = False)
	plt.plot(range(1, d+1), s, label = 'dataset'+str(i+1))

plt.legend()
plt.savefig('3j.jpg')
plt.show()



######## YOU CAN PLOT THE RESULTS HERE ########

# plt.plot, plt.scatter would be useful for plotting









