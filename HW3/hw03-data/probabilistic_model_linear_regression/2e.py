import numpy as np
from numpy.random import normal, uniform
import matplotlib.pyplot as plt
from numpy import max, min


def likelihood_function(Y, X, w):
	print('\n')
	print('min_w: {}'.format((max(Y)-0.5)/min(X)))
	print('max_w: {}'.format((min(Y)+0.5)/max(X)))
	print('\n')

	if (Y-X*w < 0.5).all() and (Y-X*w >= -0.5).all():
		return 1
	else:
		return 0

for n in [5,25,125,625]:
	##generate n data points
	true_w = 2
	X = uniform(0.2, 0.3, size = n)
	Z = uniform(-0.5, 0.5, size = n)
	Y = X * true_w + Z

	####calculate likelihood as function of w
	W = np.arange(-5, 5, 0.02)
	N = W.shape[0]
	likelihood = np.zeros(N)
	for j in range(N):
		likelihood[j] = likelihood_function(Y, X, W[j])

	plt.plot(W, likelihood)
	plt.xlabel('w', fontsize=10)
	plt.ylabel('likelihood', fontsize=10)
	plt.title(['n=' + str(n)], fontsize=14)
	plt.savefig('{}.jpg'.format(n))
	plt.show()




