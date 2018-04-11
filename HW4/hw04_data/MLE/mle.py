import numpy as np
import matplotlib.pyplot as plt

def mean(samples):
	return np.average(samples, axis = 0)

def covariance(samples, mu_hat):
	n = samples.shape[0]
	d = samples.shape[1]
	mu_hat = mu_hat.reshape((d, 1))
	return (samples.T-mu_hat) @ (samples.T-mu_hat).T/n

if __name__ == '__main__':
	mu = [15, 5]
	sigmas = [[[20, 0], [0, 10]], [[20, 14], [14, 10]], [[20, -14], [-14, 10]]]
	for i, sigma in enumerate(sigmas):
		samples = np.random.multivariate_normal(mu, sigma, size=100000)
		mu_hat = mean(samples)
		sigma_hat = covariance(samples, mu_hat)
		print('when simga = {}, the MLE mean is {}, and MLE covariance is {}'.format(sigma, mu_hat, sigma_hat))
		print('the singular values for sigma matrix is {}'.format(np.linalg.svd(sigma, compute_uv = False)))
		fig = plt.figure()
		plt.scatter(samples[:, 0], samples[:, 1])
		plt.savefig('sigma{}.jpg'.format(i))
