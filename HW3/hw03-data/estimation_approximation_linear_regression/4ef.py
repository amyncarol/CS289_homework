import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, normal
from numpy.linalg import inv, norm

# assign problem parameters
w1 = 1
w0 = 1
#interval = [-1, 1]
interval = [-4, 3]

# generate data
# np.random might be useful
def error_function(D, n, func = 'p', repeat = 40):
	error = np.zeros(repeat)
	for j in range(repeat):
		alpha = uniform(interval[0], interval[1], n).reshape((n, 1))
		noise = normal(size = n).reshape((n, 1))
		print(alpha)
		print(noise)
		X = np.ones((n, 1))
		for i in range(1, D+1):
			X = np.hstack((X, alpha**i))
		print(X)

		if func == 'p':
			y_true = w1 * alpha + w0
		if func == 'exp':
			y_true = np.exp(alpha)

		print(y_true)

		y_noise = y_true + noise

		w_hat = inv(X.T @ X) @ X.T @ y_noise

		error[j] = norm(X @ w_hat - y_true)**2/n
	return np.mean(error)
	

# fit data with different models
# np.polyfit and np.polyval might be useful


# plotting figures
# sample code
plt.figure()
plt.subplot(121)

deg = 20
error = np.zeros(deg)
n = 120
for i in range(deg):
	error[i] = error_function(i+1, n, 'exp')

plt.semilogy(np.arange(1, deg+1), error, 'o', color = 'b')
plt.xlabel('degree of polynomial')
plt.ylabel('error')

plt.subplot(122)
error = np.zeros(10)

D = 4
for i in range(10):
	error[i] = error_function(D, (i+1)*20, 'exp')
plt.plot(np.arange(20, 220, 20), error, 'o', color = 'b')

plt.xlabel('number of samples')
plt.ylabel('error')
plt.savefig('4f.jpg')
plt.show()
