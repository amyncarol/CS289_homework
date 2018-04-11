import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, norm, svd

class HW3_Sol(object):

    def __init__(self):
        pass

    def load_data(self):
        self.x_train = np.float32(pickle.load(open('x_train.p','rb'), encoding='latin1'))
        self.y_train = np.float32(pickle.load(open('y_train.p','rb'), encoding='latin1'))
        self.x_test = np.float32(pickle.load(open('x_test.p','rb'), encoding='latin1'))
        self.y_test = np.float32(pickle.load(open('y_test.p','rb'), encoding='latin1'))

    def compose_x(self, x_raw):
        n = x_raw.shape[0]
        d = x_raw.shape[1]*x_raw.shape[2]*x_raw.shape[3]
        return x_raw.reshape((n, d))

    def OLS(self, X, U):
        return inv(X.T @ X) @ X.T @ U

    def ridge(self, X, U, lambd):
        d = X.shape[1]
        return inv(X.T @ X + lambd * np.eye(d)) @ X.T @ U

    def error(self, X, U, Pi):
        n = U.shape[0]
        f_norm = norm(X @ Pi - U, ord = 'fro')
        return 1.0/n * f_norm**2

    def standardize(self, X):
        return X/255.0 * 2 - 1

    def kappa(self, X, lambd):
        d = X.shape[1]
        A = X.T @ X + lambd * np.eye(d)
        s = svd(A, compute_uv = False)
        return s[0]/s[-1]

    def visualize(self, i):
        plt.imshow(self.x_train[i])
        plt.savefig('training_image_{}'.format(i))
        plt.show()

if __name__ == '__main__':

    hw3_sol = HW3_Sol()

    hw3_sol.load_data()

    ##############(a)################
    for i in [0, 10, 20]:
        hw3_sol.visualize(i)
        print('the control vectors of image {} is {}'.format(i, hw3_sol.y_train[i]))


    ###############(b)###############
    X = hw3_sol.compose_x(hw3_sol.x_train)
    U = hw3_sol.y_train
    X_val = hw3_sol.compose_x(hw3_sol.x_test)
    U_val = hw3_sol.y_test
    #Pi = hw3_sol.OLS(X, U)

    print('we cannot do inversion with a 2700*2700 singular matrix since the rank of X.T @ X is at most n, which is 91 in this case')
    ###############(c+e)###############
    print('below is the output without standardization:')
    error = np.zeros(5)
    error_validation = np.zeros(5)
    for i, lambd in enumerate([0.1, 1, 10, 100, 1000]):
        Pi = hw3_sol.ridge(X, U, lambd)
        error[i] = hw3_sol.error(X, U, Pi)
        error_validation[i] = hw3_sol.error(X_val, U_val, Pi)


    print('the training errors for lambda = {} are {}'.format([0.1, 1, 10, 100, 1000], error))
    print('the validation errors for lambda = {} are {}'.format([0.1, 1, 10, 100, 1000], error_validation))

    ###############(d+e)###############
    print('below is the output with standardization:')
    X = hw3_sol.standardize(X)
    X_val = hw3_sol.standardize(X_val)

    error = np.zeros(5)
    error_validation = np.zeros(5)
    for i, lambd in enumerate([0.1, 1, 10, 100, 1000]):
        Pi = hw3_sol.ridge(X, U, lambd)
        error[i] = hw3_sol.error(X, U, Pi)
        error_validation[i] = hw3_sol.error(X_val, U_val, Pi)

    
    print('the training errors for lambda = {} are {}'.format([0.1, 1, 10, 100, 1000], error))
    print('the validation errors for lambda = {} are {}'.format([0.1, 1, 10, 100, 1000], error_validation))

    print('as lambda increase, bias increases(training error reflects bias), and variance decreases(validation error reflects bias+variance)')

    ###############(f)###############
    X = hw3_sol.compose_x(hw3_sol.x_train)
    print('the condition number without standardization is {}'.format(hw3_sol.kappa(X, 100)))

    X = hw3_sol.standardize(X)
    print('the condition number with standardization is {}'.format(hw3_sol.kappa(X, 100)))
    