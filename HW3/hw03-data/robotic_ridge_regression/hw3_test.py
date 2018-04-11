import unittest
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np
from hw3 import *
from math import sqrt


class HW3_Test(unittest.TestCase):

	def setUp(self):
		self.hw3_sol = HW3_Sol()

	def test_compose_x(self):
		x_raw = np.arange(2*4*4*3).reshape((2, 4, 4, 3))
		self.assertTrue((x_raw[0, :, :, :]<48).all())
		self.assertTrue((x_raw[1, :, :, :]>=48).all())
		#print(x_raw)
		
		X = np.arange(2*4*4*3).reshape((2, 48))
		assert_array_equal(X[0, :], np.arange(48))
		assert_array_equal(X[1, :], np.arange(48, 96))
		#print(X)
		assert_array_equal(self.hw3_sol.compose_x(x_raw), X)

	def test_OLS_point_on_plane(self):
		X = np.arange(1, 5).reshape((2, 2))
		Pi = np.arange(1, 5).reshape((2, 2))
		U = X @ Pi
		#print(self.hw3_sol.OLS(X, U).shape)
		#print(Pi.shape)
		assert_almost_equal(self.hw3_sol.OLS(X, U), Pi)

	def test_error_zero(self):
		X = np.arange(1, 5).reshape((2, 2))
		Pi = np.arange(1, 5).reshape((2, 2))
		U = X @ Pi
		self.assertEqual(self.hw3_sol.error(X, U, Pi), 0)

	def test_error_one(self):
		X = np.arange(1, 5).reshape((2, 2))
		Pi = np.arange(1, 5).reshape((2, 2))
		U = X @ Pi + 1
		self.assertEqual(self.hw3_sol.error(X, U, Pi), 2)

	def test_standardize(self):
		X = np.arange(1, 5).reshape((2, 2))
		assert_almost_equal(self.hw3_sol.standardize(X), np.array([[1.0/255*2-1, 2.0/255*2-1], [3.0/255*2-1, 4.0/255*2-1]]))

	def test_kappa_diagonal_case(self):
		X = np.diag([1, 2, 3, 4])
		lambd = 1
		kappa = 17.0/2
		self.assertEqual(self.hw3_sol.kappa(X, lambd), kappa)


if __name__ == '__main__':
    unittest.main()