import unittest
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np
from kernels import *
from math import sqrt, exp


class Kernels_Test(unittest.TestCase):

	def setUp(self):
		pass

	def test_ploss(self):
		X = np.array([[1, 2],[2, 0], [1, 3]])
		y = np.array([[5], [10], [7]])
		w = np.array([[1], [2]])
		self.assertEqual(ploss(X, y, w), 64.0/3.0)

	def test_assemble_feature(self):
		X = np.array([[1, 2],[2, 0]])
		p = 2
		X_poly = np.array([[1, 1, 2, 1, 2, 4], [1, 2, 0, 4, 0, 0]])
		assert_array_equal(assemble_feature(X, p), X_poly)

	def test_poly_kernel(self):
		x = np.array([[1], [2]])
		z = np.array([[1], [3]])
		p = 2
		self.assertEqual(poly_kernel(x, z, p), 64.0)

	def test_rbf_kernel(self):
		x = np.array([[1], [2]])
		z = np.array([[1], [3]])
		sigma = 2
		self.assertEqual(rbf_kernel(x, z, sigma), exp(-1.0/8.0))

	def test_kernel_matrix_poly(self):
		X = np.array([[1, 2]])
		Y = np.array([[0, 1], [3, 2]])
		Z = np.array([[9, 64]])
		assert_array_equal(kernel_matrix(X, Y, 'poly', 2), Z)

	def test_kernel_matrix_rbf(self):
		X = np.array([[1, 2]])
		Y = np.array([[0, 1], [3, 2]])
		Z = np.array([[exp(-2.0/8.0), exp(-4.0/8.0)]])
		assert_array_equal(kernel_matrix(X, Y, 'rbf', 2), Z)


if __name__ == '__main__':
    unittest.main()
