import unittest
import numpy as np
from math import sqrt
from part_b_starter import compute_gradient_of_likelihood
from part_c_starter import log_likelihood
from numpy.testing import assert_array_equal, assert_array_almost_equal

class Test(unittest.TestCase):

    def test_compute_gradient_of_likelihood(self):
        single_obj_loc = np.array([[0, 0]])
        sensor_loc = np.array([[1, 1], [1, -1], [0, 2]])
        single_distance = np.array([sqrt(2), sqrt(2), 2])
        assert_array_equal(np.array([0, 0]), compute_gradient_of_likelihood(single_obj_loc, sensor_loc, 
                                single_distance))

    def test_compute_gradient_of_likelihood_2(self):
        single_obj_loc = np.array([[0, 0]])
        sensor_loc = np.array([[1, 1], [1, -1], [0, 2]])
        single_distance = np.array([sqrt(1), sqrt(2), 2])
        assert_array_almost_equal(np.array([sqrt(2)-2, sqrt(2)-2]), compute_gradient_of_likelihood(single_obj_loc, sensor_loc, 
                                single_distance))

    def test_log_likelihood(self):
        obj_loc = np.array([[0, 0]])
        sensor_loc = np.array([[1, 1], [1, -1], [0, 2]])
        distance = np.array([sqrt(2), sqrt(2), 2])
        self.assertEqual(0.0, log_likelihood(obj_loc, sensor_loc, distance))

        

if __name__ == '__main__':
    unittest.main()