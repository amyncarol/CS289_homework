import unittest
from decision_tree_starter import DecisionTree
import numpy as np

class TestDecisionTree(unittest.TestCase):

    def test_information_gain(self):
        X = np.array([1, 4, 2, 6])
        y = np.array([1, 0, 0, 1])
        thresh = 3
        gain = DecisionTree.information_gain(X, y, thresh)
        H_y = -0.5*np.log(0.5)*2
        H_above = -0.5*np.log(0.5)*2
        H_under = -0.5*np.log(0.5)*2
        self.assertEqual(gain, H_y-0.5*H_above-0.5*H_under)

    def test_information_gain_2(self):
        X = np.array([1, 4, 2, 6])
        y = np.array([1, 0, 1, 0])
        thresh = 3
        gain = DecisionTree.information_gain(X, y, thresh)
        H_y = -0.5*np.log(0.5)*2
        H_above = 0
        H_under = 0
        self.assertEqual(gain, H_y-0.5*H_above-0.5*H_under)

    def test_gini_impurity(self):
        X = np.array([1, 4, 2, 6])
        y = np.array([1, 0, 0, 1])
        thresh = 3
        gini = DecisionTree.gini_impurity(X, y, thresh)
        G_above = 0.5
        G_under = 0.5
        self.assertEqual(gini, 0.5*G_above+0.5*G_under)

    def test_gini_impurity_2(self):
        X = np.array([1, 4, 2, 6])
        y = np.array([1, 0, 1, 0])
        thresh = 3
        gini = DecisionTree.gini_impurity(X, y, thresh)
        G_above = 0
        G_under = 0
        self.assertEqual(gini, 0.5*G_above+0.5*G_under)

if __name__ == '__main__':
    unittest.main()