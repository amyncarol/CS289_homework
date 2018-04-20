# You may want to install "gprof2dot"
import io
from collections import Counter

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

import pydot

eps = 1e-5  # a small number


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def information_gain(X, y, thresh):
        n = y.shape[0]
        p_y_0 = np.sum(y==0)/n
        H_y = DecisionTree.entropy(p_y_0)
      
        idx_under = (X < thresh)
        idx_above = (X >= thresh)

        n_under = np.sum(idx_under)
        n_above = n-n_under

        p_under_y_0 = np.sum(y[idx_under]==0)/n_under
        H_under = DecisionTree.entropy(p_under_y_0)
        
        p_above_y_0 = np.sum(y[idx_above]==0)/n_above
        H_above = DecisionTree.entropy(p_above_y_0)

        return H_y-(n_under/n*H_under+n_above/n*H_above)

    @staticmethod
    def entropy(p):
        if p==0 or p==1:
            return 0
        else:
            return -(p*np.log(p)+(1-p)*np.log(1-p))

    @staticmethod
    def gini_impurity(X, y, thresh):
        n = y.shape[0]
        
        idx_under = (X < thresh)
        idx_above = (X >= thresh)

        n_under = np.sum(idx_under)
        n_above = n-n_under

        p_under_y_0 = np.sum(y[idx_under]==0)/n_under
        G_under = DecisionTree.gini(p_under_y_0)
        
        p_above_y_0 = np.sum(y[idx_above]==0)/n_above
        G_above = DecisionTree.gini(p_above_y_0)

        return (n_under/n*G_under+n_above/n*G_above)
       
    @staticmethod
    def gini(p):
        return 1-p**2-(1-p)**2

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        for tree in self.decision_trees:
            X_tree = np.zeros_like(X)
            y_tree = np.zeros_like(y)
            for i in range(X.shape[0]):
                rand_i = np.random.choice(X.shape[0])
                X_tree[i, :] = X[rand_i, :]
                y_tree[i] = y[rand_i]
            tree = tree.fit(X_tree, y_tree)
        return self

    def predict(self, X):
        y_trees = np.zeros((X.shape[0], self.n))
        for i, tree in enumerate(self.decision_trees):
            y_trees[:, i] = tree.predict(X)
        return stats.mode(y_trees, axis=1).mode.reshape(X.shape[0])


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=1):   
        self.m = m   
        super().__init__(params, n)

    def fit(self, X, y):
        for tree in self.decision_trees:
            X_tree = np.zeros_like(X)
            y_tree = np.zeros_like(y)
            rand_js = np.random.choice(X.shape[1], size=self.m, replace=False)
            for i in range(X.shape[0]):
                rand_i = np.random.choice(X.shape[0])
                X_tree[i, rand_js] = X[rand_i, rand_js]
                y_tree[i] = y[rand_i]
            tree = tree.fit(X_tree, y_tree)
        return self

class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        for tree_i, tree in enumerate(self.decision_trees):
            X_tree = np.zeros_like(X)
            y_tree = np.zeros_like(y)
            rand_js = np.random.choice(X.shape[1], size=self.m, replace=False)
            for i in range(X.shape[0]):
                rand_i = np.random.choice(X.shape[0], p=self.w)
                X_tree[i, rand_js] = X[rand_i, rand_js]
                y_tree[i] = y[rand_i]
            tree = tree.fit(X_tree, y_tree)
            y_pred = tree.predict(X)
            I = (y != y_pred)
            e = np.dot(I, self.w)/np.sum(self.w)
            if e>0.5:
                continue
            self.a[tree_i] = 0.5*np.log(1/e-1)
            w1 = self.w * np.exp(self.a[tree_i]) * (y != y_pred)
            w2 = self.w * np.exp(-self.a[tree_i]) * (y == y_pred)
            self.w = (w1+w2)/np.sum(w1+w2)
        return self

    def predict(self, X):
        I = np.zeros((X.shape[0], self.n))
        for i, tree in enumerate(self.decision_trees):
            I[:, i] = tree.predict(X)
        z = np.tensordot(I, self.a/np.sum(self.a), axes=1)
        return (z >= 0.5).astype(int)
        


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)

def submit(y, filename):
    with open(filename, 'w') as f:
        for i in y:
            f.write('{}\n'.format(int(i)))

if __name__ == "__main__":
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=np.int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    # print("\n\nPart (a-b): simplified decision tree")
    # dt = DecisionTree(max_depth=3, feature_labels=features)
    # dt.fit(X, y)
    # print("Predictions", dt.predict(Z)[:100])

    # print("\n\nPart (c): sklearn's decision tree")
    # clf = sklearn.tree.DecisionTreeClassifier(random_state=0, **params)
    # clf.fit(X, y)
    # print(clf.score(X, y))
    # evaluate(clf)
    # print(clf)

    # ##plot the tree#########
    # out = io.StringIO()
    # sklearn.tree.export_graphviz(
    #     clf, out_file=out, feature_names=features, class_names=class_names)
    # graph = pydot.graph_from_dot_data(out.getvalue())
    # pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    # print("\n\nPart (e): bagged trees")
    # bagtrees = BaggedTrees(params = params, n=N)
    # bagtrees.fit(X, y)
    # print(bagtrees.score(X, y))
    # evaluate(bagtrees)
    # print(bagtrees)

    # print("\n\nPart (g): random forest")
    # for m in range(1, X.shape[1], 5):
       #  forest = RandomForest(params = params, n=N, m=m)
       #  forest.fit(X, y)
       #  print(forest.score(X, y))
       #  evaluate(forest)
       #  print(forest)

    # print("\n\nPart (i): boosted random forest")
    # for m in range(1, X.shape[1], 5):
       #  forest = BoostedRandomForest(params = params, n=N, m=m)
       #  forest.fit(X, y)
       #  print(forest.score(X, y))
       #  evaluate(forest)
       #  print(forest)

    # index = np.argsort(forest.w)
    # print(index[:10])
    # print(X[index[:10]])
    # print(y[index[:10]])

    # print(index[-10:])
    # print(X[index[-10:]])
    # print(y[index[-10:]])

    ##best model for titanic random forest m=10
    # forest = RandomForest(params = params, n=N, m=10)
    # forest.fit(X, y)
    # print(forest.score(X, y))
    # evaluate(forest)
    # print(forest)
    # y_pred = forest.predict(Z)
    # submit(y_pred, 'submission.txt')

    ##best model for spam boosted forest m=16
    forest = BoostedRandomForest(params = params, n=N, m=16)
    forest.fit(X, y)
    print(forest.score(X, y))
    evaluate(forest)
    print(forest)
    y_pred = forest.predict(Z)
    submit(y_pred, 'submission.txt')

    
    



