
# coding: utf-8

# In[96]:


import numpy as np
import matplotlib.pyplot as plt

# Load the training dataset
train_features = np.load("train_features.npy")
train_labels = np.load("train_labels.npy").astype("int8")

n_train = train_labels.shape[0]

def visualize_digit(features, label):
    # Digits are stored as a vector of 400 pixel values. Here we
    # reshape it to a 20x20 image so we can display it.
    plt.imshow(features.reshape(20, 20), cmap="binary")
    plt.xlabel("Digit with label " + str(label))
    plt.show()

# Visualize a digit
# visualize_digit(train_features[0,:], train_labels[0])

# TODO: Plot three images with label 0 and three images with label 1

def images(a):
    counter = 0
    for i in range(20):
        if train_labels[i] == a:
            counter += 1
            visualize_digit(train_features[i,:], train_labels[i])
        if  counter==3:
            break
images(0)
images(1) 


# In[97]:


# Linear regression
from numpy.linalg import inv

# TODO: Solve the linear regression problem, regressing
# X = train_features against y = 2 * train_labels - 1

X = train_features
y = 2 * train_labels - 1

def get_w(X, y):
    XT = X.transpose()
    w = np.dot(inv(XT @ X) @ XT, y)
    return(w)

w = get_w(X, y)
res = np.dot(X, w) - y
res_2 = np.dot(res.transpose(), res)
# TODO: Report the residual error and the weight vector

print('the residual error is {} \n'.format(res_2))
print('w is a 400*1 vector: w = ')
print(w)


# In[98]:


# Load the test dataset
# It is good practice to do this after the training has been
# completed to make sure that no training happens on the test
# set!

from numpy import logical_xor, logical_not
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy").astype("int8")

n_test = test_labels.shape[0]

# TODO: Implement the classification rule and evaluate it
# on the training and test set

def correct_rate(features, labels, n, w, rule):
    predict = (np.dot(features, w) > rule)
    correct = logical_not(logical_xor(predict, labels))
    return float(sum(correct))/n


print('the correct pertentage by this model in the training \n         set is {} \n'.format(correct_rate(train_features, train_labels, n_train, w, 0)))

print('the correct pertentage by this model in the test \n         set is {} \n'.format(correct_rate(test_features, test_labels, n_test, w, 0)))


# In[99]:


# TODO: Try regressing against a vector with 0 for class 0
# and 1 for class 1

w2 = get_w(X, train_labels)
print('0 for 0, 1 for 1 model: the correct pertentage by this model in the training \n         set is {} \n'.format(correct_rate(train_features, train_labels, n_train, w2, 0.5)))

print('0 for 0, 1 for 1 model: the correct pertentage by this model in the test \n         set is {} \n'.format(correct_rate(test_features, test_labels, n_test, w2, 0.5)))


# In[100]:


# TODO: Form a new feature matrix with a column of ones added
# and do both regressions with that matrix

train_features_1 = np.hstack((train_features, np.ones((n_train, 1))))
test_features_1 = np.hstack((test_features, np.ones((n_test, 1))))

w3 = get_w(train_features_1, 2*train_labels-1)
print('train with bias column: the correct pertentage by this model in the training \n         set is {} \n'.format(correct_rate(train_features_1, train_labels, n_train, w3, 0)))

print('train with bias column: the correct pertentage by this model in the test \n         set is {} \n'.format(correct_rate(test_features_1, test_labels, n_test, w3, 0)))


w4 = get_w(train_features_1, train_labels)
print('train with bias column & 0 for 0 1 for 1: the correct pertentage by this model in the training \n         set is {} \n'.format(correct_rate(train_features_1, train_labels, n_train, w4, 0.5)))

print('train with bias column & 0 for 0 1 for 1: the correct pertentage by this model in the test \n         set is {} \n'.format(correct_rate(test_features_1, test_labels, n_test, w4, 0.5)))


# In[101]:


# Logistic Regression

# You can also compare against how well logistic regression is doing.
# We will learn more about logistic regression later in the course.

import sklearn.linear_model

lr = sklearn.linear_model.LogisticRegression()
lr.fit(X, train_labels)

test_error_lr = 1.0 * sum(lr.predict(test_features) != test_labels) / n_test
print(1-test_error_lr)

