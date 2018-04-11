import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.svm import LinearSVC
from projection import Project2D, Projections


class LDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.001
		self.NUM_CLASSES = len(class_labels)



	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

		'''
		
		X = np.array(X)
		Y = np.array(Y)
		self.means = np.zeros((self.NUM_CLASSES, X.shape[1]))
		X_demean = np.zeros_like(X)

		for i in range(self.NUM_CLASSES):
			self.means[i, :] = np.mean(X[Y==i], axis = 0)
			X_demean[Y==i] = X[Y==i] - self.means[i, :]
		
		self.covariance = 1/Y.shape[0] * X_demean.T @ X_demean + self.reg_cov*np.eye(X.shape[1])

	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''
		pred_array = np.zeros(self.NUM_CLASSES)
		x = np.array(x)

		for i in range(self.NUM_CLASSES):
			x_demean = (x-self.means[i, :]).reshape(x.shape[0], -1)
			pred_array[i] = (x_demean.T @ inv(self.covariance) @ x_demean)[0, 0]
		return np.argmin(pred_array)








