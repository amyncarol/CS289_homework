
import random
import time


import numpy as np
import numpy.linalg as LA


from numpy.linalg import inv
from numpy.linalg import det

from projection import Project2D, Projections


class QDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.01
		self.NUM_CLASSES = len(class_labels)



	def train_model(self,X,Y): 
		''''
		FILL IN CODE TO TRAIN MODEL
		MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

		'''

		X = np.array(X)
		Y = np.array(Y)
		self.means = np.zeros((self.NUM_CLASSES, X.shape[1]))
		self.covariance = np.zeros((self.NUM_CLASSES, X.shape[1], X.shape[1]))
		X_demean = np.zeros_like(X)

		for i in range(self.NUM_CLASSES):
			self.means[i, :] = np.mean(X[Y==i], axis = 0)
			X_demean[Y==i] = X[Y==i] - self.means[i, :]
			self.covariance[i, :, :] = 1/X[Y==i].shape[0] * X_demean[Y==i].T @ X_demean[Y==i] + self.reg_cov*np.eye(X.shape[1])

		
	def eval(self,x):
		''''
		Fill in code to evaluate model and return a prediction
		Prediction should be an integer specifying a class
		'''

		pred_array = np.zeros(self.NUM_CLASSES)
		x = np.array(x)

		for i in range(self.NUM_CLASSES):
			x_demean = (x-self.means[i, :]).reshape(x.shape[0], -1)
			pred_array[i] = (x_demean.T @ inv(self.covariance[i, :, :]) @ x_demean)[0, 0] + np.log(det(self.covariance[i, :, :]))
		return np.argmin(pred_array)
	
