from common import *
from math import sqrt
import numpy as np
from numpy.linalg import norm
########################################################################
######### Part b ###################################
########################################################################

########################################################################
#########  Gradient Computing and MLE ###################################
########################################################################
def compute_gradient_of_likelihood(single_obj_loc, sensor_loc, 
								single_distance):
	"""
	Compute the gradient of the loglikelihood function for part a.   
	
	Input:
	single_obj_loc: 1 * d numpy array. 
	Location of the single object.
	
	sensor_loc: k * d numpy array. 
	Location of sensor.
	
	single_distance: k dimensional numpy array. (k: number of sensors, d: dimensionality)
	Observed distance of the object.
	
	Output:
	grad: d-dimensional numpy array.
	
	"""
	#Your code: implement the gradient of loglikelihood

	#grad = np.zeros_like(single_obj_loc)

	#print(single_obj_loc)
	#print(np.diag(single_obj_loc[0, :]))
	#print(np.ones_like(sensor_loc))
	sensor_loc_diff = sensor_loc - np.ones_like(sensor_loc) @ np.diag(single_obj_loc[0, :])
	#print(sensor_loc)
	#print(sensor_loc_diff)
	sensor_loc_diff_norm = norm(sensor_loc_diff, axis = 1)
	#print('\n')
	#print(sensor_loc_diff_norm.shape)
	#print(single_distance.shape)
	second_term = 1 - single_distance/sensor_loc_diff_norm
	#print(second_term)

	grad = -2*sensor_loc_diff.T @ second_term
	#print(grad)

	return grad

def find_mle_by_grad_descent_part_b(initial_obj_loc, 
		   sensor_loc, single_distance, lr=0.001, num_iters = 10000):
	"""
	Compute the gradient of the loglikelihood function for part a.   
	
	Input:
	initial_obj_loc: 1 * d numpy array. 
	Initialized Location of the single object.
	
	sensor_loc: k * d numpy array. Location of sensor.
	
	single_distance: k dimensional numpy array. 
	Observed distance of the object.
	
	Output:
	obj_loc: 1 * d numpy array. The mle for the location of the object.
	
	"""    
	# Your code: do gradient descent
	obj_loc = initial_obj_loc
	for i in range(num_iters):
		obj_loc = obj_loc - lr * compute_gradient_of_likelihood(obj_loc, sensor_loc, 
								single_distance)
	return obj_loc

if __name__ == "__main__":	
	########################################################################
	#########  MAIN ########################################################
	########################################################################

	# Your code: set some appropriate learning rate here
	lr = 0.001

	np.random.seed(0)
	sensor_loc = generate_sensors()
	obj_loc, distance = generate_data(sensor_loc)
	single_distance = distance[0]
	print('The real object location is')
	print(obj_loc)
	# Initialized as [0,0]
	initial_obj_loc = np.array([[0.,0.]]) 
	estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
			   sensor_loc, single_distance, lr=lr, num_iters = 1000)
	print('The estimated object location with zero initialization is')
	print(estimated_obj_loc)

	# Random initialization.
	initial_obj_loc = np.random.randn(1,2)*100+100
	#initial_obj_loc = np.array([[44.38, 33.36]])
	estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
			   sensor_loc, single_distance, lr=lr, num_iters = 1000)
	print('The estimated object location with random initialization is')
	print(estimated_obj_loc)   