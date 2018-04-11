from common import *
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
	
	single_distance: k dimensional numpy array. 
	Observed distance of the object.
	
	Output:
	grad: d-dimensional numpy array.
	
	"""
	loc_difference = single_obj_loc - sensor_loc # k * d.
	phi = np.linalg.norm(loc_difference, axis = 1) # k. 
	weight = (phi - single_distance) / phi # k.
	
	grad = -np.sum(np.expand_dims(weight,1)*loc_difference, 
				   axis = 0) # d
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
	obj_loc = initial_obj_loc
	for t in range(num_iters):
		obj_loc += lr * compute_gradient_of_likelihood(obj_loc, 
						  sensor_loc, single_distance) 
		
	return obj_loc
	
########################################################################
#########  MAIN ########################################################
########################################################################
if __name__ == "__main__":
	np.random.seed(0)
	sensor_loc = generate_sensors()
	obj_loc, distance = generate_data(sensor_loc)
	single_distance = distance[0]
	print('The real object location is')
	print(obj_loc)
	# Initialized as [0,0]
	initial_obj_loc = np.array([[0.,0.]]) 
	estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
			   sensor_loc, single_distance, lr=0.001, num_iters = 10000)
	print('The estimated object location with zero initialization is')
	print(estimated_obj_loc)

	# Random initialization.
	initial_obj_loc = np.random.randn(1,2)
	estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
			   sensor_loc, single_distance, lr=0.001, num_iters = 10000)
	print('The estimated object location with random initialization is')
	print(estimated_obj_loc)   