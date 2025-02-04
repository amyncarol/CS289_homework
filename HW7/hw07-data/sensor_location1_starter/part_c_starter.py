from common import *
from part_b_starter import find_mle_by_grad_descent_part_b
import numpy as np
from numpy.linalg import norm

########################################################################
######### Part c #################################################
########################################################################
def log_likelihood(obj_loc, sensor_loc, distance): 
  """
  This function computes the log likelihood (as expressed in Part a).
  Input: 
  obj_loc: shape [1,2]
  sensor_loc: shape [7,2]
  distance: shape [7]
  Output: 
  The log likelihood function value. 
  """  
  # Your code: compute the log likelihood

  #print(sensor_loc.shape)
  #print(obj_loc.shape)
  sensor_loc_diff = sensor_loc - np.ones_like(sensor_loc) @ np.diag(obj_loc[0, :])
  sensor_loc_diff_norm = norm(sensor_loc_diff, axis = 1)
  term = sensor_loc_diff_norm - distance

  ll = - np.sum(term * term)

  return ll

if __name__ == "__main__":
  # ########################################################################
  # ######### Compute the function value at local minimum for all experiments.###
  # ########################################################################
  num_sensors = 20

  np.random.seed(100)
  sensor_loc = generate_sensors(k=num_sensors)

  # num_data_replicates = 10
  num_gd_replicates = 100

  obj_locs = [[[i,i]] for i in np.arange(0,1000,100)]

  func_values = np.zeros((len(obj_locs),10, num_gd_replicates))
  # record sensor_loc, obj_loc, 100 found minimas
  minimas = np.zeros((len(obj_locs), 10, num_gd_replicates, 2))
  true_object_locs = np.zeros((len(obj_locs), 10, 2))

  for i, obj_loc in enumerate(obj_locs): 
      for j in range(10):


          obj_loc, distance = generate_data_given_location(sensor_loc, obj_loc, 
                                                           k = num_sensors, d = 2)
          true_object_locs[i, j, :] = np.array(obj_loc)
          print('{}_{}'.format(i, j))


          for gd_replicate in range(num_gd_replicates): 
              initial_obj_loc = np.random.randn(1,2)* (100 * i+1)
  
              obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
                         sensor_loc, distance[0], lr=0.05, num_iters = 1000)
              minimas[i, j, gd_replicate, :] = np.array(obj_loc)
              func_value = log_likelihood(obj_loc, sensor_loc, distance[0])
              func_values[i, j, gd_replicate] = func_value

  ########################################################################
  ######### Calculate the things to be plotted. ###
  ########################################################################
  local_mins = [[np.unique(func_values[i,j].round(decimals=2)) for j in range(10)] for i in range(10)]
  num_local_min = [[len(local_mins[i][j]) for j in range(10)] for i in range(10)]
  proportion_global = [[sum(func_values[i,j].round(decimals=2) == min(local_mins[i][j]))*1.0/100 \
                         for j in range(10)] for i in range(10)]


  num_local_min = np.array(num_local_min)
  num_local_min = np.mean(num_local_min, axis = 1)

  proportion_global = np.array(proportion_global)
  proportion_global = np.mean(proportion_global, axis = 1)

  # ########################################################################
  # ######### Plots. #######################################################
  # ########################################################################
  fig, axes = plt.subplots(figsize=(8,6), nrows=2, ncols=1)
  fig.tight_layout()
  plt.subplot(211)

  plt.plot(np.arange(0,1000,100), num_local_min)
  plt.title('Number of local minimum found by 100 gradient descents.')
  plt.xlabel('Object Location')
  plt.ylabel('Number')
  #plt.savefig('num_obj.png')
  # Proportion of gradient descents that find the local minimum of minimum value. 

  plt.subplot(212)
  plt.plot(np.arange(0,1000,100), proportion_global)
  plt.title('Proportion of GD that finds the global minimum among 100 gradient descents.')
  plt.xlabel('Object Location')
  plt.ylabel('Proportion')
  fig.tight_layout()
  plt.savefig('prop_obj.png')

  ########################################################################
  ######### Plots of contours. ###########################################
  ########################################################################
  np.random.seed(0) 
  sensor_loc = np.random.randn(num_sensors,2) * 100
  x = np.arange(-400,400, 4)
  y = np.arange(-400,400, 4)
  X, Y = np.meshgrid(x, y) 
  obj_loc = [[0,0]]
  obj_loc, distance = generate_data_given_location(sensor_loc, 
                                                   obj_loc, k = num_sensors, d = 2)

  #print(distance)

  Z =  np.array([[log_likelihood(np.array([[X[i,j],Y[i,j]]]), 
                                 sensor_loc, distance[0]) for j in range(len(X))] \
                 for i in range(len(X))]) 


  plt.figure(figsize=(10,4))
  plt.subplot(121)
  CS = plt.contour(X, Y, Z, 200)
  plt.clabel(CS, inline=1, fontsize=10)
  plt.title('With object at (0,0)')
  #plt.show()

  np.random.seed(0) 
  sensor_loc = np.random.randn(num_sensors,2) * 100
  x = np.arange(-400,400, 4)
  y = np.arange(-400,400, 4)
  X, Y = np.meshgrid(x, y) 
  obj_loc = [[200,200]]
  obj_loc, distance = generate_data_given_location(sensor_loc, 
                                                   obj_loc, k = num_sensors, d = 2)
  #print(distance)

  Z =  np.array([[log_likelihood(np.array([[X[i,j],Y[i,j]]]), 
                                 sensor_loc, distance[0]) for j in range(len(X))] \
                 for i in range(len(X))]) 


  # Create a simple contour plot with labels using default colors.  The
  # inline argument to clabel will control whether the labels are draw
  # over the line segments of the contour, removing the lines beneath
  # the label
  #plt.figure()
  plt.subplot(122)
  CS = plt.contour(X, Y, Z, 200)
  plt.clabel(CS, inline=1, fontsize=10)
  plt.title('With object at (200,200)')
  #plt.show()
  plt.savefig('likelihood_landscape.png')


  ########################################################################
  ######### Plots of Found local minimas. ###########################################
  ########################################################################
  #sensor_loc
  #minimas = np.zeros((len(obj_locs), 10, num_gd_replicates, 2))
  #true_object_locs = np.zeros((len(obj_locs), 10, 2))
  object_loc_i = 5
  trail = 0

  print(minimas[object_loc_i, trail, :, :])

  plt.figure()
  plt.plot(sensor_loc[:, 0], sensor_loc[:, 1], 'r+', label="sensors")
  plt.plot(minimas[object_loc_i, trail, :, 0], minimas[object_loc_i, trail, :, 1], 'g.', label="minimas")
  plt.plot(true_object_locs[object_loc_i, trail, 0], true_object_locs[object_loc_i, trail, 1], 'b*', label="object")
  plt.title('object at location (%d, %d), gradient descent recovered locations' % (object_loc_i*100, object_loc_i*100))
  plt.legend()
  plt.axis('equal')
  plt.savefig('2D_vis.png')


