import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from starter import *


def neural_network(X, Y, X_test, Y_test, num_neurons, activation):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_neurons: number of neurons in each layer
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    # Build the model
    if activation == "ReLU":
        activation_func = ReLUActivation
    if activation == "tanh":
        activation_func = TanhActivation

    model = Model(X.shape[1])
    model.addLayer(DenseLayer(num_neurons, activation_func()))
    model.addLayer(DenseLayer(num_neurons, activation_func()))
    model.addLayer(DenseLayer(Y.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())

    ##train the model and plot learning curve
    ##standardize the features first!!!
    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)
    hist = model.train(X_trans, Y, 2000, GDOptimizer(eta=0.001))
    # plt.plot(hist)
    # plt.title('Learning curve')
    # plt.show()

    ##evaluate the model
    X_test_trans = scaler.transform(X_test)
    Y_pred = model.predict(X_test_trans)
    mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
    return mse


#############################################################################
#######################PLOT PART 2###########################################
#############################################################################
def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
    return generate_dataset(
        sensor_loc,
        num_sensors=k,
        spatial_dim=d,
        num_data=n,
        original_dist=original_dist,
        noise=noise)


np.random.seed(0)
n = 200
num_neuronss = np.arange(100, 550, 50)
mses = np.zeros((len(num_neuronss), 2))

# for s in range(replicates):

sensor_loc = generate_sensors()
X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
X_test, Y_test = generate_data(sensor_loc, n=1000)
for t, num_neurons in enumerate(num_neuronss):
    ### Neural Network:
    mse = neural_network(X, Y, X_test, Y_test, num_neurons, "ReLU")
    mses[t, 0] = mse

    mse = neural_network(X, Y, X_test, Y_test, num_neurons, "tanh")
    mses[t, 1] = mse

    print('Experiment with {} neurons done...'.format(num_neurons))

### Plot MSE for each model.
plt.figure()
activation_names = ['ReLU', 'Tanh']
for a in range(2):
    plt.plot(num_neuronss, mses[:, a], label=activation_names[a])

plt.title('Error on validation data verses number of neurons')
plt.xlabel('Number of neurons')
plt.ylabel('Average Error')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('num_neurons.png')
