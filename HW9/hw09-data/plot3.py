import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler

from starter import *

def get_num_neurons(N, k):
    """
    given number of parameters N and number of hidden layer k, returns the number of neurons per layer
    """
    if k == 1:
        return int(N/10.0)
    else:
        return int((sqrt(4*N*(k-1)) - (k+9))/2/(k-1))

##small test of get_num_neurons
# N = 10000
# for k in range(1, 5):
#     l = get_num_neurons(N, k)
#     print(l)
#     print((k-1)*l*l + (k+9)*l +2)


def neural_network(X, Y, X_test, Y_test, num_layers, activation):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_layers: number of layers in neural network
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    num_neurons = get_num_neurons(10000, num_layers)

    # Build the model
    if activation == "ReLU":
        activation_func = ReLUActivation
    if activation == "tanh":
        activation_func = TanhActivation

    model = Model(X.shape[1])
    for i in range(num_layers):
        model.addLayer(DenseLayer(num_neurons, activation_func()))
    model.addLayer(DenseLayer(Y.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())

    ##train the model and plot learning curve
    ##standardize the features first!!!
    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)
    hist = model.train(X_trans, Y, 2000, GDOptimizer(eta=0.0001))
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


def main():
    np.random.seed(0)
    n = 200
    num_layerss = [1, 2, 3, 4]
    mses = np.zeros((len(num_layerss), 2))

    # for s in range(replicates):
    sensor_loc = generate_sensors()
    X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
    X_test, Y_test = generate_data(sensor_loc, n=1000)
    for t, num_layers in enumerate(num_layerss):
        ### Neural Network:
        mse = neural_network(X, Y, X_test, Y_test, num_layers, "ReLU")
        mses[t, 0] = mse

        mse = neural_network(X, Y, X_test, Y_test, num_layers, "tanh")
        mses[t, 1] = mse

        print('Experiment with {} layers done...'.format(num_layers))

    ### Plot MSE for each model.
    plt.figure()
    activation_names = ['ReLU', 'Tanh']
    for a in range(2):
        plt.plot(num_layerss, mses[:, a], label=activation_names[a])

    plt.title('Error on validation data verses number of layers')
    plt.xlabel('Number of layers')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('num_layers.png')

if __name__ == '__main__':
    main()
