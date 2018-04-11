import numpy as np
import matplotlib.pyplot as plt

from starter import *
from plot1 import *
from plot3 import get_num_neurons


def neural_network(X, Y, Xs_test, Ys_test, num_layers, eta=0.0001, epochs=2000):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_layers: number of layers in neural network
        eta: learning rate
        epochs: how many epochs to run
    Output:
        mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    num_neurons = get_num_neurons(10000, num_layers)

    # Build the model
    model = Model(X.shape[1])
    for i in range(num_layers):
        model.addLayer(DenseLayer(num_neurons, ReLUActivation()))
    model.addLayer(DenseLayer(Y.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())

    ##train the model and plot learning curve
    ##standardize the features first!!!
    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)
    hist = model.train(X_trans, Y, epochs, GDOptimizer(eta=eta))
    # plt.plot(hist)
    # plt.title('Learning curve')
    # plt.show()

    ##evaluate the model
    mses = []
    for X_test, Y_test in zip(Xs_test, Ys_test):
        X_test_trans = scaler.transform(X_test)
        Y_pred = model.predict(X_test_trans)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses

def main():
    #############################################################################
    #######################PLOT PART 1###########################################
    #############################################################################
    np.random.seed(0)

    ns = np.arange(100, 1000, 200)
    replicates = 1
    num_methods = 6
    num_sets = 3
    mses = np.zeros((len(ns), replicates, num_methods, num_sets))

    def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
        return generate_dataset(
            sensor_loc,
            num_sensors=k,
            spatial_dim=d,
            num_data=n,
            original_dist=original_dist,
            noise=noise)

    for s in range(replicates):
        sensor_loc = generate_sensors()
        X_test, Y_test = generate_data(sensor_loc, n=1000)
        X_test2, Y_test2 = generate_data(
            sensor_loc, n=1000, original_dist=False)
        for t, n in enumerate(ns):
            X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
            Xs_test, Ys_test = [X, X_test, X_test2], [Y, Y_test, Y_test2]
            # ### Linear regression:
            # mse = linear_regression(X, Y, Xs_test, Ys_test)
            # mses[t, s, 0] = mse

            # ### Second-order Polynomial regression:
            # mse = poly_regression_second(X, Y, Xs_test, Ys_test)
            # mses[t, s, 1] = mse

            # ### 3rd-order Polynomial regression:
            # mse = poly_regression_cubic(X, Y, Xs_test, Ys_test)
            # mses[t, s, 2] = mse

            ### Neural Network:
            mse = neural_network(X, Y, Xs_test, Ys_test, num_layers=4, eta=0.0001, epochs=2000)
            mses[t, s, 3] = mse
            print(mse)

            ### Generative model:
            # mse = generative_model(X, Y, Xs_test, Ys_test)
            # mses[t, s, 4] = mse

            # ### Oracle model:
            # mse = oracle_model(X, Y, Xs_test, Ys_test, sensor_loc)
            # mses[t, s, 5] = mse

            print('{}th Experiment with {} samples done...'.format(s, n))

    ### Plot MSE for each model.
    plt.figure()
    regressors = [
        'Linear Regression', '2nd-order Polynomial Regression',
        '3rd-order Polynomial Regression', 'Neural Network',
        'Generative Model', 'Oracle Model'
    ]
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 0], axis=1), label=regressors[a])

    plt.title('Error on training data for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('train_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 1], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from the same distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_same_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 2], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from a different distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_different_mse.png')
    plt.show()


if __name__ == '__main__':
    main()
