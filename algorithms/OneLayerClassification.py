import numpy as np
# Dont have the planar_utils - Need to write sigmoid function

def layer_sizes(X, Y):
    """Returns the sizes of each layer.

    Method that will return the feature sizes of each layer to make code cleaner
    in later methods that use the dimensions.

    Args:
        X (Matrix): The inputs for the training data dimension Features X DataPoints
        Y (Vector): The vector of known Y values for the training data of size DataPoints

    Returns:
        n_x (int): The size of the input layer
        n_h (int): The size of the hidden layer
        n_y (int): The size of the output layer

    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    """Initializes the weights and biases to random numbers and zero respectively.

    Method that will initialize the weights in each layer to random values so that
    the network will not fail to break symmetry. The biases are initialized to zero

    Args:
        n_x (int): The size of the input layer
        n_h (int): The size of the hidden layer
        n_y (int): The size of the output layer

    Returns:
        W1 (Matrix): Matrix of dimension Features X DataPoints
        b1 (Vector): Vector of size number of Features
        W2 (Matrix): Matrix of dimension HiddenLayerCount X DataPoints
        b2 (Vector): Vector of size one

    """

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def sigmoid(Z):
    """Calculates the sigmoid of the input value

    Calculates the sigmoid of Z

    Args:
        Z (double): The Z value calcuated from the biases and inputs

    Returns:
        S (double): The computed sigmoid of the input Z

    """

    S = (1 / (1 + np.exp(-Z)))
    
    return S

def forward_propagation(X, parameters):
    """Calculates the values in forward propagation.

    Method that calculates the A and Z values for each layer and applies the tanh
    function to the input layers and the sigmoid function to the final output

    Args:
        X (Matrix): The inputs for the training data dimension Features X DataPoints
        paramaters: Tuple containing the weights and biases for each layer

    Returns:
        A2 (double): The computed Y hat value for the corresponding data point
        cache: Map containing the Z1, A1, Z2, A2 values needed to calculate the gradients

    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost(A2, Y, parameters):
    """Calculates the overall cost of the cycle.

    Method that calculates the total loss of each input


    Args:
        A2 (double): The computed Y hat value for the corresponding data point
        Y (Vector): The vector of known Y values for the training data of size DataPoints
        paramaters: Tuple containing the weights and biases for each layer

    Returns:
        cost (double): The cost of each cycle of propogation

    """

    m = Y.shape[1]

    logprobs = np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    cost = (-1 / m) * logprobs

    cost = np.squeeze(cost)

    return cost

def backward_propagation(parameters, cache, X, Y):
    """Calculates the values in back propagation

    Method that calculates the change in the wieghts and biases which will be used
    to adjust them

    Args:
        paramaters: Tuple containing the weights and biases for each layer
        cache: Map containing the Z1, A1, Z2, A2 values needed to calculate the gradients
        X (Matrix): The inputs for the training data dimension Features X DataPoints
        Y (Vector): The vector of known Y values for the training data of size DataPoints

    Returns:
        grads: The gradients for the weights and biases calculated in backward_propagation

    """

    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    """Calculates updates the wieghts and biases

    Method that updates the weights and biases with the calculated values in
    back propagation to fine tune the network

    Args:
        paramaters: Tuple containing the weights and biases for each layer
        grads: Map containing the dW1, db1, dW2, db2 values needed to adjust the weights and biases
        learning_rate (double): The rate at which the network will learn at

    Returns:
        paramaters: The adjusted weights and biases

    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """Neural Network model

    The model of the neural network that will be trained with the test data, and will
    classify inputs given to it

    Args:
        X (Matrix): The inputs for the training data dimension Features X DataPoints
        Y (Vector): The vector of known Y values for the training data of size DataPoints
        n_h (int): The number of nodes in the hidden layer

    Returns:
        paramaters: The final trained weights and biases to classify data

    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    """Method that classifies the data inputs

    Method that will return a vector of 0s or 1s that classifies the corresponding input data

    Args:
        paramaters: The final trained weights and biases to classify data
        X (Matrix): The inputs for the training data dimension Features X DataPoints

    Returns:
        predictions (Vector): The classifications for the input data

    """

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions
