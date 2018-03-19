import numpy as np

def sigmoid(z):
    """Calculates the sigmoid of the input value

    Calculates the sigmoid of Z

    Args:
        Z (double): The Z value calcuated from the biases and inputs

    Returns:
        S (double): The computed sigmoid of the input Z

    """

    S = (1 / (1 + np.exp(-z)))
    return S

def initialize_with_zeros(dim):
    """Initializes the wieghts and biases

    Itinitalizes the wieght and bias to zero

    Args:
        dim (int): The number of input features

    Returns:
        w (double): The weight of the node
        b (double): The bias of the node

    """

    w = np.zeros((dim, 1))
    b = 0

    return w, b

def propagate(w, b, X, Y):
    """Computes forward and back propagation

    Calculates the gradients of the wieght and bias and the cost

    Args:
        w (double): The weight of the node
        b (double): The bias of the node
        X (Matrix): The inputs for the training data dimension Features X DataPoints
        Y (Vector): The vector of known Y values for the training data of size DataPoints

    Returns:
        grads: Map containing the change in the weight and change in the bias
        cost (double): The cost of the function

    """

    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / X.shape[1]) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = (1 / X.shape[1]) * np.dot(X, (A - Y).T)
    db = (1 / X.shape[1]) * np.sum(A - Y)

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """Trains the regression model

    Trains the regression model based on the specified paramters

    Args:
        w (double): The weight of the node
        b (double): The bias of the node
        X (Matrix): The inputs for the training data dimension Features X DataPoints
        Y (Vector): The vector of known Y values for the training data of size DataPoints
        num_iterations (int): The number of iterations you want to train your model over
        learning_rate (double): The learning rate of the model

    Returns:
        params: Map contining the adjusted weight and biases
        grads: Map containing the change in the weight and change in the bias
        cost (double): The cost of the function

    """

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X):
    """Classifies the data based on the trained model

    Classifies the data with 0 or 1 based on the trained weight and bias

    Args:
        w (double): The weight of the node
        b (double): The bias of the node
        X (Matrix): The inputs for the training data dimension Features X DataPoints

    Returns:
        Y_prediction (int): Classification of the X input based on the trained model

    """

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        print (i)
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """Trains the regression model

    Trains the regression model based on the specified paramters

    Args:
        X_train (double): The inputs for the training data dimension Features X DataPoints
        Y_train (double): The vector of known Y values for the training data of size DataPoints
        X_test (Matrix): The inputs for the test data dimension Features X DataPoints
        Y_test (Vector): The results of the test data (not actually needed for computations)

    Returns:
        d: Map containing all the values computed in the regression model

    """

    w, b = initialize_with_zeros(X_test.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d
