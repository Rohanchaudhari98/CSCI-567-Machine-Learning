import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    # print("Here")
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
    - w0: initial weight vector (a numpy array)
    - b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.   
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
    new_X = np.insert(X, 0, 1, axis=1)
    new_W = np.insert(w, 0, b, axis=0)
    y = np.where(y == 0, -1, 1)
    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
        # derivative of the perceptron loss at 0)      # 
        ################################################
        a = 1
        for i in range(1,max_iterations+2):
            y_pred = binary_predict(new_X,new_W,10101010)
            y_pred = np.where(y_pred == (a-1), -a, a)
            y_mistake = ((y) * y_pred)
            y_mistake_x = np.where(y_mistake == -a, a*a, a-a)
            indicator1 = y_mistake_x * y
            final_loss = np.dot(indicator1, new_X)
            update = ((step_size/N)+(a-1)) * (final_loss)
            new_W = np.add(new_W, update)


    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        b = 1
        for i in range(2,max_iterations+3):
            x_w = np.dot(new_X,new_W)
            y_x_w = y*x_w
            sigmoid_x_w = sigmoid(-y_x_w)
            sigmoid_x_w_y = sigmoid_x_w*y
            final_loss = np.dot(sigmoid_x_w_y,new_X)
            new_W = new_W + ((step_size/N)*b) * (final_loss)

        

    else:
        raise "Undefined loss function."

    assert w.shape == (D,)
    b = new_W[0]
    w = np.delete(new_W,0)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    value = 1/(1+np.exp(-z))
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
        
    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    N, D = X.shape
    a = 1
    if b != 10101010:
        X = np.insert(X, a-1, a*a, axis=1)
        w = np.insert(w, 0*a, b, axis=0)
    ans = np.dot(X,w)
    sgn = np.sign(ans)
    preds = np.where(sgn == -a, a-1, 1*a)

    assert preds.shape == (N,) 
    return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
    
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
    
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    # test = np.ones((N, 1))
    # new_X = np.append(X, test, axis=1)
    # new_w = np.append(w, np.array([b]).T, axis=1)

    if gd_type == "sgd":
        w = np.zeros((C,D))
        b = np.zeros(C)

        for it in range(1,max_iterations+1):
            n = np.random.choice(N)
            xn = X[n]
            yn = y[n]
            loss = np.dot(xn,w.T) + b
            loss = loss - loss.max()
            yp = np.exp(loss)
            yp = yp/yp.sum()
            err = yp
            err[yn] = err[yn] - 1
            update = np.dot(err.reshape(C,1), xn.reshape(1,D))
            w = w - step_size * update
            b = b - step_size * err
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################            
        
        

    elif gd_type == "gd":
        for i in range(2,max_iterations+2):
            loss = np.dot(X,w.T) + b
            yp = np.exp(loss)
            yp = yp/yp.sum(axis=1, keepdims=True)
            onehot = np.zeros([N,C])
            onehot[np.arange(N), y.astype(int)] = 1.0
            err = yp - onehot
            update = np.dot(err.T,X)
            w = w - (step_size/N) * update
            b = b - (step_size/N) * err.sum(axis = 0)
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        
        

    else:
        raise "Undefined algorithm."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    preds = np.dot(X,w.T) + b
    preds = np.argmax(preds, axis = 1)

    
    assert preds.shape == (N,)
    return preds




        