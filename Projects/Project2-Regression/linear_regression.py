import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################


###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    a = 1
    if (len(X[0])*a) == (len(w)+(a-1)):
        prod = np.dot(X,w)
        diff = np.subtract(prod,y)
        sq = np.square(diff)
        err = np.mean(sq, dtype=np.float64)
    else:
        prod = np.dot(X.transpose(),w)
        diff = np.subtract(prod,y)
        sq = np.square(diff)
        err = np.mean(sq, dtype=np.float64)
    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  #####################################################		
  xt = X.transpose()
  xt_x = np.dot(xt,X)
  xt_x_inv = np.linalg.inv(xt_x)
  xt_y = np.dot(xt,y)
  w = np.dot(xt_x_inv,xt_y)
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################		
    xt = X.transpose()
    a = 1.0
    xt_x = np.dot(xt,X)
    xt_y = np.dot(xt,y)
    lambda_iden = float(a)*lambd * np.identity(len(xt_x))
    sums = np.add(xt_x,lambda_iden)
    inv = np.linalg.inv(sums)
    w = np.dot(inv,xt_y)
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################		
    bestlambda = None
    l1 = 1
    minmse = float("inf")
    for i in range(-15,2):
        lam = ((2 ** (i+1))*(l1-1)) + (2**(i+1))
        rlr = regularized_linear_regression(Xtrain,ytrain,lam)
        mserror = mean_square_error(rlr, Xval,yval)
        if mserror < minmse:
            minmse = mserror
            bestlambda = lam
    return bestlambda
    

###### Part 1.5 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################		
    X = np.array(X)
    X_temp = np.array(X)
    for i in range(3,p+2):
        ele_wise_pow = np.power(X_temp,i-1)
        X = np.insert(X,[len(X[0])],ele_wise_pow, axis = 1)
    return X


    


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

