import numpy as np
import sys
import numpy.linalg as linalg
from utils import *

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        X = training_data
        N = training_data.shape[0]
        D = training_data.shape[1]

        # weights = linalg.inv(X.T@X + self.lmda * np.eye(D)) @ X.T @ training_labels
        weights = np.linalg.solve(X.T@X + self.lmda * np.eye(D), X.T @ training_labels)
        self.weights = weights
        pred_regression_targets = training_data @ weights

        return pred_regression_targets
    
    def fit_with_kernel(self, kernel_data, training_labels):
        K = kernel_data
        N = kernel_data.shape[0]

        # A = np.linalg.inv(K + self.lmda * np.eye(N)) @ training_labels
        A = np.linalg.solve(K + self.lmda * np.eye(N), training_labels)
        self.A = A 

        pred_regression_targets = K @ A

        return pred_regression_targets 


    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##

        pred_regression_targets = test_data @ self.weights

        return pred_regression_targets
    

    def predict_with_kernel(self, kernel_test_data):
         pred_regression_targets = kernel_test_data @ self.A 

         return pred_regression_targets 
