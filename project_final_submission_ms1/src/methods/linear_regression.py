import numpy as np
import sys
import numpy.linalg as linalg
from ..utils import *

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda, task_kind="regression"):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.task_kind = task_kind

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

        if self.lmda == 0:
            weights = np.linalg.pinv(X) @ training_labels
        else:
            weights = np.linalg.solve(X.T@X + self.lmda * np.eye(D), X.T @ training_labels)
        self.weights = weights
        pred_regression_targets = training_data @ weights

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
    
    
    def fit_with_kernel(self, kernel_data, training_labels):
        K = kernel_data
        N = kernel_data.shape[0]

        if self.lmda == 0:
            A = np.linalg.pinv(K) @ training_labels
        else:
            A = np.linalg.solve(K + self.lmda * np.eye(N), training_labels)
        self.A = A 

        pred_regression_targets = K @ A

        return pred_regression_targets 


    def predict_with_kernel(self, kernel_test_data):
         pred_regression_targets = kernel_test_data @ self.A 

         return pred_regression_targets 


    def cross_validation_one_iteration(self, X_train, X_validate, Y_train, Y_validate):
        self.fit(X_train, Y_train)
        Y_predicted = self.predict(X_validate)

        loss = mse_fn(Y_predicted, Y_validate)
        return loss
        
    def global_cross_validation(self, k, training_data, training_labels):
        
        N = training_data.shape[0]
        D = training_data.shape[1]
        batch_size = N//k

        # voir plus tard pour le reste
        random_X_indices = np.random.permutation(N)
        all_loss = list()

        for i in range(k + 1):
            if i == k:
                cross_validate_indices = random_X_indices[batch_size*k:]
                if cross_validate_indices.shape[0] == 0: break 
            else:
                cross_validate_indices = random_X_indices[batch_size*i:batch_size*(i+1)]

            training_indices = np.setdiff1d(random_X_indices, cross_validate_indices)

            X_train = training_data[training_indices]
            Y_train = training_labels[training_indices]

            X_validate = training_data[cross_validate_indices]
            Y_validate = training_labels[cross_validate_indices]

            all_loss.append(self.cross_validation_one_iteration(X_train, X_validate, Y_train, Y_validate))

        mean_loss = np.mean(np.array(all_loss))
        return mean_loss
