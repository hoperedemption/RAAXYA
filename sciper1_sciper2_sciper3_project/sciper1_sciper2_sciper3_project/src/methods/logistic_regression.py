import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn


class LogisticRegression(object):


    """
    Logistic regression classifier.
    """

    def loss_logistic_multi(self, data, labels, w):
        """ 
        Loss function for multi class logistic regression, i.e., multi-class entropy.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value 
        """
        ### WRITE YOUR CODE HERE 
        
        predictions = self.f_softmax(data, w)
        result = labels * np.log(predictions)
        
        return -np.sum(result)


    def gradient_logistic_multi(self, data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        ### WRITE YOUR CODE HERE 
        predictions = self.f_softmax(data, W)
        
        
        return data.T @ (predictions - labels)

    def f_softmax(self, data, W):
        """
        Softmax function for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and 
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        ### WRITE YOUR CODE HERE 
        # Hint: try to decompose the above formula in different steps to avoid recomputing the same things.
        aux = np.exp(data @ W)
        sums = aux.sum(axis=1)
        result = aux / sums[:, np.newaxis]
        
        return result
    
    
    def logistic_regression_predict_multi(self, data, W):
        """
        Prediction the label of data for multi-class logistic regression.
        
        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """
        ### WRITE YOUR CODE HERE
        predictions = self.f_softmax(data, W)
    
        return np.argmax(predictions, axis=1)



    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.N, self.D, self.C = training_data.shape[0], training_data.shape[1], get_n_classes(training_labels)
        self.weights = np.random.normal(0, 0.1, (self.D, self.C))

        for i in range(self.max_iters):
            gradient = self.gradient_logistic_multi(training_data, training_labels, self.weights)
            self.weights = self.weights - self.lr * gradient   
                    
            predictions = self.logistic_regression_predict_multi(training_data, self.weights)
            if accuracy_fn(predictions, onehot_to_label(training_labels)) == 100:
                break

        # return pred_labels
        return predictions

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##


        predictions = self.logistic_regression_predict_multi(test_data, self.weights)
        # return pred_labels
        return predictions