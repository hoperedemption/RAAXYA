import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn, mse_fn, macrof1_fn


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
        linear = data @ W
        max_element = np.max(linear, axis=1)[:, np.newaxis]
        linear -= max_element
        aux = np.exp(linear) # this allows to avoid overflow

        sums = aux.sum(axis=1)[:, np.newaxis]
        result = np.where(sums == 0, 0, aux / sums) #if the sum is zero then necessarly the numerator 
        #is very low, so near zero. Otherwise we can just compute the usual divisinon        
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



    def __init__(self, lr=0.001, max_iters=1000):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.sigma = 1

    def cross_validation_one_iteration(self, batch_size, X_train, X_validate, Y_train, Y_validate):
        self.fit(X_train, Y_train)
        Y_predicted = self.predict(X_validate)

        loss = macrof1_fn(Y_predicted, Y_validate)
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

            all_loss.append(self.cross_validation_one_iteration(batch_size, X_train, X_validate, Y_train, Y_validate))

        mean_loss = np.mean(np.array(all_loss))
        return mean_loss


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


        self.weights = np.random.normal(0, self.sigma, (self.D, self.C))
        

        labels = label_to_onehot(training_labels, self.C)

        for i in range(self.max_iters):
            if i == 2:
                print("Here")
            gradient = self.gradient_logistic_multi(training_data, labels, self.weights)

            self.weights = self.weights - self.lr * gradient
                    
            predictions = self.logistic_regression_predict_multi(training_data, self.weights)

            accuracy = accuracy_fn(predictions, onehot_to_label(labels))

            if (accuracy == 100):
                break

        # print("final weights: ")
        # print(self.weights)
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

    