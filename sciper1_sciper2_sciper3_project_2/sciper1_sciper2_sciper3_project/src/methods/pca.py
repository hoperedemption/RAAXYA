import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None
        ###
        self.allExvar = None
        self.U = None
        self.S = None
        self.eigenvalues = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        X = training_data
        self.mean = np.mean(X, axis=0)

        X_tilde = X - self.mean
        U, S, V = np.linalg.svd(X_tilde, full_matrices=False)
        self.U = U
        self.S = S

        self.eigenvalues = S**2 / (X.shape[0]-1)

        self.allExvar = np.cumsum(self.eigenvalues)/np.sum(self.eigenvalues)
        exvar = self.allExvar[self.d - 1]

        self.W = V[:self.d, :].T # dim W = D * d ----> 

        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        if data is None:
            data_reduced = self.U[:, :self.d] @ np.diag(self.S[:self.d])
        else:
            data_reduced = (data - self.mean) @ self.W

        return data_reduced