import numpy as np

from ..utils import get_n_classes


class DummyClassifier(object):
    """
    This method is a dummy method! It returns a random label for classification.
    """
    
    def __init__(self, arg1, arg2=0):
        """
        Initialization function. This get called when you create a new object of the class.
        The arguments can be used to correctly initialize it.

        Arguments:
            arg1: 
                Some dummy argument. As it has no default values, it needs to be given.
            arg2: int (default=0)
                Some dummy argument. As it has a default value, it is optional.
        """
        # `self` means "myself" as in "the current object of the class"
        # Below, we store the value of arg1 and arg2 in self
        self.arg1 = arg1
        self.arg2 = arg2
        # We can then access them in any function of this class by using `self.arg1`, for example.
    
    def random_predict(self, C, N):
        """
        Generate random classification predictions.

        This serves as an example function: this is how you can add your own
        function to your classes. See how it is called in predict(), and pay
        attention to the first argument "self" in the definition above.

        Arguments:
            C (int): number of classes
            N (int): number of predictions to make
        Returns:
            predictions (array): random predictions of shape (N,)
        """
        return np.random.randint(low=0, high=C, size=N)

    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        In the case of the DummyClassifier, this method will return 
        random labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # Here, we save some information during training by creating new members
        # with "self.variable = value". Then, they can be accessed in other function
        # using "self.variable".

        # D := dimension of data, C := number of classes
        self.D, self.C = training_data.shape[1], get_n_classes(training_labels)

        # Return the prediction on the training data.
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        In the case of the DummyClassifier, this method will return 
        random predicted labels.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = self.random_predict(self.C, test_data.shape[0])
        return pred_labels