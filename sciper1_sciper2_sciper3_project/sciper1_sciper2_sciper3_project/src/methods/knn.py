import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    # Some helper functions
    """
    Euclidean distance

    Computes the euclidean distance of one target_vector with respect to all
    training_vectors

    Inputs:
        target_vector: (D, )
        training_vectors: (N x D)
    Outputs:
        the distance between target vector and all training vectors
    """
    def euclid_distance(self, target_vector):
        diff = self.training_vectors - target_vector.T
        return np.linalg.norm(diff, axis=1).reshape(-1)
    
    """
    Minkowski distance

    Computes the minkowski distance of one target vector with respect to all training vectors

    Inputs:
        target_vector: (D, )
        training_vectors: (N x D)
    Outputs:
        the minkowski distance between a target vector and all training vectors
    """
    def minkowski(self, target_vector, p=4):
        diff = self.training_vectors - target_vector.T
        sum_power_p_diff = np.sum(np.abs(diff) ** p, axis=1)
        distances = sum_power_p_diff ** (1 / p);
        return distances;

    """
    Simple weighting function: inverse of the distances 

    Inputs:
        distances: (N, )
    Outputs:
        inverse of the distances as weights
    """
    def simple_weights(self, distances):
        return 1 / distances

    """
    Decaing weighting function: inverse of the exp of the distances

    Inputs:
        distances: (N, )
    Ouputs: 
        inverse of the exp of the distances as weights
    """
    def decaying_weights(self, distances):
        return np.exp(-distances)
        
    """
    Find neighbours

    Finds the k smallest distances from a list of all distances

    Inputs:
        distances: (N, )
    Outputs:
        the indices of the K samples with smallest distance   
    """
    def find_k_smallest(self, distances):
        indices = np.argsort(distances, kind='quicksort');
        return indices[:self.k]
    
    """
    Majority vote

    Finds the most frequent class label in a list of class labels

    Inputs:
        labels: (k x D)
    Outputs:
        the most frequent class label in labels
    """
    def majority_vote(self, labels, weighting_function=None, w=False, distances=None):
        if(w == False):
            bins = np.bincount(labels)
            return np.argmax(bins)
        else:
            sample_weights = weighting_function(distances)
            bins = np.bincount(labels, weights=sample_weights)
            return np.argmax(bins)
        
    """
    Mean vote (for regression task of knn)

    Finds the mean value of the k nearest neighbours target values

    Inputs: 
        labels: (k x D)
    Outputs:
        the mean value of the values of the k nearest neighbours 
    """
    def mean_vote(self, target_values, weighting_function=None, w=False, distances=None):
        if(w == False):
            mean_val = np.mean(target_values, axis=0)
        else:
            mean_val = np.average(target_values, axis=0, weights=weighting_function(distances))
        return mean_val 
    
    """
    KNN one step target vector

    Performs one iteration of KNN algorithm for a given target vector

    Inputs:
        target_vector: (D, )
        training_vectors: (N, D)
        training_labels: (N, )

    Outputs:
        predicted label for the target vector
    """
    def knn_one_step_target_vector(self, target_vector, distance_func):
        distances = distance_func(target_vector)
        indices = self.find_k_smallest(distances)
        k_nearest_labels = self.training_labels[indices]

        if self.task_kind == "classification":
            predicted = self.majority_vote(k_nearest_labels)
        elif self.task_kind == "regression":
            predicted = self.mean_vote(k_nearest_labels)
        return predicted
    
    """
    KNN 

    Performs all iterations of KNN algorithm for a given set of target vectors

    Inputs:
        target_vectors: (M x D)
        training_vectors: (N x D)
        training_labels: (N, )
    Output:
        predicted labels for each target vector
    """
    def knn(self, target_vectors):
        return np.apply_along_axis(self.knn_one_step_target_vector, 1, target_vectors, self.euclid_distance)
        

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        self.training_vectors = training_data 
        self.training_labels = training_labels
        pred_labels = self.knn(training_data)
        
        # print(np.argwhere((pred_labels == self.training_labels) == 0))
        # print(training_labels[246])
        # print(pred_labels[246])

        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        test_labels = self.knn(test_data)
        return test_labels