import numpy as np


def load_data(data_path):
    """
    Return the dataset as numpy arrays.
    
    Arguments:
        directory (str): path to the dataset directory
    Returns:
        train_images (array): images of the train set, of shape (N,H,W)
        test_images (array): images of the test set, of shape (N',H,W)
        train_labels (array): labels of the train set, of shape (N,)
        test_labels (array): labels of the test set, of shape (N',)
        train_centers (array): centers of the dog of the train set, of shape (N,2)
        test_centers (array): centers of the dog of the test set, of shape (N',2)
    """
    xtrain = np.load(data_path + '/train_data.npy', allow_pickle=True)
    ytrain = np.load(data_path + '/train_label.npy', allow_pickle=True)
    xtest = np.load(data_path + '/test_data.npy', allow_pickle=True)

    return xtrain, xtest, ytrain

