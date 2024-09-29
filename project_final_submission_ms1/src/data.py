import os
import numpy as np
import cv2
import pickle as pkl
import random


def _load_images_from_directory(dir, idx):
    images = []
    for i in idx:
        image = np.transpose(cv2.imread(os.path.join(dir, '%05d.png')%i).astype(float), (2,0,1))
        images.append(image / 255.)
    return np.array(images)


def load_data(directory, skip=1, partition_rate=0.9):
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

    with open(os.path.join(directory,'annotation.pkl'), 'rb') as f:
        annos = pkl.load(f)
    labels = annos['labels']
    centers = annos['centers'].astype(float)

    # shuffled idx
    idx = annos['idx'][::skip]
    
    labels = labels[idx]
    centers = centers[idx]
    images = _load_images_from_directory(os.path.join(directory, 'images'), idx)

    partition = int(len(idx)*partition_rate)
    train_images = images[:partition]
    test_images = images[partition:]
    train_labels = labels[:partition]
    test_labels = labels[partition:]
    train_centers = centers[:partition]
    test_centers = centers[partition:]

    return train_images, test_images, train_labels, test_labels, train_centers, test_centers


if __name__ == "__main__":
    print('Testing data loading...')

    # change skip to downsample the dataset
    xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data('<PATH TO DIRECTORY>', skip=1)

    print(xtrain.shape, xtest.shape)
    print(ytrain.shape, ytest.shape)
    print(ctrain.shape, ctest.shape)

    print('Done!')