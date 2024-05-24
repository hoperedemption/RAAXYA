import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

import scipy.ndimage
from scipy.ndimage import interpolation

def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix
def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]]) # <-- 
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)
def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = scipy.ndimage.convolve(image, kernel)
    sharpened_image = np.clip(sharpened_image, 0, 255)
    return sharpened_image
def sharpen_turbo(image):
    kernel = -np.array([[1, 4, 6, 4, 1], [4, 16, 24, 26, 4],
                        [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], 
                        [1, 4, 6, 4, 1]])/256
    sharpened_image = scipy.ndimage.convolve(image, kernel)
    sharpened_image = np.clip(sharpened_image, 0, 255)
    return sharpened_image
def cdf_to_uniform(image):
    image_histogram, image_bins = np.histogram(image, 256, density=True)
    cumsum = np.cumsum(image_histogram)
    cdf = image_bins[-1] * cumsum / cumsum[-1]

    image_edited = np.interp(image, image_bins[:-1], cdf)
    return image_edited
    

    

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    # convert the images arrays to numpy float format
    xtrain = xtrain.astype('float64')
    xtest = xtest.astype('float64')
    ytrain = ytrain.astype('int64')
    ytest = None

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        N = xtrain.shape[0]

        percent_split = 0.8
        n_train = int(percent_split * N)

        random_index = np.random.permutation(N)

        train_index = random_index[:n_train]
        test_index = random_index[n_train:]

        xtest = xtrain[test_index]
        xtrain = xtrain[train_index]
        
        ytest = ytrain[test_index]
        ytrain = ytrain[train_index]

    new_xtrain = np.zeros(xtrain.shape)
    new_ytrain = np.zeros(ytrain.shape)
    new_xtrain[:] = xtrain 
    new_ytrain[:] = ytrain

    for i in range(new_xtrain.shape[0]):
        #sharpen_turbo
        #xtrain[i] = deskew(xtrain[i].reshape(28, 28)).reshape(28 * 28)
        #xtrain[i] = sharpen(xtrain[i].reshape(28, 28)).reshape(28 * 28)

        new_xtrain[i] = cdf_to_uniform(new_xtrain[i])
        new_xtrain[i] = sharpen_turbo(new_xtrain[i].reshape(28, 28)).reshape(28 * 28)

        # u, s, v = np.linalg.svd(xtrain[i].reshape(28, 28), full_matrices=False)
        # cumsum = np.cumsum(s)
        # ratio = cumsum / cumsum[-1]
        # threshold = 0.85
        # ind = np.argmax(ratio > threshold)
        # xtrain[i] = (u[:, :ind] @ np.diag(s)[:ind, :ind] @ v[:ind, :]).reshape(28 * 28)
    
    for i in range(xtest.shape[0]):
        #xtest[i] = deskew(xtest[i].reshape(28, 28)).reshape(28 * 28)
        #xtest[i] = sharpen(xtest[i].reshape(28, 28)).reshape(28 * 28)
        xtest[i] = cdf_to_uniform(xtest[i])
        xtest[i] = sharpen_turbo(xtest[i].reshape(28, 28)).reshape(28 * 28)

        # u, s, v = np.linalg.svd(xtest[i].reshape(28, 28), full_matrices=False)
        # cumsum = np.cumsum(s)
        # ratio = cumsum / cumsum[-1]
        # threshold = 0.85
        # ind = np.argmax(ratio > threshold)
        # xtest[i] = (u[:, :ind] @ np.diag(s)[:ind, :ind] @ v[:ind, :]).reshape(28 * 28)

    # normalise the pixel ranges
    xtrain /= 255
    xtest /= 255

    labels_printed = np.zeros(10)
    for i in range(xtrain.shape[0]):
        if(np.sum(labels_printed) == 10):
            break
        label = ytrain[i]
        labels_printed[label] = 1

        image = xtrain[i]
        imgplot = plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.savefig(f'random_sample_{i}_label_{label}_original.png', bbox_inches='tight') 

        transformed_image = new_xtrain[i]
        imgplot = plt.imshow(transformed_image.reshape(28, 28), cmap='gray')
        plt.savefig(f'random_sample_{i}_label_{label}_tranformed_image.png', bbox_inches='tight') 

    # center the images on the screen

    random_image = xtrain[np.random.randint(0, xtrain.shape[0])]
    imgplot = plt.imshow(random_image.reshape(28, 28), cmap='gray')
    plt.savefig('random_sample.png', bbox_inches='tight') 

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    ### WRITE YOUR CODE HERE to do any other data processing

    # D = xtrain.shape[1]
    # pca_obj = PCA(D)
    # pca_obj.find_principal_components(xtrain)
    # xtrain_normalised = pca_obj.reduce_dimension(xtrain) / np.sqrt(pca_obj.eigenvalues)
    # xtest_normalised = pca_obj.reduce_dimension(xtest) / np.sqrt(pca_obj.eigenvalues)

    # random_image_normalised = xtrain_normalised[np.random.randint(0, xtrain_normalised.shape[0])]
    # imgplot_normalised = plt.imshow(random_image_normalised.reshape(28, 28), cmap='gray')
    # plt.savefig('random_sample_normalised.png', bbox_inches='tight') 

    # Dimensionality reduction (MS2)
    if args.use_pca: #only for MLP
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        pca_obj.d = args.pca_d
        x_train_reduced = pca_obj.reduce_dimension(xtrain)
        x_test_reduced = pca_obj.reduce_dimension(xtest)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data

    # xtrain = xtrain_normalised
    # xtest = xtest_normalised
    
    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)

    if args.nn_type == "mlp":
        if args.use_pca:
            xtrain = x_train_reduced
            xtest = x_test_reduced 
            model = MLP(pca_obj.d, n_classes)
        else:
            model = MLP(28*28, n_classes)
    elif args.nn_type == "cnn":
        # the input to the CNN model are images of shape (28, 28)
        # and not vectors of shape (784, ) thus we need to reshape the xtrain and xtest matrices
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)

        model = CNN(1, n_classes) ### WRITE YOUR CODE HERE
    elif args.nn_type == "transformer":
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        
        n_patches = 7
        n_blocks = 9 # 2
        hidden_d = 14 # 8
        n_heads = 2
        out_d = n_classes
        model = MyViT(chw = (1, 28, 28), n_patches=n_patches, n_blocks=n_blocks,
              hidden_d=hidden_d, n_heads=n_heads, out_d=out_d)

    summary(model)

    # Trainer object
    if not args.test:
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, validation_set=xtest, validation_labels=ytest, validation=True)
    else:
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, xtest)
    macrof1 = macrof1_fn(preds, xtest)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)


    """  # function to compute the centroid of an image
    def compute_covariance_matrix(image):
        total_sum = np.sum(image)
        if total_sum == 0: # empty image do nothing
            return None
        else: # otherwise need to go trhough the formula
            x_index, y_index = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
            x_centroid, y_centroid = np.sum(x_index * image) / total_sum, np.sum(y_index * image) / total_sum
            var_x = np.sum((x_index - x_centroid) ** 2 * image) / total_sum 
            var_y = np.sum((y_index - y_centroid) ** 2 * image) / total_sum 
            cov_xy = np.sum((x_index - x_centroid) * (y_index - y_centroid) * image) / total_sum 
            mean_vector = np.array([x_centroid, y_centroid])
            covariance_matrix = np.array([[var_x, cov_xy], [cov_xy, var_y]])
            return mean_vector, covariance_matrix
    
    def apply_affine_transform(image):
        mean, cov = compute_covariance_matrix(image)
        alpha = cov[0, 1] / cov[0, 0]
        affine = np.array([[1, 0], [alpha, 1]])
        image_center = np.array([image.shape[0] // 2, image.shape[1] // 2])
        offset = mean - affine @ image_center
        affine = np.hstack((affine, offset[:, None]))

        modified_affine = torch.from_numpy(affine).reshape([1, 2, 3])
        modified_image = torch.from_numpy(image).reshape([1, 1, 28, 28])
        grid = F.affine_grid(modified_affine, modified_image.size())
        modified_image = F.grid_sample(modified_image, grid)
        return modified_image.cpu().detach().numpy()
    
    # function to shift an the content of an image
    def shift_image(image, delta_x, delta_y):
        image = np.roll(image, delta_x, axis=0) # circular shift
        image = np.roll(image, delta_y, axis=1) # circular shift

        # can zero fill (to determine)
        if delta_x > 0:
            image[:delta_x, :] = 0
        elif delta_x < 0:
            image[delta_x:, :] = 0
        
        if delta_y > 0:
            image[:, :delta_y] = 0
        elif delta_y < 0:
            image[:, delta_y:] = 0

        return image 

    def recenter_image(image):
        x_center, y_center = image.shape[0] // 2, image.shape[1] // 2
        x_centroid, y_centroid = compute_centroid(image)
        delta_x, delta_y = int(x_center - x_centroid), int(y_center - y_centroid)
        image = shift_image(image, delta_x, delta_y)
        return image
    
    
    for i in range(xtrain.shape[0]):
        xtrain[i] = apply_affine_transform(xtrain[i].reshape(28, 28)).reshape(784, )
     """