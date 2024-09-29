import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchvision
from typing import List
from typing import Tuple
from src.methods.deep_network import InvertedResidual, ConvolutionLayer

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


#Legacy code : the first data augmentation + processing we did

# import scipy.ndimage
# from scipy.ndimage import interpolation

# def moments(image):
#     c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
#     totalImage = np.sum(image) #sum of pixels
#     m0 = np.sum(c0*image)/totalImage #mu_x
#     m1 = np.sum(c1*image)/totalImage #mu_y
#     m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
#     m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
#     m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
#     mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
#     covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
#     return mu_vector, covariance_matrix
# def deskew(image):
#     c,v = moments(image)
#     alpha = v[0,1]/v[0,0]
#     affine = np.array([[1,0],[alpha,1]]) # <-- 
#     ocenter = np.array(image.shape)/2.0
#     offset = c-np.dot(affine,ocenter)
#     return interpolation.affine_transform(image,affine,offset=offset)
# def sharpen(image):
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     sharpened_image = scipy.ndimage.convolve(image, kernel)
#     sharpened_image = np.clip(sharpened_image, 0, 255)
#     return sharpened_image
# def sharpen_turbo(image):
#     kernel = -np.array([[1, 4, 6, 4, 1], [4, 16, 24, 26, 4],
#                         [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], 
#                         [1, 4, 6, 4, 1]])/256
#     sharpened_image = scipy.ndimage.convolve(image, kernel)
#     sharpened_image = np.clip(sharpened_image, 0, 255)
#     return sharpened_image
# def cdf_to_uniform(image):
#     image_histogram, image_bins = np.histogram(image, 256, density=True)
#     cumsum = np.cumsum(image_histogram)
#     cdf = image_bins[-1] * cumsum / cumsum[-1]

#     image_edited = np.interp(image, image_bins[:-1], cdf)
#     return image_edited
# def preprocess(xtrain, xtest):
#     for i in range(xtrain.shape[0]):
#         xtrain[i] = cdf_to_uniform(xtrain[i])
#     for i in range(xtest.shape[0]): 
#         xtest[i] = cdf_to_uniform(xtest[i])
#     return xtrain, xtest
# def augment(xtrain, ytrain):
#     num_indices = np.random.randint(xtrain.shape[0] // 4, xtrain.shape[0] // 2)
#     selected_indices = np.random.choice(xtrain.shape[0], num_indices, replace=False)
#     new_xtrain = np.zeros((xtrain.shape[0] +  2 * num_indices, xtrain.shape[1]))
#     new_ytrain = np.zeros(ytrain.shape[0] + 2 * num_indices)
#     new_xtrain[:xtrain.shape[0], :] = xtrain 
#     new_ytrain[:ytrain.shape[0]] = ytrain
    

    # for i in range(0, 2 * num_indices, 2):
    #     index = selected_indices[i//2]

    #     horizontal_flip = (xtrain[index].reshape(28, 28))[::, ::-1].reshape(28 * 28)

    #     reshaped_image = xtrain[index].reshape(28, 28)
    #     rand_bis = np.random.uniform(0, 1)
    #     if rand_bis <= 0.5:
    #         reshaped_image[:int(rand_bis*28), :] = 0
    #     elif rand_bis > 0.5:
    #         reshaped_image[int(rand_bis*28):, :] = 0
    #     reshaped_image = reshaped_image.reshape(28 * 28)

    #     new_xtrain[xtrain.shape[0] + i] = horizontal_flip
    #     new_xtrain[xtrain.shape[0] + (i + 1)] = reshaped_image
    #     new_ytrain[ytrain.shape[0] + i] = ytrain[index]
    #     new_ytrain[ytrain.shape[0] + (i + 1)] = ytrain[index]

    # xtrain = new_xtrain.astype('float64')
    # ytrain = new_ytrain.astype('int64')
    # return xtrain, ytrain

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

        if args.k_fold:
            k = 5
            N_test = xtest.shape[0]
            D_test = xtest.shape[1]
            batch_size = N_test//k

            random_X_indices = np.random.permutation(N_test)

    # normalise the pixel ranges
    xtrain /= 255
    xtest /= 255

    random_image = xtrain[np.random.randint(0, xtrain.shape[0])]
    imgplot = plt.imshow(random_image.reshape(28, 28), cmap='gray')
    plt.savefig('random_sample.png', bbox_inches='tight') 
    plt.cla()
    
    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    ### WRITE YOUR CODE HERE to do any other data processing

    # Dimensionality reduction (MS2)
    if args.use_pca: #only for MLP
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        pca_obj.d = args.pca_d
        pca_obj.find_principal_components(xtrain)
        x_train_reduced = pca_obj.reduce_dimension(xtrain)
        x_test_reduced = pca_obj.reduce_dimension(xtest)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data
    
    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)

    if args.nn_type == "mlp":
        if args.use_pca:
            xtrain = x_train_reduced
            xtest = x_test_reduced 
            model = MLP(pca_obj.d, n_classes, use_pca=True)
        else:
            model = MLP(28*28, n_classes)

    elif args.nn_type == "cnn":
        # the input to the CNN model are images of shape (28, 28)
        # and not vectors of shape (784, ) thus we need to reshape the xtrain and xtest matrices
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
    
        model = CNN(1, n_classes) ### WRITE YOUR CODE HERE


    elif args.nn_type == "transformer": # try on batch size 256 
        # 
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        
        n_patches = 4 # 7
        n_blocks = 8 #9 # 2
        hidden_d = 44 # 8
        n_heads = 4 # 2
        out_d = n_classes
        model = MyViT(chw = (1, 28, 28), n_patches=n_patches, n_blocks=n_blocks,
              hidden_d=hidden_d, n_heads=n_heads, out_d=out_d)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print("--------------> Model has been transfered to CUDA.")
    else: 
        device = torch.device("cpu")
        print("--------------> CUDA is not available. Model is on CPU.")

    print(f" -----> Model is on device: {next(model.parameters()).device}")

    summary(model)

    if args.load_model:
        PATH = "best_model.pth.tar"
        model.load_state_dict(torch.load(PATH))
        model.eval()

    # Trainer object
    if not args.test:
        #  N, D = 10, 3
        # training_data = np.random.rand(N, D)
        # training_labels = np.random.randint(0, D, N)
        
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, validation_set=xtest, validation_labels=ytest, validation=True, i=96)
    else:
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, i=96)

    if not args.load_model:
        ## 4. Train and evaluate the method

        # Fit (:=train) the method on the training data
        preds_train = method_obj.fit(xtrain, ytrain)

        # pred_labels = method_obj.fit(training_data, training_labels)

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    if not args.test: # since no testing data can only do testing on validation set
        ## As there are no test dataset labels, check your model accuracy on validation dataset.
        # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
         # Predict on unseen data

        if args.k_fold:
            print("-------------------------- K fold on validation set --------------------------")
            print(f"-------------------------- K value is {k} --------------------------")

            results_acc = list()
            results_f1 = list()
            for i in range(k + 1):
                if i == k:
                    cross_validate_indices = random_X_indices[batch_size*k:]
                    if cross_validate_indices.shape[0] == 0: break
                else:
                    cross_validate_indices = random_X_indices[batch_size * i:batch_size * (i + 1)]

                x_validate = xtest[cross_validate_indices]
                y_validate = ytest[cross_validate_indices]

                preds_validate = method_obj.predict(x_validate)
                acc_validate = accuracy_fn(preds_validate, y_validate)
                macrof1_validate = macrof1_fn(preds_validate, y_validate)
                print(f"K fold Validation set:  accuracy = {acc_validate:.3f}% - F1-score = {macrof1_validate:.6f}")

                results_acc.append(acc_validate)
                results_f1.append(macrof1_validate)
            
            acc_mean = np.mean(np.array(results_acc))
            f1_mean = np.mean(np.array(results_f1))

            print(f"-------------------------- K fold results  --------------------------")
            print(f"-------------------------- Mean accuracy: {acc_mean}  --------------------------")
            print(f"-------------------------- Mean f1 score: {f1_mean}  --------------------------")


        print("-------------------------- Predictions on validation set --------------------------")
        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        print("-------------------------- Predictions on validation set --------------------------")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--k_fold", action="store_true")
    parser.add_argument("--load_model", action="store_true")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)