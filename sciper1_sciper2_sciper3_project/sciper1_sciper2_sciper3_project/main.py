import argparse

import numpy as np

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os
np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz',allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
        feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path,'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)



    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        N = xtrain.shape[0]

        percent_split = 0.8
        n_train = int(percent_split * N)

        random_index = np.random.permutation(N)

        train_index = random_index[:n_train]
        test_index = random_index[n_train:]

        xtest = xtrain[test_index]
        xtrain = xtrain[train_index]
        
        if(args.task == "center_locating"):
            ctest = ctrain[test_index]
            ctrain = ctrain[train_index]
        elif (args.task == "breed_identifying"):
            ytest = ytrain[test_index]
            ytrain = ytrain[train_index]
        
    
    ### WRITE YOUR CODE HERE to do any other data processing

    # normalize the data z-score
    mu = np.mean(xtrain, axis=0)
    sigma = np.std(xtrain, axis=0)

    xtrain = (xtrain - mu) / sigma
    xtest = (xtest - mu) / sigma

    # add a bias term to test and training data
    zeros_train = np.zeros((xtrain.shape[0], xtrain.shape[1] + 1))
    zeros_train[:, 0] = np.ones(xtrain.shape[0])
    zeros_train[:, 1:xtrain.shape[1] + 1] = xtrain

    xtrain = zeros_train 

    zeros_test = np.zeros((xtest.shape[0], xtest.shape[1] + 1))
    zeros_test[:, 0] = np.ones(xtest.shape[0])
    zeros_test[:, 1:xtest.shape[1] + 1] = xtest

    xtest = zeros_test 

    # Add the Kernel trick (experiment)

    # Gaussian Kernel: Radial basis functions

    def radial_basis_function(X, against, sigma):
        X_diff = X[:, np.newaxis] - against
        X_distances = np.linalg.norm(X_diff, axis=2) ** 2
        X_kernel_radial = np.exp(-X_distances / (2 * sigma ** 2))
        return X_kernel_radial

    sigma = 1.0

    xtrain_kernel = radial_basis_function(xtrain, xtrain, sigma)
    xtest_kernel = radial_basis_function(xtest, xtrain, sigma)
    
    ## 3. Initialize the method you want to use.

    k = 2
    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    elif args.method == "knn":
        
        if args.task == "center_locating":
            task = "regression"
            train = ctrain
        elif args.task == "breed_identifying":
            task = "classification"
            train = ytrain
        method_obj = KNN(1, task)

        if args.K == None :
            
            k_list = np.arange(1, 26)

            distance_functions = [method_obj.euclid_distance, method_obj.minkowski, method_obj.radial_basis_function]

            model_performance =  np.zeros((len(k_list), 3, 2))
            
            for i in k_list:
                for j, f in enumerate(distance_functions):
                    method_obj.distance_function = f
                    method_obj.K = i
                    method_obj.weighting_fucntion = None
                    model_performance[i-1][j][0] = method_obj.global_cross_validation(k, xtrain, train)
                    method_obj.weighting_fucntion = method_obj.decaying_weights
                    model_performance[i-1][j][1] = method_obj.global_cross_validation(k, xtrain, train)

            argmax = np.argmax(model_performance)
            best_K = k_list[argmax // 6]
            best_distance_function = distance_functions[(argmax % 6) // 2]
            best_weights_function = None if argmax % 2 == 0 else method_obj.decaying_weights
            method_obj.K = best_K
            method_obj.distance_function = best_distance_function
            method_obj.weighting_fucntion = best_weights_function

        else:
            method_obj = KNN(args.K, task)

    elif args.method == "logistic_regression":
        print("I AM A CRAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBb")
        method_obj = LogisticRegression(1, 1)
        print("---------------------------------------")
        print(args.max_iters)
        print("---------------------------------------")
        print(args.lr)
        print("---------------------------------------")
        # Should we consider the case where only one hyperparameter is defined
        if args.lr == None or args.max_iters == None: 
            #print("We are in crosss validation ------------------------------------------------------------")
            #print("AMMMMMMMMMMMMMMMMMOOOOOOOOOOOOOOOOOGGGGGGGGGGGGGGGGGGGGGGGGUUUUUUUUUUUUSSSSSSSSSSSS")
            D = xtrain.shape[1]
            #choosing_best_hyperparameters
            
            sigma_list = np.float_power(10, np.arange(-3, 1) / np.sqrt(D))
            index_list = np.arange(10) 
            lr_list = np.float_power(10,  (index_list - 7))
            max_iters_list = 500 * (index_list[::-1] + 1)
            
            model_performance = np.zeros((len(index_list), 1))

            for index in index_list:
                print("----------------------------------------------------------------------------")
                print(f"Iteration number in index_list is {index}")
                print("----------------------------------------------------------------------------")
                method_obj.lr = lr_list[index]
                method_obj.max_iters = max_iters_list[index]
                
                model_performance[index] = method_obj.global_cross_validation(k, xtrain, ytrain)
            
            best_index = np.argmax(model_performance)
            best_lr = lr_list[best_index]
            best_max_iters = max_iters_list[best_index]

            method_obj.lr = best_lr
            method_obj.max_iters = best_max_iters

            sigma_performance = np.zeros((len(sigma_list), 1))
            for i, sigma in enumerate(sigma_list):
                print("----------------------------------------------------------------------------")
                print(f"Iteration number in sigma_list is {i}")
                print("----------------------------------------------------------------------------")
                method_obj.sigma = sigma
                sigma_performance[i] = method_obj.global_cross_validation(k, xtrain, ytrain)
            
            best_sigma = sigma_list[np.argmax(sigma_performance)]
            method_obj.sigma = best_sigma
        else:
            method_obj = LogisticRegression(args.lr, args.max_iters)
        
    elif args.method == "linear_regression":
        method_obj = LinearRegression(1)
        if args.lmda != None:
            method_obj = LinearRegression(args.lmda)
        else :
            #choosing_best_hyperparameters
            lambda_list = np.float_power(10, np.arange(-10, 2))
            model_performance = np.zeros((len(lambda_list), 1))

            for i, lmbda in enumerate(lambda_list):
                print("----------------------------------------------------------------------------")
                print(f"Iteration number in lmbda_list is {i}")
                print("----------------------------------------------------------------------------")
                method_obj.lmda = lmbda
                model_performance[i] = method_obj.global_cross_validation(k, xtrain, ctrain)

            best_lambda = lambda_list[np.argmax(model_performance)]
            method_obj.lmda = best_lambda
        



    ## 4. Train and evaluate the method

    if args.task == "center_locating":
        # Fit parameters on training data
        preds_train = method_obj.fit(xtrain, ctrain)

        # Perform inference for training and test data
        train_pred = method_obj.predict(xtrain)
        preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        train_loss = mse_fn(train_pred, ctrain)
        loss = mse_fn(preds, ctest)

        print(f"\nTrain loss = {train_loss:.3f}% - Test loss = {loss:.3f}")

    elif args.task == "breed_identifying":

        # Fit (:=train) the method on the training data for classification task
        preds_train = method_obj.fit(xtrain, ytrain)

        # Predict on unseen data
        preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=None, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=None, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=None, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=None, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    # TODO Change back to original vaclues after asking question

    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
