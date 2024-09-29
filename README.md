# RAAXYA

# Machine Learning Methods for Dog Classification and Fashion MNIST Image Classification


## Disclaimer

This project is intended for educational purposes only. The models and code provided in this repository are not guaranteed to be free from errors and are not optimized for production environments.

While every effort has been made to ensure the accuracy and functionality of the code, the authors make no warranties, express or implied, about the fitness for any particular purpose. 

This code is not meant to be copied or reused for future editions of any course. It represents solely our work, and we do not agree to its reproduction or use for any course or academic work without explicit permission.

## Project Overview

This project was part of the course 'Introduction to Machine Learning' (Spring 2024 edition) at EPFL.It is joint effort by the three contributors to the repository.

Our models achieved an average F1-score of 0.941 and an ajusted accuracy score of 93.5% on AIcrowd, competition hosted by the course. We ranked 4th among all teams.

The project consists of two main tasks:

1. **Dog Classification Task (Milestone One)**: 
   We implemented basic machine learning algorithms (logistic regression, linear regression and KNN) for classifying dogs and finding the center point of images based on features extracted by a CNN. The dogs images are taken from the Stanford Dogs dataset.
   
2. **Fashion MNIST Image Classification (Milestone Two)**: 
   We explored more advanced deep learning models for classifying Fashion MNIST images. This included the use of Convolutional Neural Networks (CNN), Neural Architecture Search (NAS), and Vision Transformers (ViT) optimized for performance and parallelism.

A more detailed overview can be found in our two reports under project_final_submission_ms{1, 2}/src/report.pdf.


## Table of Contents
- [Task 1: Dog Classification](#task-1-dog-classification)
  - [K-Nearest Neighbors (KNN)](#knn)
  - [Linear Regression](#linear-regression)
  - [Logistic Regression](#logistic-regression)
- [Task 2: Fashion MNIST Image Classification](#task-2-fashion-mnist-image-classification)
  - [CNN (MobileNet-inspired)](#cnn-mobilenet-inspired)
  - [Neural Architecture Search for MLP](#neural-architecture-search-for-mlp)
  - [Vision Transformer (ViT)](#vision-transformer-vit)
  - 
 ## Task 1: Dog Classification
In the first milestone of the project, we implemented basic machine learning algorithms to classify dogs based on their features. 

### KNN
K-Nearest Neighbors (KNN) is a simple, instance-based learning method. For this task, we used KNN to classify dog breeds. We used different distance metrics, kernel trick with RBF and K-fold cross validation to select the best model possible.

- **Algorithm**: KNN
- **Implementation Details**: 
  - KNN searches for the `k` nearest neighbors to a query point and predicts the majority class.
  - We used different distance metrics that showed different results. 
  - A 3d hyperparameter search was used
 
### Linear Regression

We used linear regression to find the center point of dog images.

- **Algorithm**: Linear Regression
- **Implementation Details**: 
  - We fitted a linear model using ordinary least squares (OLS) to map input features to a continuous target variable. 
  - We used the moore penrose pseudo inverse (as computed by SVD decomposition)

### Logistic Regression

Logistic regression was implemented to predict the dog breed based on categorical outputs, allowing us to handle binary classification tasks.

- **Algorithm**: Logistic Regression
- **Implementation Details**: 
  - We used gradient descent for optimization and employed techniques to handle multicollinearity and feature scaling.
  - Numerical issues and overflow were adressed (see the report)


## Task 2: Fashion MNIST Image Classification

In this task, we explored more advanced techniques for image classification using the Fashion MNIST dataset.

### CNN (MobileNet-inspired)

We implemented a Convolutional Neural Network (CNN) architecture inspired by Google's MobileNet, optimized for lightweight performance while maintaining high accuracy.

- **Model**: MobileNet-inspired CNN
- **Implementation Details**: 
  - Depthwise separable convolutions were used to reduce the computational complexity.
  - Batch normalization, inverted residual blocks and SiLU activations were employed for faster convergence. Residual connections allowed for better gradient flow.
  - We tried adding inception modules for better performance

### Neural Architecture Search for MLP

We implemented a simple Neural Architecture Search (NAS) to find the best Multi-Layer Perceptron (MLP) architecture for classifying Fashion MNIST images.

- **Model**: MLP (searched architecture)
- **Implementation Details**: 
  - The NAS searched through hyperparameters such as number of layers, hidden units, and activation functions.
  - Techniques like random search were used to optimize the architecture.

### Vision Transformer (ViT)

We implemented a Vision Transformer (ViT) optimized for better parallelism and vectorization, providing efficient computation on the Fashion MNIST dataset.

- **Model**: Vision Transformer (ViT)
- **Implementation Details**: 
  - Images were split into patches (all vectorised, which allowed for parallelism), and the transformer model attended to these patches using multi-head self-attention.
  - The model was vectorized and optimized for GPU parallelism to enhance performance.
  - We fine-tuned hyperparameters to improve training time and classification accuracy.

