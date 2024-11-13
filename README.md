Image Classification with Convolutional Neural Networks (CNNs)

This project focuses on building, training, and validating convolutional neural networks to classify images from the CIFAR10 dataset. CIFAR10 consists of 60,000 RGB images (32x32 px) spanning 10 classes, including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The aim of this assignment is to implement a series of tasks to progressively improve the model’s performance.

CIFAR10 is a dataset that contains (small) RGB images of 32x32 px of ten different classes:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck
More details can be found at this link: https://www.cs.toronto.edu/~kriz/cifar.html

Based on an overview found on this website: https://paperswithcode.com/sota/image-classification-on-cifar-10, the current state-of-the-art result on this dataset has reached an accuracy of 99.7%.

Objectives
The following tasks guide you through model development with CNNs in Keras, including architectural improvements and training strategies.

Tasks
Split Data and Build Convolutional Networks

Define the training and validation sets.
Implement the initial CNN architecture.
Train Convolutional Networks

Train the initial model on CIFAR10, applying it to the test set.
Reuse functions from previous assignments for training.
Add Dropout Layers

Integrate dropout layers to reduce overfitting.
Train the modified network and compare results.
Add Batch Normalization

Incorporate batch normalization to stabilize training.
Evaluate and compare with previous architectures.
Explore Different Initialization Strategies

Test different weight initialization methods (e.g., Xavier, He) and report their impact.
Experiment with Different Activation Functions

Try activation functions other than ReLU (e.g., Leaky ReLU, ELU).
Assess performance differences.
Add L2 Regularization

Modify the loss function to include L2 regularization.
Record the effects on model performance.
Implement Data Augmentation

Apply data augmentation techniques (e.g., rotation, flipping) to expand the training dataset.
Test on-the-fly augmentation to improve generalization.
Experiment with Different Architectures

Enhance the architecture by adding more layers or filters.
Combine techniques from previous tasks to optimize accuracy.
Monitor Training Progress

Visualize training statistics, activation patterns, and filter evolution.
Implement tools to observe how the network refines its filters over time.
Expected Results
With this series of tasks, the model is expected to achieve 60%-70% accuracy on the test set, although state-of-the-art results for CIFAR10 reach up to 99.7%.





Get to know  data: Load data and define datasets
CIFAR10 contains 5 batches that can be used for training/validation, and one batch that consists of the test set. In order to train your network, you will have to define a training set and a validation set. Do not use the test set as training data, and do not use any knowledge on the labels of the test set (being a publicly available dataset, we cannot avoid exposing the labels of the test set).

Think of the best way to split data into training and validation set. Note that the format that layers in convolutional networks like (at least in the Keras/Tensorflow libraries that we are using), is as follows:

(n_samples, rows, cols, n_channels)
This means that each training (but also validation and test) sample needs to have four dimensions. This kind of structure (multi-dimensional array), is called a tensor. In practice, this format is also convenient because the first index of the tensor refers to the sample itself, so we can use:

tensor[i]
to extract the i-th example.

During training, several samples will be used to update the parameters of a network. In the case of CIFAR10, if we use M samples per mini-batch, the shape of the mini-batch data is:

(M, 32, 32, 3)
Make sure data is organized in this way, for the training, validation and test datasets.
