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
