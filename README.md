# Neural Network for MINST Dataset Classification

## Overview

This repository contains a simple implementation of a neural network using numpy for the MNIST dataset. The neural network has one hidden layer, and the number of neurons in the hidden layer can be adjusted. The purpose of this implementation is to test the performance of the neural network with different hidden layer configurations.

## Table of Contents

1. [Dependencies](#dependencies)
2. [MINST Dataset](#minst-dataset)
3. [Neural Network](#neural-network)
    1. [Data Preparation](#data-preparation)
    2. [One-hidden Layer Model](#one-hidden-layer-model)
        - [Activation Functions](#activation-functions)
        - [Parameters Initialization, Forward/Backward Propagations, and Parameters Updating](#parameters-initiation-forwardbackward-propagations-and-parameters-updating)
        - [Gradient Descent, Predictions, and Accuracy](#gradient-descent-predictions-and-accuracy)
4. [Neural Network Training](#neural-network-training-with-variable-hidden-layer-neurons)

## 1. Dependencies

Ensure you have the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib

```bash
pip install numpy pandas matplotlib

## 2. MINST Dataset

The MINST dataset is read using the Pandas library. The training data is loaded from the CSV file located at [path]\train.csv.

## 3. Neural Network
### 3.1 Data Preparation

The dataset is processed using NumPy arrays for efficient mathematical computations. Data normalization is performed, and the dataset is split into training and testing sets.

### 3.2 One-hidden Layer Model

Activation Functions

Two activation functions are implemented:

    - Rectified Linear Unit (ReLU)
    - Softmax

Parameters Initialization, Forward/Backward Propagations, and Parameters Updating

The neural network parameters, including weights and biases, are initialized, and forward/backward propagations are implemented. Gradient descent is used for parameter updates.

### 3.3 Gradient Descent, Predictions, and Accuracy

The training process involves gradient descent with a specified learning rate, and accuracy is calculated at regular intervals.

## 4. Neural Network Training with Variable Hidden Layer Neurons

The neural network is trained with different hidden layer neuron configurations (8, 10, 16, 32, 64) for a specified number of iterations (400) and learning rate (0.10).
