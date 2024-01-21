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

## Dependencies

Ensure you have the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib

## MINST Dataset

The MINST dataset is read using the Pandas library. The training data is loaded from the CSV file located at [path]\train.csv

```python
# Pandas library is used to read only the dataset 
df = pd.read_csv(r"[path]\train.csv")
df.head()
```
## Neural Network
### Data Preparation
- The dataset is processed using NumPy arrays for efficient mathematical computations. 
- Data normalization is performed, and the dataset is split into training and testing sets.

```python

data = np.array(df)
m, n = data.shape # m: number of records (42000); n: number of fields (785)
np.random.shuffle(data) # Shuffling the data so that our results would not be biased by their arrangement

data_test = data[:1000].T
Y_test = data_test[0]
X_test = data_test[1:n] / 255.

data_train = data[1000:m].T 
Y_train = data_train[0]
X_train = data_train[1:n] / 255.
n_train, _ = X_train.shape

### One-hidden Layer Model

Activation Functions

Two activation functions are implemented:

- Rectified Linear Unit (ReLU)
- Derivative of ReLU activation function
- Softmax

```python
def ReLU(Z):
    return np.maximum(0, Z)

# ReLU Derivative
def derivReLU(Z):
    return np.where(Z > 0, 1, 0)

def Softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


### Parameters Initialization, Forward/Backward Propagations, and Parameters Updating

The neural network parameters, including weights and biases, are initialized, and forward/backward propagations are implemented. Gradient descent is used for parameter updates.

```python
def initParams(n_neurons):
    # weight(1) and bias(1) for the hidden layer of (n) neurons
    # Applying a scaling factor (np.sqrt(1/n)) to help mitigate the vanishing/exploding gradient problems
    W1 = np.random.normal(size=(n_neurons, n_train)) * np.sqrt(1./n_train)
    b1 = np.random.normal(size=(n_neurons, 1)) * np.sqrt(1./n_neurons)
    
    # weight(2) and bias(2) for the output layer of 10 neurons
    W2 = np.random.normal(size=(10, n_neurons)) * np.sqrt(1./n_neurons)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./n_train)
    return W1, b1, W2, b2

def forwardProp(W1, b1, W2, b2, X):
    # Hidden Layer: N neurons
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    
    # Output layer: 10 neurons
    Z2 = W2.dot(A1) + b2
    A2 = Softmax(Z2)
    return Z1, A1, Z2, A2

# Converting/Encoding digit labels into a one-hot encoded format
def oneHot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backwardProp(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = oneHot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * derivReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1    
    W2 -= alpha * dW2  
    b2 -= alpha * db2
    return W1, b1, W2, b2

### Gradient Descent, Predictions, and Accuracy

The training process involves gradient descent with a specified learning rate, and accuracy is calculated at regular intervals.

## Neural Network Training with Variable Hidden Layer Neurons

The neural network is trained with different hidden layer neuron configurations (8, 10, 16, 32, 64) for a specified number of iterations (400) and learning rate (0.10).
