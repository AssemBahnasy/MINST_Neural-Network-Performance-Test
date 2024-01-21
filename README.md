# Neural Network for MINST Dataset Classification

## Overview

This repository contains a simple mathematical implementation of a neural network using only numpy, without any importation of predesigned libraries, to classify the MNIST dataset. The neural network has one hidden layer, and the number of neurons in the hidden layer can be adjusted. The purpose of this implementation is to test the performance of the neural network with different hidden layer configurations.

## Table of Contents

1. [Dependencies](#dependencies)
2. [MINST Dataset](#minst-dataset)
3. [Neural Network](#neural-network)
    - 3.1. [Data Preparation](#data-preparation)
    - 3.2. [One-hidden Layer Model](#one-hidden-layer-model)
       - 3.2.1 [Activation Functions](#activation-functions)
       - 3.2.2 [Parameters Initialization, Forward/Backward Propagations, and Parameters Updating](#parameters-initiation-forwardbackward-propagations-and-parameters-updating)
       - 3.2.3 [Gradient Descent, Predictions, and Accuracy](#gradient-descent-predictions-and-accuracy)
4. [Neural Network Training](#neural-network-training-with-variable-hidden-layer-neurons)
    - 4.1. [Model Training](#Model-Training)
    - 4.2  [Model Evaluation](#Model-Evaluation)
5. [Model Performance Viualization](#Model-Performance-Viualization)
6. [Conclusion](#Conclusion)

## Dependencies

Ensure you have the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib

## MINST Dataset

The MINST dataset is read using the Pandas library. The training data is loaded from the CSV file located at [path]\train.csv
[Reference Dataset][[https://www.kaggle.com/competitions/digit-recognizer/overview]

```python
df = pd.read_csv(r"[path]\train.csv")
df.head()
```
## Neural Network
### Data Preparation
- The dataset is processed using NumPy arrays for efficient mathematical computations. 
- Data normalization is performed, and the dataset is split into training and testing sets.
```python
data_test = data[:1000].T
Y_test = data_test[0]
X_test = data_test[1:n] / 255.

data_train = data[1000:m].T 
Y_train = data_train[0]
X_train = data_train[1:n] / 255.
n_train, _ = X_train.shape
```
### One-hidden Layer Model
#### Activation Functions

Two activation functions are implemented:
- Rectified Linear Unit (ReLU) for the hidden layer
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
```

#### Parameters Initialization, Forward/Backward Propagations, and Parameters Updating

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
```

#### Gradient Descent, Predictions, and Accuracy Optimization

The training process involves gradient descent with a specified learning rate, and accuracy is calculated at regular intervals.

```python
for neurons in n_neurons:
    W1, b1, W2, b2 = initParams(neurons) # To initiate parameters of the hidden and output layer
    iteration_list = []  # To store iteration numbers
    accuracy_list = []   # To store corresponding accuracies
    print('\nHidden Layer: '+ str(neurons) + ' Neurons\n' + 25*'*')
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backwardProp(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            
        if i % 10 == 0:
            print(f"Iteration: {i + 10}")
            iteration_list.append(i + 10)
            predictions = getPredictions(A2) 
            print(predictions, Y)
            accuracy = getAccuracy(predictions, Y)
            accuracy_list.append(accuracy)
            print(f"Accuracy: {accuracy * 100:.2f}%")

```
## Neural Network Training with Variable Hidden Layer Neurons

The neural network is trained with different hidden layer neuron configurations (8, 10, 16, 32, 64) for a specified number of iterations (400) and learning rate (0.10).

### Model Training
```python
n_neurons = [8, 10, 16, 32, 64] # Specified number of neurons in the hidden layer
iterations = 400 # Specified number of iterations
alpha = 0.10 # Specified training rate
X = X_train # Trained data
Y = Y_train # Data True labels
W1_list = []
b1_list = []
W2_list = []
b2_list = []
model_accuracy = []

W1_list, b1_list, W2_list, b2_list, model_accuracy, iteration_list = gradientDescent(X, Y, n_neurons, alpha, iterations)
```

### Model Evaluation
```python
def getPredictions(A2):
    return np.argmax(A2, 0)

# Calculate the accuracy of a set of predictions compared to the true labels
def getAccuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size
```

## Model Performance Viualization

A visual representation of the accuracy of different neural network models was created as a function of the number of neurons in each model. The code iterates over a list of model_accuracy data, where each element corresponds to the accuracy achieved during training for a specific number of neurons. For each set of accuracy values, a line plot is generated using Matplotlib. The x-axis represents the iterations, and the y-axis represents the accuracy percentage.

## Conclusion

This simple neural network achieved a test accuracy of approximately 90% on the MNIST dataset. The performance can be further improved by tuning hyperparameters, increasing the complexity of the model, or using more advanced techniques such as convolutional neural networks (CNNs).
