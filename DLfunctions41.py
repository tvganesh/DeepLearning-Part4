# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 19:10:25 2018

@author: Ganesh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm



# Conmpute the Relu of a vector
def relu(Z):
    A = np.maximum(0,Z)
    cache=Z
    return A,cache

# Conmpute the softmax of a vector
def softmax(Z):  
    # get unnormalized probabilities
    exp_scores = np.exp(Z.T)
    # normalize them for each example
    A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)   
    cache=Z
    return A,cache

# Compute the detivative of Relu 
def reluDerivative(dA, cache):
  
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.  
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0 
    return dZ


# Compute the derivative of softmax
def softmaxDerivative(dA, cache,y,numTraining):
      # Note : dA not used. dL/dZ = dL/dA * dA/dZ = pi-yi
      Z = cache 
      # Compute softmax
      exp_scores = np.exp(Z.T)
      # normalize them for each example
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  
      
      # compute the gradient on scores
      dZ = probs
      # dZ = pi- yi
      dZ[range(int(numTraining)),y] -= 1
      return(dZ)
      

# Initialize the model 
# Input : number of features
#         number of hidden units
#         number of units in output
# Returns: Weight and bias matrices and vectors
def initializeModel(numFeats,numHidden,numOutput):
    np.random.seed(1)
    W1=np.random.randn(numHidden,numFeats)*0.01 #  Multiply by .01 
    b1=np.zeros((numHidden,1))
    W2=np.random.randn(numOutput,numHidden)*0.01
    b2=np.zeros((numOutput,1))
    
    # Create a dictionary of the neural network parameters
    nnParameters={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    return(nnParameters)



# Compute the activation at a layer 'l' for forward prop in a Deep Network
# Input : A_prec - Activation of previous layer
#         W,b - Weight and bias matrices and vectors
#         activationFunc - Activation function - sigmoid, tanh, relu etc
# Returns : The Activation of this layer
#         : 
# Z = W * X + b
# A = sigmoid(Z), A= Relu(Z), A= tanh(Z)
def layerActivationForward(A_prev, W, b, activationFunc):
    
    # Compute Z
    Z = np.dot(W,A_prev) + b
    forward_cache = (A_prev, W, b) 
    # Compute the activation for sigmoid
    if activationFunc == "sigmoid":
        A, activation_cache = sigmoid(Z)  
    # Compute the activation for Relu
    elif activationFunc == "relu":  
        A, activation_cache = relu(Z)
    # Compute the activation for tanh
    elif activationFunc == 'tanh':
        A, activation_cache = tanh(Z)  
    elif activationFunc == 'softmax':
        A, activation_cache = softmax(Z)  
    cache = (forward_cache, activation_cache)
    return A, cache



# Compute the backpropoagation for 1 cycle
# Input : Neural Network parameters - dA
#       # cache - forward_cache & activation_cache
#       # Input features
#       # Output values Y
# Returns: Gradients
# dL/dWi= dL/dZi*Al-1
# dl/dbl = dL/dZl
# dL/dZ_prev=dL/dZl*W
def layerActivationBackward(dA, cache, y, activationFunc):
    forward_cache, activation_cache = cache
    A_prev, W, b = forward_cache
    numtraining = float(A_prev.shape[1])
    if activationFunc == "relu":
        dZ = reluDerivative(dA, activation_cache)           
    elif activationFunc == "sigmoid":
        dZ = sigmoidDerivative(dA, activation_cache)      
    elif activationFunc == "tanh":
        dZ = tanhDerivative(dA, activation_cache)
    elif activationFunc == "softmax":
        dZ = softmaxDerivative(dA, activation_cache,y,numtraining)
  
    if activationFunc == 'softmax':
        dW = 1/numtraining * np.dot(A_prev,dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ,W)
    else:
        #print(numtraining)
        dW = 1/numtraining *(np.dot(dZ,A_prev.T))
        #print("dW=",dW)
        db = 1/numtraining * np.sum(dZ, axis=1, keepdims=True)
        #print("db=",db)
        dA_prev = np.dot(W.T,dZ)
        
    return dA_prev, dW, db


# Plot a decision boundary
# Input : Input Model,
#         X
#         Y
#         sz - Num of hiden units
#         lr - Learning rate
#         Fig to be saved as
# Returns Null
def plot_decision_boundary(X, y,W1,b1,W2,b2,fig1):
    #plot_decision_boundary(lambda x: predict(parameters, x.T), x1,y1.T,str(0.3),"fig2.png") 
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W1.T) + b1.T), W2.T) + b2.T
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
