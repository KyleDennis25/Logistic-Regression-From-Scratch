import numpy as np
import pandas as pd
import math
import copy

def sigmoid(observation, parameters):
    probability = (1/(1 + math.exp(-1*(np.dot(observation, parameters)))))    
    return probability
'''
This function evaluates the sigmoid function, which maps logistic regression model inputs to probabilities.

Parameters
----------
observation: 1D NumPy array
A row of a design matrix.

parameters: 1D NumPy array
Vector containing logistic regression parameters.

Returns
-------
probability: float
The probability that the given observation has a true binary classification label of 1.
'''
 
def design_matrix(dataset):
    design_matrix = np.hstack((np.ones(len(dataset))[:, np.newaxis], dataset[:, :(len(dataset[0]) - 1)]))
    return design_matrix
'''
This function converts a dataset into a design matrix (it adds a column of 1's and gets rid of the dependent variable). 
This design matrix will be used in the later functions for calculations.

Parameters
----------
dataset: 2D NumPy array 
Dataset in the form of an array.

Returns
-------
design_matrix: 2D NumPy array
Matrix containing a column of 1's for the regression intercept and the independent variables.
'''  

def probability_iteration(design_matrix, parameter_vector):
    probability_vector = np.zeros(len(design_matrix))
    for i in range(0, len(design_matrix)):
        probability_vector[i] = sigmoid(design_matrix[i], parameter_vector)
    return probability_vector
'''
This function calculates the probability of each row of the design matrix (it evaluates the sigmoid for each row). These 
probabilities are used to construct the gradient vector and the hessian matrix of the log-likelihood.

Parameters
----------
design_matrix: 2D NumPy array 
Matrix containing a column of 1's for the regression intercept and the independent variables.

parameter_vector: 1D NumPy array
Vector containing logistic regression parameters.

Returns
-------
probability_vector: 1D NumPy array
Vector containing probabilities of each row of the design matrix.
'''

def Gradient(dataset, design_matrix, probability_vector):
    gradient_vector = np.zeros(len(design_matrix[0]))
    summation = 0
    for j in range(0, len(gradient_vector)):
        for i in range(0, len(design_matrix)): 
            summation += (dataset[i, len(dataset[0]) - 1] - probability_vector[i])*design_matrix[i, j]
        gradient_vector[j] = summation
        summation = 0
    return gradient_vector  
'''
This function constructs the gradient vector of the log-likelihood (iterative formula found in Agresti, section 5.4.1, 
page 176- cited in the report).

Parameters
----------
dataset: 2D NumPy array 
Dataset in the form of an array.

design_matrix: 2D NumPy array 
Matrix containing a column of 1's for the regression intercept and the independent variables.

probability_vector: 1D NumPy array
Vector containing probabilities of each row of the design matrix.

Returns
-------
gradient_vector: 1D NumPy array
Gradient vector of the log-likelihood.
'''

def Hessian(design_matrix, probability_vector):
    hessian = np.zeros((len(design_matrix[0]), len(design_matrix[0])))
    operations = np.zeros(len(probability_vector))
    for element in range(0, len(operations)):
        operations[element] = probability_vector[element] * (1 - probability_vector[element]) 
    summation = 0
    for a in range(0, len(hessian)):
        for b in range(0, len(hessian[0])):
            for i in range(0, len(design_matrix)):
                summation += (design_matrix[i, a] * design_matrix[i, b]) * (operations[i])
            hessian[a, b] = -1 * summation 
            summation = 0
    return hessian
'''
This function constructs the hessian matrix of the log-likelihood (iterative formula found in Agresti, section 5.4.1, 
page 176- cited in the report).

Parameters
----------
design_matrix: 2D NumPy array 
Matrix containing a column of 1's for the regression intercept and the independent variables.

probability_vector: 1D NumPy array
Vector containing probabilities of each row of the design matrix.

Returns
-------
hessian: 2D NumPy array
Hessian matrix of the log-likelihood.
'''

def Newton_Raphson_Method(dataset, design_matrix, parameter_estimates):
    i = np.zeros(len(parameter_estimates))
    while(np.linalg.norm(parameter_estimates - i) > 1e-8):
        i = copy.deepcopy(parameter_estimates)
        probability_vector = probability_iteration(design_matrix, parameter_estimates)
        hessian = Hessian(design_matrix, probability_vector)
        gradient = Gradient(dataset, design_matrix, probability_vector)
        x = np.linalg.solve(hessian, -1*(gradient))
        parameter_estimates += x
        
    return parameter_estimates
'''
This function iteratively estimates logistic regression model parameters, which maximize the log-likelihood of the 
dataset. This is done by following the Newton-Raphson method.

Parameters
----------
dataset: 2D NumPy array 
Dataset in the form of an array.

design_matrix: 2D NumPy array 
Matrix containing a column of 1's for the regression intercept and the independent variables.

parameter_estimates: 1D NumPy array
Vector containing inital logistic regression parameter estimates (initial parameter guesses).

Returns
-------
parameter_estimates: 1D NumPy array
Estimated logistic regression parameters after convergence has been reached.
'''