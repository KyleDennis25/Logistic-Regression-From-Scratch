# DS 310 Final Project  
**Author**: Kyle Dennis

## Overview  

In this project, my goal is to fit a logistic regression model from scratch, using Python. Instead of using existing libraries such as scikit-learn for model fitting, I will build algorithms from the ground up. The purpose of this work is to help understand the mathematics behind logistic regression.  


The main algorithm I will implement is the **Newton-Raphson algorithm**. This algorithm, at the fundamental level, iteratively estimates the roots of a function. In this application, I will use it to estimate logistic regression parameters that best fit a dataset. The dataset I chose contains data on credit card transactions and indicates whether they were fraudulent or not. The logistic regression model will be trained to predict fraud using a subset of features.

## Mathematical Background  

A likelihood function measures how well a set of parameters describes a dataset (i.e. the probability of the dataset given the parameters). A common goal of machine learning optimization is to maximize the likelihood function of a dataset. That is, to find parameter values that best describe a dataset. Here, the **Newton-Raphson algorithm** will be used to compute logistic regression parameter values that maximize the log-likelihood function of the dataset (we take the log to avoid underflow). 


To apply the Newton-Raphson method, we complete the following steps: We approximate the log-likelihood as a quadratic function using the Taylor series. We then take the first derivative of this function with respect to the parameters (the parameters are represented as a vector) and set this equal to zero (a vector of zeros). We are left with a linear system we can solve for, where the solution is the step size of our parameter guesses (Heath). The Newton-Raphson method repeats this process until convergence (until we maximize the log-likelihood function).


To obtain the linear system, we need to calculate the **gradient vector** and the **Hessian matrix** of the log-likelihood function. The gradient vector contains the first-ordered partial derivatives with respect to each parameter. This represents the first derivative in the truncated Taylor series expansion. The Hessian matrix contains the second-ordered partial derivatives with respect to the parameters. This represents the second derivative in the truncated Taylor series expansion (Heath). I will implement two main algorithms for this — one to calculate the gradient and one to calculate the Hessian for each iteration of the Newton-Raphson algorithm.  


Once the best parameters are found, we can use them for classification. To do so, we can model the **logit** (the log-odds ratio) as a linear combination of parameters with predictor variables. We then plug in this linear combination into the sigmoid function to obtain the probability of success (Agresti). Then, we make a threshold for classification. Finally, we have a classification model. This model can classify fraudulent transactions when provided input (a transaction). For the purposes of this project, however, I will focus on model fitting rather than classification examples.  


Logistic regression at its core predicts the probability of a binary outcome. It has assumptions about the dataset we will be using. One important assumption is that the observations in the dataset are independent of each other. Another is that the observations are **Bernoulli trials**, which means they are random experiments with only two outcomes. In addition, it assumes the relationship between the logit transformation and the predictor variables is linear.

## Dataset  

The dataset I will use contains credit card transactions made in September 2013 by European cardholders. The data has been collected and analyzed during a research collaboration of Worldline and the Machine Learning Group of ULB on big data mining and fraud detection. The sample size of this dataset is 284,807 transactions.  


For model fitting, I will be using the independent variables **“V1”** and **“V2”** to predict the dependent variable **“Class”**. The V1 and V2 variables are principal components resulting from a principal components analysis. The original variables and their contexts are anonymous to maintain confidentiality (Machine Learning Group 2013). The class variable is binary: 1 represents a fraudulent transaction and 0 represents a non-fraudulent transaction.

## Reflection  

This project was very productive for me. I learned the core mathematical principles that underlie logistic regression. It helped me understand what really goes on "under the hood" during machine learning optimization.

## Citations  

- **Dataset**: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Other**:  
  - Heath, MT. 2005. *Scientific Computing: An introductory survey (2nd ed.)*. McGraw-Hill.  
  - Agresti, A. 2015. *Foundations of Linear and Generalized Linear Models*. John Wiley & Sons, Incorporated.
