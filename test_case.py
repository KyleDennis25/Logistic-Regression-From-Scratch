from functions import design_matrix, Newton_Raphson_Method
import numpy as np
import pandas as pd


# Load in dataset
full_data = pd.read_csv("dataset.csv")

# Select relevant columns
subset_data = full_data[['V1', 'V2', 'Class']]

# Convert dataframe into a numpy array
subset_data_values = subset_data.values
 
# Create design matrix for three variable dataset
design_matrix_ = design_matrix(subset_data_values) 

# Make initial guess for true parameters
initial_params = np.array([.50, .70, .12])  
 
# Use Newton Raphson Method to find logistic regression parameters which maximize the log-likelihood function.
params_vector = Newton_Raphson_Method(subset_data_values, design_matrix_, initial_params) 

print(params_vector)
# The algorithm converged after about 35 seconds on my machine. The paremeters are: -6.96475858, -0.10128139, and 
# 0.50197949.