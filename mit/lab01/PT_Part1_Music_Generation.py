import torch
import torch.nn as nn

# MIT Introduction to Deep Learning package
import mitdeeplearning as mdl
 
import numpy as np
import matplotlib.pyplot as plt

def tensor_intitialization():
    # Add the main logic of your script here
    print("Music generation script is running!")
    
    # 0-dimensional tensors
    integer = torch.tensor(12345)
    decimal = torch.tensor(3.1414926)
    
    # Print section for 0-dimensional tensors
    print(f"'integer' is a {integer.ndim}-d Tensor: {integer}")
    print(f"'decimal' is a {decimal.ndim}-d Tensor: {decimal}")
    
    # 1-dimensional tensors
    fibonacci = torch.tensor([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])  # First 10 Fibonacci numbers
    count_to_100 = torch.tensor(list(range(100)))  # Numbers from 0 to 99
    
    # Print section for 1-dimensional tensors
    print(f"'fibonacci' is a {fibonacci.ndim}-d Tensor with shape {fibonacci.shape}")
    print(f"'count_to_100' is a {count_to_100.ndim}-d Tensor with shape {count_to_100.shape}")
    
    ### Defining higher-order Tensors ###
    
    '''Define a 2-d Tensor'''
    matrix = torch.tensor([[1, 2, 3], 
                           [4, 5, 6], 
                           [7, 8, 9]])
    assert isinstance(matrix, torch.Tensor), "matrix must be a torch.Tensor object"
    assert matrix.ndim == 2, "matrix must be a 2-d Tensor"
    
    print(f"'matrix' is a {matrix.ndim}-d Tensor with shape {matrix.shape}")
    
    '''Define a 4-d Tensor'''
    images = torch.zeros(10, 3, 256, 256)
    assert isinstance(images, torch.Tensor), "images must be a torch.Tensor object"
    assert images.ndim == 4, "images must be a 4-d Tensor"
    assert images.shape == (10, 3, 256, 256), f"images must have shape (10, 3, 256, 256), but got {images.shape}"
    
    print(f"'images' is a {images.ndim}-d Tensor with shape {images.shape}")
    
    row_vector = matrix[1]
    column_vector = matrix[:, 1]
    scalar = matrix[0, 1]
    
    print(f"'row_vector' is a {row_vector.ndim}-d Tensor with shape {row_vector.shape} and contents: {row_vector}")
    print(f"'column_vector' is a {column_vector.ndim}-d Tensor with shape {column_vector.shape} and contents: {column_vector}")
    print(f"'scalar' is a 0-d Tensor with value: {scalar}")
    
def tensor_computation():
    # Create the nodes in the graph and initialize values
    a = torch.tensor(15)
    b = torch.tensor(61)
    
    # Add them
    c1 = torch.add(a, b)
    c2 = a + b
    
    # Print results
    print(f"c1: {c1}")
    print(f"c2: {c2}")
    
    # Consider axample values for a, b
    a, b = 1.5, 2.5
    # Execute the computation
    e_out = compute(a, b)
    print(f"e_out: {e_out}")
    
### Define Tensonr Computations ###

# Construct a simple computation function
def compute(a, b):
    # Add two tensors
    c = a + b
    # Multiply the result by 2
    d = b - 1
    return c * d

#
# NN
#

### Define a dence layer ###

# num_inputs: number of input nodes
# num_outputs: number of output nodes
# x: input to the layer

class DenseLayer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DenseLayer, self).__init__()
        # Define and initialize parameters: a weight matrix W and bias b
        # Note that the parameter initialize is random.
        self.W = torch.nn.Parameter(torch.randn(num_inputs, num_outputs))
        self.bias = torch.nn.Parameter(torch.randn(num_outputs))
    
    def forward(self, x):
        z = torch.matmul(x, self.W) + self.bias
        y = torch.sigmoid(z)
        return y

    
def neural_network():
    # Define a layer and test the output
    num_inputs = 2
    num_outputs = 3
    layer = DenseLayer(num_inputs, num_outputs)
    x_input = torch.tensor([1.0, 2.0])
    y_output = layer(x_input)
    print(f"Input shape: {x_input.shape}")
    print(f"Input: {x_input}")
    print(f"Output shape: {y_output.shape}")
    print(f"Output: {y_output}")
    
def sequential_nn():
    ### Defining s neutal network using the PyTorch Sequential API ###
    #define the model of inputs and outputs
    n_inputs_nodes = 2
    n_output_nodes = 3
    # Define the model
    '''Use the Sequential API to define a neural network with a
        single linear (dense!) layer, folloed by non-linearity to compute z'''
    model = torch.nn.Sequential(
        # linear layer with input size 2 and output size 3
        torch.nn.Linear(n_inputs_nodes, n_output_nodes),
        # Sigmoid activation function
        torch.nn.Sigmoid()
    )
    # Test the model with example input
    x_input = torch.tensor([[1, 2.]])
    model_output = model(x_input)
    # Print the model output
    print(f"Input shape: {x_input.shape}")
    print(f"Output shape: {model_output.shape}")
    print(f"Output result: {model_output}")

# 
# Main function
#
    
def main():
    tensor_intitialization()
    tensor_computation()
    neural_network()

if __name__ == "__main__":
    main()




