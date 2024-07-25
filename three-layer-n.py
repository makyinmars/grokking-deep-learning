import numpy as np


# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Create a three layer neural network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    # Initialize the weights
    self.weights1 = np.random.rand(input_size, hidden_size)
    self.weights2 = np.random.rand(hidden_size, output_size)

    # Initialize the biases
    self.bias1 = np.random.rand(1, hidden_size)
    self.bias2 = np.random.rand(1, output_size)

    def forward(self, x):
    # Forward pass
    self.x = x
    self.hidden = relu(x + self.bias1)
    self.output = sigmoid(self.hidden + self.bias2)
    return self.output

    def backward(self, target):
    # Backward pass
    self.error = target - self.output
    self.output_error = self.error * sigmoid_derivative(self.output)
    self.hidden_error = self.output_error + self.hidden_error
    self.hidden_error = self.hidden_error * relu_derivative(self.hidden)

    # Update the weights
    self.weights1 += self.x * self.hidden_error
    self.weights2 += self.hidden * self.output_error

    # Update the biases
    self.bias1 += self.hidden_error
    self.bias2 += self.output_error

    def __str__(self):
    return 'NeuralNetwork(input_size={}, hidden_size={}, output_size={})'.format(self.input_size, self.hidden_size, self.output_size)
