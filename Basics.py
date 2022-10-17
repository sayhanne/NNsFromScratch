"""PART 1"""

import numpy as np


# !!! input is NOT the same as sample !!!

# rows -> sample index
# columns -> feature index
def simple_nn():
    inputs = [[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]

    # 3 neurons in the hidden layer 1. Each neuron has 4 incoming inputs
    weights1 = [[0.2, 0.8, -0.5, 1.0],
                [0.5, -0.91, 0.26, -0.5],
                [-0.26, -0.27, 0.17, 0.87]]

    biases1 = [2, 3, 0.5]

    # 3 neurons in the hidden layer 2. Each neuron has 3 incoming inputs
    weights2 = [[0.1, -0.14, 0.5],
                [-0.5, 0.12, -0.33],
                [-0.44, 0.73, -0.13]]

    biases2 = [-1, 2, -0.5]

    hidden_layer1_output = np.dot(inputs, np.array(weights1).T) + biases1
    hidden_layer2_output = np.dot(hidden_layer1_output, np.array(weights2).T) + biases2
    print(hidden_layer1_output)
    print(hidden_layer2_output)
