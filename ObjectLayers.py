import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

# A method to create dataset. It creates points that look like spirals
def create_data(points, classes):
    # points -> sample number
    # classes -> output class number
    # There are 2 features. So each sample has 2 features.

    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros(shape=(1, n_neurons))
        # We can define weights the other way to get rid of the transpose operation
        # in the forward function
        # self.weights = 0.10 * np.random.randn(n_inputs, n_inputs)

    def forward(self, inputs):
        self.output = np.dot(inputs, np.array(self.weights).T) + self.biases


class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)  # ReLU function: if x>0 y=x else y=0


# Input -> Exponential (to get rid of negative values) -> Normalization (to keep the values between 0-1) -> Output
# Layer outputs -> M x N        M: # of samples     N: # of neurons in that layer
class Activation_Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # Since exponentiation grows too fast, we have to do some prior operations to prevent overflow.
        # input[u] = input[u] - max(inputs)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # To get the sum of each row axis must be 1
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


if __name__ == '__main__':
    X, y = create_data(100, 3)

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.show()

    # Since we have 2 features for each sample, there are 2 inputs coming from the input layer.
    layer1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    layer2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    layer1.forward(X)  # z = wT*x + b
    activation1.forward(layer1.output)  # y = ReLU(z)

    layer2.forward(layer1.output)  # z = wT*x + b   x -> layer1 output
    activation2.forward(layer2.output)  # y = softmax(z)

    print(activation1.output)
    print('####')
    print(activation2.output)

