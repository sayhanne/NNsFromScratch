import numpy as np
import matplotlib.pyplot as plt
import math

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
        self.weights = 0.01 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros(shape=(1, n_neurons))
        # We can define weights the other way to get rid of the transpose operation
        # in the forward function
        # self.weights = 0.10 * np.random.randn(n_inputs, n_weights)

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


class Loss:
    def calculate(self, output, target):
        sample_losses = self.forward(output, target)
        cost = np.mean(sample_losses)
        return cost


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_target):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # first check the output format
        # whether if it's one-hot vector or just a scalar value
        # one-hot vector -> a vector that has 1 in the index of the correct class, others are 0.
        # scalar value -> indicating the correct class

        if len(y_target.shape) == 1:  # scalar valued outputs
            # Getting the value of our prediction which corresponds to our target class index
            correct_confidences = y_pred_clipped[range(samples), y_target]
        elif len(y_target.shape) == 2:  # one-hot vector
            # Take dot product of the predicted outputs(confidences) and target values(one-hot vector).
            # Then take the sum of each row to put get the input which we will put to log function later.
            correct_confidences = np.sum(y_pred_clipped * y_target, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


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

    layer2.forward(activation1.output)  # z = wT*x + b   x -> activation1 output
    activation2.forward(layer2.output)  # y = softmax(z)

    print(activation2.output[:5])

    loss_function = Loss_CategoricalCrossEntropy()
    loss = loss_function.calculate(activation2.output, y)
    print("Loss: ", loss)
