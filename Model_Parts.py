import numpy as np
# import matplotlib.pyplot as plt

np.random.seed(0)


# axis = 0 : rows
# axis = 1 : columns

# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

# A method to create dataset. It creates samples that look like spirals
def create_data(samples, classes):
    # samples -> sample number
    # classes -> output class number
    # There are 2 features. So each sample has 2 features.

    X_data = np.zeros((samples * classes, 2))
    y_data = np.zeros(samples * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.2
        X_data[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y_data[ix] = class_number
    return X_data, y_data


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.output = None
        # To store the derivative values in backward pass
        self.d_inputs = None
        self.d_weights = None
        self.d_biases = None
        # To remember the inputs when calculating the gradients
        self.inputs = None
        # We can define weights pre-transposed to get rid of the transpose operation
        # each time in the forward function
        # Normally -> self.weights = 0.10 * np.random.randn(n_neurons, n_inputs)
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(shape=(1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs  # For gradient calculation
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, d_values):  # d_values -> derivatives coming from next layer
        # Gradient for the parameters
        self.d_weights = np.dot(self.inputs.T, d_values)  # Chain Rule
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)  # Chain Rule
        # Gradient for the inputs (to use in the previous layer in the next iteration of the back prop.)
        self.d_inputs = np.dot(d_values, self.weights.T)  # Chain Rule


class Activation_ReLU:
    def __init__(self):
        self.output = None
        # To store the derivative values in backward pass
        self.d_inputs = None
        # To remember the inputs when calculating the gradients
        self.inputs = None

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs  # For gradient calculation
        self.output = np.maximum(0, inputs)  # ReLU function: if x>0 y=x else y=0

    # Backward pass
    def backward(self, d_values):  # d_values -> derivatives coming from next layer
        # Gradient for ReLU activation function
        self.d_inputs = np.zeros_like(self.inputs)
        self.d_inputs[self.inputs > 0] = 1
        self.d_inputs *= d_values  # Chain Rule


# Input -> Exponential (to get rid of negative values) -> Normalization (to keep the values between 0-1) -> Output
# Layer outputs -> M x N        M: # of samples     N: # of neurons in that layer
class Activation_Softmax:
    def __init__(self):
        self.d_inputs = None  # For gradient calculation
        self.output = None

    def forward(self, inputs):
        # Since exponentiation grows too fast, we have to do some prior operations to prevent overflow.
        # input[u] = input[u] - max(inputs)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # To get the sum of each row axis must be 1
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    # (explained in the NNFS book, not going into detail since we will use the combined partial derivative
    # of loss function and softmax function)
    def backward(self, d_values):
        # Create uninitialized array
        self.d_inputs = np.zeros_like(d_values)
        # Enumerate outputs and gradients
        for index, (single_output, single_d_values) in enumerate(zip(self.output, d_values)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.d_inputs[index] = np.dot(jacobian_matrix, single_d_values)


class Loss:
    def __init__(self):
        self.d_inputs = None  # The derivative values that'll be used in previous layer

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

        correct_confidences = None
        if len(y_target.shape) == 1:  # scalar valued outputs
            # Getting the value of our prediction which corresponds to our target class index
            correct_confidences = y_pred_clipped[range(samples), y_target]
        elif len(y_target.shape) == 2:  # one-hot vector
            # Take dot product of the predicted outputs(confidences) and target values(one-hot vector).
            # Then take the sum of each row to put get the input which we will put to log function later.
            correct_confidences = np.sum(y_pred_clipped * y_target, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # The derivative of this loss function with respect to its inputs
    # (predicted values at the i-th sample, since we are interested in a gradient
    # with respect to the predicted values) equals the negative ground-truth vector,
    # divided by the vector of the predicted values (which is also the output vector of the softmax function)
    def backward(self, d_values, y_target):

        # Number of samples
        samples = len(d_values)
        # Number of classes in each sample
        classes = len(d_values[0])

        # If labels are spars (just true class), turn them into one-hot vector
        if len(y_target.shape) == 1:
            y_target = np.eye(classes)[y_target]

        # Calculate gradient
        self.d_inputs = -y_target / d_values

        # Normalize gradient
        self.d_inputs = self.d_inputs / samples


class Activation_Softmax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.d_inputs = None  # The derivative values that'll be used in previous layer
        self.output = None  # Activation Softmax result -> confidences
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_target):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_target)

    # Backward pass
    def backward(self, d_values, y_target):
        # Number of samples
        samples = len(d_values)
        # If classes are one-hot encoded vector, turn them into discrete values (true class index)
        if len(y_target.shape) == 2:
            y_target = np.argmax(y_target, axis=1)

        self.d_inputs = d_values.copy()
        # Calculate gradient over all samples only on the true class indexes
        self.d_inputs[range(samples), y_target] -= 1  # combined derivative of loss and softmax -> y_pred - y_target
        # # Normalize gradient
        self.d_inputs = self.d_inputs / samples


if __name__ == '__main__':
    X, y = create_data(samples=100, classes=3)

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
    #
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    # plt.show()

    # Since we have 2 features for each sample, there are 2 inputs coming from the input layer.
    layer1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()

    layer2 = Layer_Dense(3, 3)
    # activation2 = Activation_Softmax()

    activation_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()
    layer1.forward(X)  # z = wT*x + b
    activation1.forward(layer1.output)  # y = ReLU(z)

    layer2.forward(activation1.output)  # z = wT*x + b   x -> activation1 output
    loss = activation_loss.forward(layer2.output, y)  # combined loss + activation (forward, backward)
    print(activation_loss.output[:5])
    print("Loss: ", loss)

    # activation2.forward(layer2.output)   # y = softmax(z)
    # print(activation2.output[:5])

    # loss_function = Loss_CategoricalCrossEntropy()
    # loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of last activation function and targets
    # calculate values along first axis
    predictions = np.argmax(activation_loss.output, axis=1)
    if len(y.shape) == 2:  # Check if it's one-hot vector
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    # Print accuracy
    print('acc:', accuracy)

    # 1 simple backward pass for 2 layers
    activation_loss.backward(activation_loss.output, y)
    layer2.backward(activation_loss.d_inputs)
    activation1.backward(layer2.d_inputs)
    layer1.backward(activation1.d_inputs)

    # Print gradients of the parameters
    print(layer1.d_weights)
    print(layer1.d_biases)
    print(layer2.d_weights)
    print(layer2.d_biases)
