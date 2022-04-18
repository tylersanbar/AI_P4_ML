### code base: ai.berkeley.edu

import nn


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #Input 784d array
        #Output 10d array representing 0-9 digits, 0 for all except 1 for classification
        #98% validation accuracy 
        feature_size = 784
        output_size = 10
        #dataset.get_validation_accuracy()
        #Hidden Layer Size - (10, 400)
        hidden_layer_size = 300
        self.num_hidden_layers = 2
        #Batch Size - (1, dataset size) 
        self.batch_size = 500
        #Learning Rate - (0.001, 1.0)
        self.alpha = .5

        self.hidden_layers = []
        for layer in range(self.num_hidden_layers):
            if layer == 0:
                W = nn.Parameter(feature_size, hidden_layer_size)
                b = nn.Parameter(1, hidden_layer_size)
            else:
                W = nn.Parameter(hidden_layer_size, hidden_layer_size)
                b = nn.Parameter(1, hidden_layer_size)
            self.hidden_layers.append([W, b])
        self.output_w = nn.Parameter(hidden_layer_size, output_size)
        self.output_b = nn.Parameter(1, output_size)
        #Number of hidden layers - (1, 3)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #f(x)=relu(x⋅W1+b1)⋅W2+b2
        def hiddenRelu(layer, x):
            W = self.hidden_layers[layer][0]
            b = self.hidden_layers[layer][1]
            #x⋅W
            x_W = nn.Linear(x, W)
            #x⋅W+b
            add_b = nn.AddBias(x_W, b)
            #relu(x⋅W+b)
            relu = nn.ReLU(add_b)
            layer += 1
            if layer == self.num_hidden_layers: return relu
            else: return hiddenRelu(layer, relu)

        relu = hiddenRelu(0, x)
        #relu(x⋅W1+b1)⋅W2
        times_output_W = nn.Linear(relu, self.output_w)
        #relu(x⋅W1+b1)⋅W2+b2
        logits = nn.AddBias(times_output_W, self.output_b)

        return logits


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(x), y)
        print(nn.as_scalar(loss))
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for example in dataset.iterate_forever(self.batch_size):
            x = example[0]
            y = example[1]
            if dataset.get_validation_accuracy() > .98: break
            loss = self.get_loss(x, y)
            params = []
            for layer in range(self.num_hidden_layers):
                params.append(self.hidden_layers[layer][0])
                params.append(self.hidden_layers[layer][1])
            params.append(self.output_w)
            params.append(self.output_b)
            grads = nn.gradients(loss, params)
            #I negate the learning rate instead of the gradient direction
            for layer in range(self.num_hidden_layers):
                self.hidden_layers[layer][0].update(grads[layer * 2], -1 * self.alpha)
                self.hidden_layers[layer][1].update(grads[layer * 2 + 1], -1 * self.alpha)
            self.output_w.update(grads[-2], -1 * self.alpha)
            self.output_b.update(grads[-1], -1 * self.alpha)





