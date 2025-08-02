"""
Simple feedforward neural network model using fully connected dense layers, 
rectified Linear Activation Function (ReLU), Mean Squared Error (MSE) loss for training and validation,
Root Mean Squared Error (RMSE) loss for testing and Adam Optimizer.

Designed for basic regression tasks.
"""

import matplotlib.pyplot as plt
import numpy as np 
import pickle

class Layer_Dense:

    """ 
    Fully connected dense layer.

    Performs linear transformation on the input: output = inputs * weights + biases.

    Args:
        n_inputs (int): number of input features for the layer
        
        n_neurons (int): number of neurons (outputs) for the layer

    Attributes:
        weights (np.ndarray): Weight matrix of shape (n_inputs, n_neurons)
            initialized with He initialzing to prevent early neuron deaths
        
        biases (np.ndarray): Bias matrix of shape (1, n_neurons), initialized with zeros
        
        outputs (np.ndarray): Output after forward pass
    """
    
    def __init__(self, n_inputs, n_neurons): 

        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
       
        """
        Performs the forward pass.

        Args: 
            inputs (np.ndarray): Input data of shape (n_samples, n_inputs)

        Sets:
            outputs (np.array): Computed outputs of shape (n_samples, n_neurons)
        """

        self.inputs = inputs 
        self.outputs = np.dot(self.inputs, self.weights) + self.biases 

    def backward(self, dvalues):

        """
        Performs the backward pass

        Computes gradients of the loss with respect to weights, biases, and inputs.
        These gradients are passed to the previous layer during backpropagation.

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the layers outputs
            shape(n_samples, n_neurons)

        Sets:
            dweights (np.ndarray): Gradient of the loss with respect to the layers weights
            shape(n_inputs, n_neurons)
            
            dbiases (np.array): Gradient of the loss with respect to the layers biases
            shape(1, neurons)
            
            dinputs (np.array): Gradient of the loss with respect to the layers inputs
            shape(n_samples, n_inputs)
        """

        # order is used to keep same shape as weights
        self.dweights = np.dot(self.inputs.T, dvalues)

        # dvalues shape is (n_sample, n_neuron), axis = 0 sums down the rows
        # keep dims makes the final shape (1, n_neuron), shape now matches biases
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True) 

        self.dinputs = np.dot(dvalues, self.weights.T) 

class ReLU():

    """
    Rectified Linear Unit (ReLU) activation function.

    Applies ReLU to each element: output = max(0, input)
    
    Introduces non-linearity into the network
    """

    def forward(self, inputs):
        
        """
        Performs the forward pass.

        Args:
            inputs (np.ndarray): Input data of any shape

        Sets:
            outputs (np.ndarray): Output data after ReLU activation, same shape as input
        """

        self.inputs = inputs 
        self.outputs = np.maximum(0, self.inputs)

    def backward(self, dvalues):
        
        """
        Performs backward pass

        Applies gradient of ReLU: sets the gradient to 0 where the inputs were <= 0

        Args:
            dvalues (np.ndarray): Gradient of loss with respect to this layers outputs
        
        Sets:
            dinputs (np.ndarray): Gradient of loss with respect to this layers inputs

        """

        self.dinputs = dvalues.copy() # copy because modifying dvalues directly
        
        # if neuron was inactive in forward pass, make its gradient 0 on backward pass
        self.dinputs[self.inputs <= 0] = 0 

class Dropout():

    """
    Neuron Dropout creates a mask that zeros random neurons on each pass to prevent overfitting

    Attributes:
        rate (float): Probability of dropping a neuron
    """
    
    def __init__(self, rate):
        self.rate = rate

    def forward(self, inputs, training = True):

        """
        The forward pass of dropout, creates the mask and applies it to the inputs

        Args:
            inputs (np.ndarray): Input data of any shape
            
            Training (Bool): Determines whether its a training loop, else does not modify inputs

        Sets:
            Outputs (np.ndarray): Output data after dropout, same shape as input
            
            mask (np.ndarray): mask of values of 1 / (1 - rate) and 0, same shape as input
                scale with 1 / (1 - rate) to compensate for dropped neurons
        """
        
        self.inputs = inputs
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size = inputs.shape) / (1 - self.rate)
            self.outputs = inputs * self.mask

        else:
            self.mask = 1
            self.outputs = inputs

    def backward(self, dvalues):

        """
        Backward pass of dropout, applies the mask to the dvalues

        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to this layers inputs

        Sets:
            dinputs (np.ndarray): The dvalues scaled by the mask
        """

        self.dinputs = dvalues * self.mask


class Loss():

    """
    Parent loss class.
    """

    def calculate(self, y_pred, y_true):

        """
        Calculates loss by averaging sample loss calculated through the subclass specific method 

        Args:
            y_pred (np.ndarray): Output values from the final layer's neurons
            
            y_true (np.ndarray): Actual values used to calculate loss

        Returns:
            data_loss (float): loss for the whole batch averaged over the sample losses
        """
        
        sample_loss = self.forward(y_pred, y_true) # implemented in the subclass
        data_loss = np.mean(sample_loss) # averages loss over samples and gets a single value
        
        return data_loss

    def forward(self, y_pred, y_true):
        
        """
        Raises an error if called. Method should be defined in the specific subclass.
        """
        
        raise NotImplementedError 

class Loss_MSE(Loss):

    """
    Emphasizes larger errors due to squaring, making the model sensitive to outliers.

    Commonly used for regression tasks. Not recommended for classification.
    """

    def forward(self, y_pred, y_true):

        """
        Performs the forward pass of the loss class

        MSE = mean((y_pred - y_true) ** 2)

        Args:
            y_pred (np.ndarray): Output values from the final layer's neurons
            
            y_true (np.ndarray): Actual values used to calculate loss

        Returns:
            sample_loss (np.ndarray): Array of loss for each sample
            
            shape (n_samples, 1)
        """
        
        # axis = -1 computes average on last axis, so it can work for 2-D or 3-D arrays
        sample_loss = np.mean((y_pred - y_true) ** 2, axis = -1)

        return sample_loss
    
    def backward(self, y_pred, y_true):

        """
        Performs the backward pass of the loss class

        Computes the derivative of the loss with respect to the model outputs
        This forms the starting point for backpropagation.

        Derivative of MSE = mean(2*(y_pred - y_true))

        Args:
            y_pred (np.ndarray): Output values from the final layer's neurons
            
            y_true (np.ndarray): Actual values used to calculate loss

        Sets:
            dinputs (np.ndarray): Derivative of loss with respect to outputs,
            the first dvalues that get calculated for backpropogation
        """
        
        num_samples = len(y_pred) # gets number of rows aka number of samples
        num_outputs = len(y_pred[0]) # gets number of columns in a row aka number of outputs
       
        self.dinputs = 2 * (y_pred - y_true) / num_samples / num_outputs
        
class Optimizer_SGD():

    """
    Stochastic Gradient Descent (SGD) updates the weights and biases of each layer  
    using the gradients computed during backpropagation. Can utilize momentum, 
    allowing the previous gradient to influence the current gradient. 

    Args:
        learning_rate (float): controls how much each derivative contributes to changing
        the weights and biases. Defaults to 0.01 if no argument is supplied
        
        momentum (float): Controls how the extent of the effect that the previous 
        derivatives have on the new derivatives

    
    Attributes:
        learning_rate (float): Learning rate stored for use in update method 
        momentum (float): Degree that the previous weights effect the calculation of new weights
    """

    def __init__(self, learning_rate = 0.01, momentum = 0.):
        
        """
        Initializes the learning rate and momentum 
        """
        
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_params(self, layer): # updates values for a specific dense layer
       
       """
       Updates the weights and biases of the dense layer.

       Performs parameter update using:
         parameter = parameter - learning_rate * gradient
         This formula applies to both weights and biases.

       If there is momentum, make sure the layer has the weight_momentum
         and bias_momentum attributes else create the attribute and initialize it to 0 

       layer.weight_momentums = self.momentum * layer.weight_momentums - self.learning rate * layer.dweights

       Args:
           layer ('Dense_Layer'): The layer that is being updated 
       """

       if self.momentum:
           if not hasattr(layer, "weight_momentums"): # momentum is part of optimizer. Create it in the specific optimizer
               layer.weight_momentums = np.zeros_like(layer.weights)
               layer.bias_momentums = np.zeros_like(layer.biases)

           weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.learning_rate * layer.dweights
           
           layer.weight_momentums = weight_updates

           bias_updates = \
                self.momentum * layer.bias_momentums - \
                layer.dbiases * self.learning_rate
           
           layer.bias_momentums = bias_updates

       else:
           weight_updates = -self.learning_rate * layer.dweights
           bias_updates = -self.learning_rate * layer.dbiases


       layer.weights += weight_updates
       layer.biases += bias_updates

    def post_update_params(self):
       
        """
        Method utilized in other optimizers. Not needed in this optimizer, 
        pass is used to keep code modular, so the optimizer can be swapped as needed
        without error.
        """
        
        pass 

class Optimizer_Adam():

    """
    Adam updates weights using a combination of momentum (via beta_1)
    and second-order moment estimation (via beta_2). This allows it to 
    smooth out gradients across iterations (momentum) and scale updates 
    adaptively based on the magnitude of past gradients.

    Args:
        learning rate (flaot): Controls how much each derivative contributes to changing
        the weights and biases. Defaults to 0.001 if no argument is supplied

        epsilon (float): Used in updating weights and biases to prevent division by 0. Defaults to 1e-7.
        
        beta_1 (float): Controls the extent of the effect that the previous 
        derivatives have on the new derivatives. Defaults to 0.9

        beta_2 (float): Controls how much memory the optimizer keeps of past squared gradients. Defaults to 0.999
    """

    def __init__(self, learning_rate = 0.001, epsilon = 1e-7, beta_1 = 0.9, beta_2= 0.999):
        
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0

    def update_params(self, layer):

        """
        Initializes momentum and cache terms on the first update.
        Then performs adaptive updates using corrected first and 
        second moment estimates.

        Args:
            layer (Layer_Dense): Specific layer that needs to be updated
        """
        
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * \
                                    layer.weight_momentums + \
                                    (1 - self.beta_1) * layer.dweights
        
        layer.bias_momentums = self.beta_1 * \
                                    layer.bias_momentums + \
                                    (1 - self.beta_1) * layer.dbiases
        
        # corrects for early bias since initialized to 0 
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.learning_rate * weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        
        layer.biases += -self.learning_rate * bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    def post_update_params(self):
        
        """
        Update the iteration number at the end of the pass.

        Sets:
            self.iterations += 1 
        """
        
        self.iterations += 1

class Model():

    """
    Modular model class for regression tasks that creates layers and specified activation functions
    based on the input layer sizes. 

    Atributes:
        layer_size (list): list of layer sizes. First number must be the number of input features

        dropout_rate (float): rate at which neurons are dropped for a pass. Helps generalization.
            Defaults to 0. 
    """
    
    def __init__(self, layer_size : list, dropout_rate = 0.):
        self.layers_size = layer_size
        self.dropout_rate = dropout_rate
        self.layers = []

    def set(self, activation : object , loss_function : object, optimizer : object):

        """
        initializes the activation function, loss function, and optimizer.

        Attributes:
            activation (ReLU): Can be any activation function but ReLU is the only one defined
            
            loss_function (loss_function)L can be any loss function but MSE is the only one defined
            
            optimizer (obj): can be any optimizer so long as it has a update_params and post_update_params.
                post_update_params can be trivial, using pass, so long as the method can be called.
        """

        self.activation = activation
        self.loss_function = loss_function
        self.optimizer = optimizer

    def build(self):

        """
        Builds the Dense_Layers, ReLU activation layers, and dropout layers in accordance with layer size.
        Dropout defaults to 0 so if its not wanted it serves as a trivial layer.

        Args:
            None
        """

        for i in range(len(self.layers_size) - 1):

            self.layers.append(Layer_Dense(self.layers_size[i], self.layers_size[i + 1]))

            if i < len(self.layers_size) - 2:
                self.layers.append(ReLU())
                self.layers.append(Dropout(rate=self.dropout_rate))

    def forward(self, inputs, training = False):

        """
        Loops through the layers and performs the forward pass. 

        Args:
            inputs (np.ndarray): Input data of shape (n_samples, n_inputs)

            training (boolean): Training determines if dropout is used in the forward pass. Defaults to False
        """
        
        output = inputs
    
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.forward(output, training)
            else:
                layer.forward(output)
            
            output =  layer.outputs
    
    def calc_loss(self, predictions, y_true):
       
        """
        Calculates loss using the loss_function's forward method and calculates the dinputs using 
        the loss_function's backward method.

        Args:
            y_pred (np.ndarray): Output values from the final layer's neurons
            
            y_true (np.ndarray): Actual values used to calculate loss
        """
       
        self.data_loss = self.loss_function.forward(predictions, y_true)
        self.loss_function.backward(predictions, y_true)

    def backward(self):

        """
        Loops through the layers and performs their backward method
        """

        dinputs = self.loss_function.dinputs

        for layer in reversed(self.layers):
            layer.backward(dinputs)
            dinputs = layer.dinputs

    def update(self):

        """
        loops through the layers and updates them based on the method defined 
        in the optimizer
        """

        for layer in self.layers:
            if hasattr(layer, "weights"):
                self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

    def train(self, X, y, epochs = 2500, batch_size = 64):

        """
        Training loop that uses the forward, loss, backward, and update methods. 
        Every epoch batches are randomized for better generalization

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_inputs)

            y (np.ndarray): Actual values used to calculate loss

            epochs (int): number of training iterations. Defaults to 2500.

            batch_size (int): subset of sample size. Amount of samples used in a single forward and backward pass.
                Defaults to 64.
                

        """
        self.epochs = epochs
        num_samples = len(X)
        num_batches = (num_samples + batch_size -1) // batch_size

        # main loop

        self.epoch_loss_list = []
        for epoch in range(epochs + 1):

            permutation = np.random.permutation(len(X)) # randomize batch for each epoch
            X_epoch = X[permutation]                          # for better generalization
            y_epoch = y[permutation]

            batch_loss = 0

            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_samples) # safety net in case of exceeding sample size

                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end].reshape(-1, 1)

                self.forward(inputs=X_batch, training=True)
                self.calc_loss(self.layers[-1].outputs, y_batch)
                batch_loss += np.mean(self.data_loss)
                self.backward()

                self.update()
           
            epoch_loss = batch_loss / num_batches
            self.epoch_loss_list.append(epoch_loss) 
            if epoch % 100 == 0:
                print(f"epoch {epoch} | loss {epoch_loss:.4f}")

    def validate(self, X, y):

        """
        Uses a single forward pass and loss call to make predictions with a trained model.
        Prints the vall loss

        X (np.ndarray): Input data of shape (n_samples, n_inputs)

        y (np.ndarray): Actual values used to calculate loss
        """
        
        self.forward(inputs=X, training=False)
        self.calc_loss(self.layers[-1].outputs, y.reshape(-1, 1))

        val_loss = np.mean(self.data_loss)
        print(f"validation loss {val_loss:.4f}")

    def test(self, X, y):
        
        """
        Uses a single forward pass and loss call to make predictions with a trained model.
        Prints the test loss using Root Mean Squared Error. Loss units will be in the 
        original size.

        X (np.ndarray): Input data of shape (n_samples, n_inputs)

        y (np.ndarray): Actual values used to calculate loss
        """
        
        self.forward(inputs=X, training=False)
        self.calc_loss(self.layers[-1].outputs, y.reshape(-1, 1))

        test_loss = np.mean(self.data_loss)
        print(f"test loss {np.sqrt(test_loss):.4f}")
    

    def plot_loss(self):

        """
        plot method that takes epoch loss found in the training method and plots it against epochs
        """
       
        plt.figure(1)
        plt.plot(range(self.epochs + 1), self.epoch_loss_list, label = "Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def save(self, path):

        """
        save method, saves the fully trained model using pickle.

        Args:
            path (string): relative path to the directory that the model will be saved in
        """

        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        """
        load method, loads the fully trained model using pickle.

        Args:
            path (string): relative path to the directory that the model will be loaded from 
        """
        with open(path, "rb") as f:
            return pickle.load(f)

       
        

       
           



                    
                
        

             


        




