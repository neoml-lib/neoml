""" Copyright (c) 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------
"""

import neoml.PythonWrapper as PythonWrapper
from .Dnn import Layer
from .Utils import check_input_layers


class Loss(Layer):
    """The base class for layers estimating error.
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.Loss):
            raise ValueError('The `internal` must be PythonWrapper.Loss')

        super().__init__(internal)

    @property
    def last_loss(self):
        """Gets the value of the loss function on the last step.
        """
        return self._internal.get_last_loss()
        
    @property
    def loss_weight(self):
        """Gets the multiplier for the loss function during training.
        """
        return self._internal.get_loss_weight()

    @loss_weight.setter
    def loss_weight(self, weight):
        """Sets the multiplier for the loss function during training.
        """
        self._internal.set_loss_weight(weight)

    @property
    def train_labels(self):
        """Checks if gradients should also be calculated for the second input,
        which contains the class labels.
        """
        return self._internal.get_train_labels()

    @train_labels.setter
    def train_labels(self, train):
        """Specifies if gradients should also be calculated for the second input,
        which contains the class labels.
        """
        self._internal.set_train_labels(train)

    @property
    def max_gradient(self):
        """Gets the upper limit for the absolute value of the function gradient.
        """
        return self._internal.get_max_gradient()

    @max_gradient.setter
    def max_gradient(self, max_value):
        """Sets the upper limit for the absolute value of the function gradient.
        """
        if max_value <= 0 :
            raise ValueError('The `max_value` must be > 0.')

        self._internal.set_max_gradient(max_value)

# ----------------------------------------------------------------------------------------------------------------------


class CrossEntropyLoss(Loss):
    """The layer that calculates the loss value as cross-entropy 
    between the result and the standard:
    loss = -sum(y_i * log(z_i)),
    where for each i class 
        y_i represents the class label, 
        z_i is the network response (with softmax applied or not)

    Layer inputs
    ------------
    #1: the network response for which you are calculating the loss.
    It should contain the probability distribution for objects over classes.
    If you are not going to apply softmax in this layer, each element should already be >= 0, 
    and the sum over Height * Width * Depth * Channels dimension should be equal to 1.
    The dimensions:
    - BatchLength * BatchWidth * ListSize - the number of objects
    - Height * Width * Depth * Channels - the number of classes
    
    #2: the correct class labels. Two formats are acceptable:
    - The blob contains float data, the dimensions are equal to the first input dimensions. 
      It should be filled with zeros, and only the coordinate of the class to which 
      the corresponding object from the first input belongs should be 1.
    - The blob contains int data with BatchLength, BatchWidth, and ListSize
      equal to these dimensions of the first input, and the other dimensions equal to 1.
      Each object in the blob contains the number of the class 
      to which the corresponding object from the first input belongs.
    
    #3 (optional): the objects' weights.
    The dimensions:
    - BatchLength, BatchWidth, ListSize should be the same as for the first input
    - the other dimensions should be 1

    Layer outputs
    -------------
    The layer has no output.
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    softmax : bool, default=True
        Specifies if softmax function should be applied to the result.
    loss_weight : float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, softmax=True, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.CrossEntropyLoss:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.CrossEntropyLoss(str(name), layers, outputs, bool(softmax), float(loss_weight))
        super().__init__(internal)

    @property
    def apply_softmax(self):
        """Checks if softmax function should be applied to the result.
        """
        return self._internal.get_apply_softmax()

    @apply_softmax.setter
    def apply_softmax(self, value):
        """Specifies if softmax function should be applied to the result.
        """
        self._internal.set_apply_softmax(int(value))

# ----------------------------------------------------------------------------------------------------------------------


class BinaryCrossEntropyLoss(Loss):
    """The layer that calculates the cross-entropy loss function
    for binary classification:
    loss = y * -log(sigmoid(x)) + (1 - y) * -log(1 - sigmoid(x)), where
        x is the network response,
        y is the correct class label (can be -1 or 1)

    Layer inputs
    ------------
    #1: the network response for which you are calculating the loss.
    It should contain the probability distribution for objects over classes.
    
    #2: the correct class labels (1 or -1). 
    
    #3 (optional): the objects' weights.
    
    The dimensions of all inputs are the same:
    - BatchLength * BatchWidth * ListSize - the number of objects
    - Height, Width, Depth, Channels should be equal to 1

    Layer outputs
    -------------
    The layer has no output.
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    positive_weight : float, default=1.0
        The multiplier for correctly classified objects. 
        Tune this value to prioritize precision (positive_weight < 1) 
        or recall (positive_weight > 1) during training.
    loss_weight : float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, positive_weight=1.0, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.BinaryCrossEntropyLoss:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.BinaryCrossEntropyLoss(str(name), layers, outputs, float(positive_weight), float(loss_weight))
        super().__init__(internal)

    @property
    def positive_weight(self):
        """Gets the multiplier for the term that corresponds 
        to the correct results.
        """
        return self._internal.get_positive_weight()

    @positive_weight.setter
    def positive_weight(self, weight):
        """Sets the multiplier for the term that corresponds
        to the correct results. Tune this value to prioritize 
        precision (positive_weight < 1) or accuracy (positive_weight > 1).
        """
        self._internal.set_positive_weight(float(weight))

# ----------------------------------------------------------------------------------------------------------------------


class EuclideanLoss(Loss):
    """The layer that calculates the loss function equal to Euclidean distance
    between the network response and the correct classes.
    
    Layer inputs
    ------------
    #1: the network response for which you are calculating the loss.
    
    #2: the correct class objects. 
    
    #3 (optional): the objects' weights.
    
    The dimensions of all inputs are the same:
    - BatchLength * BatchWidth * ListSize - the number of objects
    - Height * Width * Depth * Channels - the object size

    Layer outputs
    -------------
    The layer has no output.
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    loss_weight : float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.EuclideanLoss:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.EuclideanLoss(str(name), layers, outputs, float(loss_weight))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class HingeLoss(Loss):
    """The layer that calculates hinge loss function for binary classification:
    f(x) = max (0, 1 - x * y), where 
    x is the network response, 
    y is the correct class label (1 or -1).
    
    Layer inputs
    ------------
    #1: the network response for which you are calculating the loss.
    It should contain the probability distribution for objects over classes.
    
    #2: the correct class labels (1 or -1). 
    
    #3 (optional): the objects' weights.
    
    The dimensions of all inputs are the same:
    - BatchLength * BatchWidth * ListSize - the number of objects
    - Height, Width, Depth, Channels should be equal to 1

    Layer outputs
    -------------
    The layer has no output.
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    loss_weight : float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.HingeLoss:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.HingeLoss(str(name), layers, outputs, float(loss_weight))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class SquaredHingeLoss(Loss):
    """The layer that calculates squared hinge loss function
    for binary classification:
    f(x) = -4 * x * y               if x * y < -1
    f(x) = sqr(max(0, 1 - x * y))   if x * y >= -1
    where:
    x is the network response, 
    y is the correct class label (1 or -1).
    
    Layer inputs
    ------------
    #1: the network response for which you are calculating the loss.
    It should contain the probability distribution for objects over classes.
    
    #2: the correct class labels (1 or -1). 
    
    #3 (optional): the objects' weights.
    
    The dimensions of all inputs are the same:
    - BatchLength * BatchWidth * ListSize - the number of objects
    - Height, Width, Depth, Channels should be equal to 1

    Layer outputs
    -------------
    The layer has no output.
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    loss_weight : float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.SquaredHingeLoss:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.SquaredHingeLoss(str(name), layers, outputs, float(loss_weight))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class FocalLoss(Loss):
    """The layer that calculates the focal loss function for multiple class
    classification. It is a version of cross-entropy in which 
    the easily-distinguished objects receive smaller penalties. This helps
    focus on learning the difference between similar-looking elements of
    different classes.
    
    f(x) = -pow(1 - x_right, force) * log(x_right)
    where x_right is the network response element that represents 
    the probability for the object to belong to the correct class.
    
    Layer inputs
    ------------
    #1: the network response for which you are calculating the loss.
    It should contain the probability distribution for objects over classes.
    If you are not going to apply softmax in this layer, each element should already be >= 0, 
    and the sum over Height * Width * Depth * Channels dimension should be equal to 1.
    The dimensions:
    - BatchLength * BatchWidth * ListSize - the number of objects
    - Height * Width * Depth * Channels - the number of classes
    
    #2: the correct class labels. Two formats are acceptable:
    - The blob contains float data, the dimensions are equal to the first input dimensions. 
      It should be filled with zeros, and only the coordinate of the class to which 
      the corresponding object from the first input belongs should be 1.
    - The blob contains int data with BatchLength, BatchWidth, and ListSize
      equal to these dimensions of the first input, and the other dimensions equal to 1.
      Each object in the blob contains the number of the class 
      to which the corresponding object from the first input belongs.
    
    #3 (optional): the objects' weights.
    The dimensions:
    - BatchLength, BatchWidth, ListSize should be the same as for the first input
    - the other dimensions should be 1

    Layer outputs
    -------------
    The layer has no output.
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    force : float, > 0, default=2.0
        The focal force, that is, the degree to which learning
        will concentrate on similar objects.
    loss_weight : float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, force, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.FocalLoss:
            super().__init__(input_layers)
            return

        if force <= 0 :
            raise ValueError('The `force` must be > 0.')

        layers, outputs = check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.FocalLoss(str(name), layers, outputs, float(force), float(loss_weight))
        super().__init__(internal)

    @property
    def force(self):
        """Gets the focal force multiplier.
        """
        return self._internal.get_force()

    @force.setter
    def force(self, force):
        """Sets the focal force multiplier.
        """
        if force <= 0 :
            raise ValueError('The `force` must be > 0.')

        self._internal.set_force(float(force))

# ----------------------------------------------------------------------------------------------------------------------


class BinaryFocalLoss(Loss):
    """The layer that calculates the focal loss function for binary
    classification. It is a version of cross-entropy in which 
    the easily-distinguished objects receive smaller penalties. This helps
    focus on learning the difference between similar-looking elements of
    different classes.
    
    f(x) = -pow(sigmoid(-y * x), force) * log(1 + exp(-y * x))
    where:
    x is the network response, 
    y is the correct class label (1 or -1).
    
    Layer inputs
    ------------
    #1: the network response for which you are calculating the loss.
    It should contain the probability distribution for objects over classes.
    
    #2: the correct class labels (1 or -1). 
    
    #3 (optional): the objects' weights.
    
    The dimensions of all inputs are the same:
    - BatchLength * BatchWidth * ListSize - the number of objects
    - Height, Width, Depth, Channels should be equal to 1

    Layer outputs
    -------------
    The layer has no output.
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    force : float, > 0, default=2.0
        The focal force, that is, the degree to which learning
        will concentrate on similar objects.
    loss_weight : float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, force, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.BinaryFocalLoss:
            super().__init__(input_layers)
            return

        if force <= 0 :
            raise ValueError('The `force` must be > 0.')

        layers, outputs = check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.BinaryFocalLoss(str(name), layers, outputs, float(force), float(loss_weight))
        super().__init__(internal)

    @property
    def force(self):
        """Gets the focal force multiplier.
        """
        return self._internal.get_force()

    @force.setter
    def force(self, force):
        """Sets the focal force multiplier.
        """
        if force <= 0 :
            raise ValueError('The `force` must be > 0.')


        self._internal.set_force(float(force))

# ----------------------------------------------------------------------------------------------------------------------


class CenterLoss(Loss):
    """The layer that penalizes large differences between objects 
    of the same class. 
    See the paper at http://ydwen.github.io/papers/WenECCV16.pdf
    
    Layer inputs
    ------------
    #1: the network response for which you are calculating the loss.
    It should contain the probability distribution for objects over classes.
    If you are not going to apply softmax in this layer, each element should already be >= 0, 
    and the sum over Height * Width * Depth * Channels dimension should be equal to 1.
    The dimensions:
    - BatchLength * BatchWidth * ListSize - the number of objects
    - Height * Width * Depth * Channels - the number of classes
    
    #2: the correct class labels. Two formats are acceptable:
    - The blob contains float data, the dimensions are equal to the first input dimensions. 
      It should be filled with zeros, and only the coordinate of the class to which 
      the corresponding object from the first input belongs should be 1.
    - The blob contains int data with BatchLength, BatchWidth, and ListSize
      equal to these dimensions of the first input, and the other dimensions equal to 1.
      Each object in the blob contains the number of the class 
      to which the corresponding object from the first input belongs.
    
    #3 (optional): the objects' weights.
    The dimensions:
    - BatchLength, BatchWidth, ListSize should be the same as for the first input
    - the other dimensions should be 1

    Layer outputs
    -------------
    The layer has no output.
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    class_count : integer
        The number of classes in the model.
    rate : float, [0..1]
        Class center convergence rate: the multiplier used 
        for calculating the moving mean of the class centers 
        for each subsequent iteration.
    loss_weight : float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, class_count, rate, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.CenterLoss:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.CenterLoss(str(name), layers, outputs, int(class_count), float(rate), float(loss_weight))
        super().__init__(internal)

    @property
    def class_count(self):
        """Gets the number of classes in the model.
        """
        return self._internal.get_class_count()

    @class_count.setter
    def class_count(self, class_count):
        """Sets the number of classes in the model.
        """
        self._internal.set_class_count(int(class_count))

    @property
    def rate(self):
        """Gets the convergence rate multiplier.
        """
        return self._internal.get_rate()

    @rate.setter
    def rate(self, rate):
        """Sets the convergence rate multiplier.
        """
        self._internal.set_rate(float(rate))

# ----------------------------------------------------------------------------------------------------------------------


class MultiHingeLoss(Loss):
    """The layer that calculates hinge loss function for multiple class
    classification:
    f(x) = max(0, 1 - (x_right - x_max_wrong))
    where 
    x_right is the network response for the correct class,
    x_max_wrong is the largest response for all the incorrect classes.
    
    Layer inputs
    ------------
    #1: the network response for which you are calculating the loss.
    It should contain the probability distribution for objects over classes.
    If you are not going to apply softmax in this layer, each element should already be >= 0, 
    and the sum over Height * Width * Depth * Channels dimension should be equal to 1.
    The dimensions:
    - BatchLength * BatchWidth * ListSize - the number of objects
    - Height * Width * Depth * Channels - the number of classes
    
    #2: the correct class labels. Two formats are acceptable:
    - The blob contains float data, the dimensions are equal to the first input dimensions. 
      It should be filled with zeros, and only the coordinate of the class to which 
      the corresponding object from the first input belongs should be 1.
    - The blob contains int data with BatchLength, BatchWidth, and ListSize
      equal to these dimensions of the first input, and the other dimensions equal to 1.
      Each object in the blob contains the number of the class 
      to which the corresponding object from the first input belongs.
    
    #3 (optional): the objects' weights.
    The dimensions:
    - BatchLength, BatchWidth, ListSize should be the same as for the first input
    - the other dimensions should be 1

    Layer outputs
    -------------
    The layer has no output.
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    loss_weight :  float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.MultiHingeLoss:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.MultiHingeLoss(str(name), layers, outputs, float(loss_weight))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class MultiSquaredHingeLoss(Loss):
    """The layer that calculates squared hinge loss function for multiple class
    classification:
    f(x) = -4 * (x_right - x_max_wrong)             if x_right - x_max_wrong < -1
    f(x) = sqr(max(0, 1 - (x_right - x_max_wrong))) if x_right - x_max_wrong >= -1
    where 
    x_right is the network response for the correct class,
    x_max_wrong is the largest response for all the incorrect classes.
    
    Layer inputs
    ------------
    #1: the network response for which you are calculating the loss.
    It should contain the probability distribution for objects over classes.
    If you are not going to apply softmax in this layer, each element should already be >= 0, 
    and the sum over Height * Width * Depth * Channels dimension should be equal to 1.
    The dimensions:
    - BatchLength * BatchWidth * ListSize - the number of objects
    - Height * Width * Depth * Channels - the number of classes
    
    #2: the correct class labels. Two formats are acceptable:
    - The blob contains float data, the dimensions are equal to the first input dimensions. 
      It should be filled with zeros, and only the coordinate of the class to which 
      the corresponding object from the first input belongs should be 1.
    - The blob contains int data with BatchLength, BatchWidth, and ListSize
      equal to these dimensions of the first input, and the other dimensions equal to 1.
      Each object in the blob contains the number of the class 
      to which the corresponding object from the first input belongs.
    
    #3 (optional): the objects' weights.
    The dimensions:
    - BatchLength, BatchWidth, ListSize should be the same as for the first input
    - the other dimensions should be 1

    Layer outputs
    -------------
    The layer has no output.
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    loss_weight :  float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.MultiSquaredHingeLoss:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.MultiSquaredHingeLoss(str(name), layers, outputs, float(loss_weight))
        super().__init__(internal)
