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
--------------------------------------------------------------------------------------------------------------*/
"""

import neoml.PythonWrapper as PythonWrapper
from .Dnn import Layer
from .Utils import check_input_layers


class Linear(Layer):
    """The layer that calculates a linear activation function
    for each element of a single input:
    f(x) = multiplier * x + free_term

    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    multiplier : float
        The linear function multiplier.
    free_term : float
        The linear function free term.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, multiplier, free_term, name=None):

        if type(input_layer) is PythonWrapper.LinearLayer:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.LinearLayer(str(name), layers[0], int(outputs[0]), float(multiplier), float(free_term))
        super().__init__(internal)

    @property
    def multiplier(self):
        """Gets the multiplier.
        """
        return self._internal.get_multiplier()

    @multiplier.setter
    def multiplier(self, multiplier):
        """Sets the multiplier.
        """

        self._internal.set_multiplier(multiplier)

    @property
    def free_term(self):
        """Gets the free term.
        """
        return self._internal.get_free_term()

    @free_term.setter
    def free_term(self, free_term):
        """Sets the free term.
        """

        self._internal.set_free_term(free_term)

# ----------------------------------------------------------------------------------------------------------------------


class ELU(Layer):
    """The layer that calculates the ELU activation function
    for each element of the single input:
    f(x) = alpha * (exp(x) - 1)    if x < 0
    f(x) = x                       if x >= 0
    
    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    alpha : float
        The multiplier before the exponential function used
        for negative values of x.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, alpha, name=None):

        if type(input_layer) is PythonWrapper.ELU:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.ELU(str(name), layers[0], int(outputs[0]), float(alpha))
        super().__init__(internal)

    @property
    def alpha(self):
        """Gets the exponent multiplier.
        """
        return self._internal.get_alpha()

    @alpha.setter
    def alpha(self, alpha):
        """Sets the exponent multiplier.
        """

        self._internal.set_alpha(alpha)

# ----------------------------------------------------------------------------------------------------------------------


class ReLU(Layer):
    """The layer that calculates the ReLU activation function
    for each element of the single input:
    f(x) = 0    if x <= 0
    f(x) = x    if x > 0

    You also can set the cutoff upper threshold:
    f(x) = 0            if x <= 0
    f(x) = x            if 0 < x < threshold
    f(x) = threshold    if threshold <= x
    
    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements

    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    threshold : float, default=0
        The upper cutoff threshold. 0 resets the threshold.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, threshold=0.0, name=None):

        if type(input_layer) is PythonWrapper.ReLU:
            super().__init__(input_layer)
            return  

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.ReLU(str(name), layers[0], int(outputs[0]), float(threshold))
        super().__init__(internal)

    @property
    def threshold(self):
        """Gets the upper cutoff threshold.
        """
        return self._internal.get_threshold()

    @threshold.setter
    def threshold(self, threshold):
        """Sets the upper cutoff threshold.
        """
        self._internal.set_threshold(threshold)

# ----------------------------------------------------------------------------------------------------------------------


class LeakyReLU(Layer):
    """The layer that calculates the "leaky" ReLU activation function
    for each element of the single input:
    f(x) = alpha * x    if x <= 0
    f(x) = x            if x > 0

    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    alpha : float, default=0
        The multiplier used for negative values of x.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, alpha, name=None):

        if type(input_layer) is PythonWrapper.LeakyReLU:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.LeakyReLU(str(name), layers[0], int(outputs[0]), float(alpha))
        super().__init__(internal)

    @property
    def alpha(self):
        """Gets the multiplier used for negative values of x.
        """
        return self._internal.get_alpha()

    @alpha.setter
    def alpha(self, alpha):
        """Sets the multiplier used for negative values of x.
        """

        self._internal.set_alpha(alpha)

# ----------------------------------------------------------------------------------------------------------------------


class HSwish(Layer):
    """The layer that calculates the H-Swish activation function
    for each element of the single input:
    f(x) = 0                    if x <= -3
    f(x) = x * (x + 3) / 6      if -3 < x < 3
    f(x) = x                    if x >= 3
    
    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.HSwish:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.HSwish(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class Abs(Layer):
    """The layer that calculates the absolute value
    of each element of the single input.
    
    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.Abs:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.Abs(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class Sigmoid(Layer):
    """The layer that calculates the sigmoid activation function
    for each element of the signle input:
    f(x) = 1 / (1 + exp(-x))
    
    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.Sigmoid:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.Sigmoid(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class Tanh(Layer):
    """The layer that calculates the tanh activation function
    for each element of the single input:
    f(x) = (exp(2 * x) - 1) / (exp(2 * x) + 1)
    
    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.Tanh:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.Tanh(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class HardTanh(Layer):
    """The layer that calculates the HardTanh activation function
    for each element of the single input:
    f(x) = -1    if x <= -1
    f(x) = x     if -1 < x < 1
    f(x) = 1     if x >= 1
    
    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.HardTanh:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.HardTanh(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class HardSigmoid(Layer):
    """The layer that calculates the "hard sigmoid" activation function
    for each element of the single input:
    f(x) = 0                    if x <= -bias / slope
    f(x) = slope * x + bias     if -bias / slope < x < (1 - bias) / slope
    f(x) = 1                    if x >= (1 - bias) / slope
    
    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    slope : float, > 0
        The slope of the linear component.
    bias : float
        The shift of the linear component from 0.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, slope, bias, name=None):

        if type(input_layer) is PythonWrapper.HardSigmoid:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.HardSigmoid(str(name), layers[0], int(outputs[0]), float(slope), float(bias))
        super().__init__(internal)

    @property
    def slope(self):
        """Gets the slope of the linear component.
        """
        return self._internal.get_slope()

    @slope.setter
    def slope(self, slope):
        """Sets the slope of the linear component.
        """

        self._internal.set_slope(float(slope))

    @property
    def bias(self):
        """Gets the shift of the linear component from 0.
        """
        return self._internal.get_bias()

    @bias.setter
    def bias(self, bias):
        """Sets the shift of the linear component from 0.
        """

        self._internal.set_bias(float(bias))

# ----------------------------------------------------------------------------------------------------------------------


class Power(Layer):
    """The layer that raises each element of the input to the given power.
    
    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    exponent : float
        The power to which the input elements will be raised.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, exponent, name=None):

        if type(input_layer) is PythonWrapper.Power:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.Power(str(name), layers[0], int(outputs[0]), float(exponent))
        super().__init__(internal)

    @property
    def exponent(self):
        """Gets the power to which the input elements will be raised.
        """
        return self._internal.get_exponent()

    @exponent.setter
    def exponent(self, exponent):
        """Sets the power to which the input elements will be raised.
        """
        self._internal.set_exponent(float(exponent))

# ----------------------------------------------------------------------------------------------------------------------


class GELU(Layer):
    """The layer that calculates the GELU activation function
    for each element of the signle input:
    f(x) = x / (1 + exp(-1.702 * x))
    
    Layer inputs
    ----------
    #1: a data blob of any size
    
    Layer outputs
    ----------
    #1: a data blob of the same size as the input,
    with activation function values on each of the input elements
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.GELU:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.GELU(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)
