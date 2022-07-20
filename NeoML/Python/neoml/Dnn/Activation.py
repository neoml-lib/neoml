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
from neoml.Utils import check_input_layers


class Linear(Layer):
    """The layer that calculates a linear activation function
    for each element of a single input:
    :math:`f(x) = multiplier * x + free\_term`

    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param multiplier: The linear function multiplier.
    :type multiplier: float
    :param free_term: The linear function free term.
    :type free_term: float
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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

    - :math:`f(x) = alpha * (e^x - 1)`    if :math:`x < 0`
    - :math:`f(x) = x`                    if :math:`x \ge 0`
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param alpha: The multiplier before the exponential function used
        for negative values of x.    
    :type alpha: float
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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

    - :math:`f(x) = 0`    if :math:`x \le 0`
    - :math:`f(x) = x`    if :math:`x > 0`

    You also can set the cutoff upper threshold:

    - :math:`f(x) = 0`            if :math:`x \le 0`
    - :math:`f(x) = x`            if :math:`0 < x < threshold`
    - :math:`f(x) = threshold`    if :math:`threshold \le x`
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param threshold: The upper cutoff threshold. 0 resets the threshold.   
    :type threshold: float, default=0
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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

    - :math:`f(x) = alpha * x`    if :math:`x \le 0`
    - :math:`f(x) = x`            if :math:`x > 0`

    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param alpha: The multiplier used for negative values of x.    
    :type alpha: float, default=0
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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

    - :math:`f(x) = 0`                    if :math:`x \le -3`
    - :math:`f(x) = x * (x + 3) / 6`      if :math:`-3 < x < 3`
    - :math:`f(x) = x`                    if :math:`x \ge 3`
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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
    :math:`f(x) = 1 / (1 + e^{-x})`
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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
    :math:`f(x) = (e^{2 * x} - 1) / (e^{2 * x} + 1)`
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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
  
    - :math:`f(x) = -1`    if :math:`x \le -1`
    - :math:`f(x) = x`     if :math:`-1 < x < 1`
    - :math:`f(x) = 1`     if :math:`x \ge 1`
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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

    - :math:`f(x) = 0`                    if :math:`x \le -bias / slope`
    - :math:`f(x) = slope * x + bias`     if :math:`-bias / slope < x < (1 - bias) / slope`
    - :math:`f(x) = 1`                    if :math:`x \ge (1 - bias) / slope`
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param slope: The slope of the linear component.   
    :type slope: float, > 0
    :param bias: The shift of the linear component from 0.
    :type bias: float
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param exponent: The power to which the input elements will be raised.
    :type exponent: float
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

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
    for each element of the single input:
    :math:`f(x) = x / (1 + e^{-1.702 * x})`
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with activation function values on each of the input elements

    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.GELU:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.GELU(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class Exp(Layer):
    """The layer that calculates the exponent function
    for each element of the single input.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with exponent function values on each of the input elements

    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.Exp:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.Exp(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class Log(Layer):
    """The layer that calculates the logarithm function
    for each element of the single input.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with logarithm function values on each of the input elements

    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.Log:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.Log(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class Erf(Layer):
    """The layer that calculates the error function
    for each element of the single input.

    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size
    
    .. rubric:: Layer outputs:

    (1) a data blob of the same size as the input,
        with error function values on each of the input elements

    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.Erf:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.Erf(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)

