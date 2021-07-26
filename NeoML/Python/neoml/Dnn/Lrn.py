""" Copyright (c) 2017-2021 ABBYY Production LLC

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

class Lrn(Layer):
    """Lrn layer performs local response normlization with the following formula:
    :math:`LRN(x)[obj][ch] = x[obj][ch] * / ((bias + alpha * sqrSum[obj][ch] / windowSize) ^ beta)`
    where :math:`obj` is index of the object , :math:`ch` is index of the channel,
    :math:`window_size`, :math:`bias`, :math:`alpha` and :math:`beta` are layer settings
    and :math:`sqrSum[obj][ch] = \sum_{i=\max(0, ch - \lfloor\frac{windowSize - 1}{2}\rfloor)}^{\min(C - 1, ch + \lceil\frac{windowSize - 1}{2}\rceil)}x[obj][i]^2`

    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param window_size: The size of window used in normalization
    :type: int, default=1
    :param bias: value added to the scaled sum of squares 
    :type: float, default=1.
    :param alpha: scale value of sum of squares
    :type: float, default=1e-4
    :param beta: exponent of the formula
    :type: float, default=0.75

    .. rubric:: Layer inputs:

    (1) the set of objects, of dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** * **Height** * **Width** * **Depth** - the number of objects
        - **Channels** - the size of the object

    .. rubric:: Layer outputs:

    (1) the result of the layer, of the dimensions of the input.
    """

    def __init__(self, input_layer, window_size=1, bias=1., alpha=1e-4, beta=0.75, name=None):
        if type(input_layer) is PythonWrapper.Lrn:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if window_size <= 0:
            raise ValueError('The `window_size` must be > 0.')

        internal = PythonWrapper.Lrn(str(name), layers[0], int(outputs[0]), int(window_size),
            float(bias), float(alpha), float(beta))
        super().__init__(internal)

    @property
    def window_size(self):
        """Gets the window size.
        """
        return self._internal.get_window_size()

    @window_size.setter
    def window_size(self, window_size):
        """Sets the window size.
        """
        self._internal.set_window_size(int(window_size))

    @property
    def bias(self):
        """Gets the bias.
        """
        return self._internal.get_bias()

    @bias.setter
    def bias(self, bias):
        """Sets the bias.
        """
        self._internal.set_bias(bias)

    @property
    def alpha(self):
        """Gets the alpha.
        """
        return self._internal.get_alpha()

    @alpha.setter
    def alpha(self, alpha):
        """Sets the alpha.
        """
        self._internal.set_alpha(alpha)

    @property
    def beta(self):
        """Gets the beta.
        """
        return self._internal.get_beta()

    @beta.setter
    def beta(self, beta):
        """Sets the beta.
        """
        self._internal.set_beta(beta)
