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
import neoml.Blob as Blob


class ObjectNormalization(Layer):
    """The layer that performs object normalization using the formula:
    objectNorm(x)[i][j] = ((x[i][j] - mean[i]) / sqrt(var[i] + epsilon)) *
        * scale[j] + bias[j]
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param epsilon: The small value added to the variance to avoid division by zero. 
    :type epsilon: float, default=0.00001
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a set of objects.
        The dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** is the number of objects
        - **Height** * **Width** * **Depth** * **Channels** is the object size
    
    .. rubric:: Layer outputs:

    (1) the normalized result, of the same size as the input.
    """

    def __init__(self, input_layer, epsilon=0.00001, name=None):

        if type(input_layer) is PythonWrapper.ObjectNormalization:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if float(epsilon) <= 0:
            raise ValueError('The `epsilon` must be > 0.')

        internal = PythonWrapper.ObjectNormalization(str(name), layers[0], int(outputs[0]), float(epsilon))
        super().__init__(internal)

    @property
    def epsilon(self):
        """Gets the small value used to avoid division by zero.
        """
        return self._internal.get_epsilon()

    @epsilon.setter
    def epsilon(self, epsilon):
        """Sets the small value used to avoid division by zero.
        """
        if float(epsilon) <= 0:
            raise ValueError('The `epsilon` must be > 0.')
        self._internal.set_epsilon(float(epsilon))

    @property
    def scale(self):
        """Gets scale, one of the trainable parameters in the formula.
        The total blob size is equal to the input **Height** * **Width** * **Depth** * **Channels**.
        """
        return Blob.Blob(self._internal.get_scale())

    @scale.setter
    def scale(self, scale):
        """Sets scale, one of the trainable parameters in the formula.
        The total blob size is equal to the input **Height** * **Width** * **Depth** * **Channels**.
        """
        self._internal.set_scale(scale._internal)

    @property
    def bias(self):
        """Gets bias, one of the trainable parameters in the formula.
        The total blob size is equal to the input **Height** * **Width** * **Depth** * **Channels**.
        """
        return Blob.Blob(self._internal.get_bias())

    @bias.setter
    def bias(self, bias):
        """Sets bias, one of the trainable parameters in the formula.
        The total blob size is equal to the input **Height** * **Width** * **Depth** * **Channels**.
        """
        self._internal.set_bias(bias._internal)
