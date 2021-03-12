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
import neoml.Blob as Blob


class ObjectNormalization(Layer):
    """
    
    Parameters
    ----------
    input_layers :
    count :
    size :
    name : str, default=None
        The layer name.
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
        """
        """
        return self._internal.get_epsilon()

    @epsilon.setter
    def epsilon(self, epsilon):
        """
        """
        if float(epsilon) <= 0:
            raise ValueError('The `epsilon` must be > 0.')
        self._internal.set_epsilon(float(epsilon))

    @property
    def scale(self):
        """
        """
        return Blob.Blob(self._internal.get_scale())

    @scale.setter
    def scale(self, scale):
        """
        """
        self._internal.set_scale(scale._internal)

    @property
    def bias(self):
        """
        """
        return Blob.Blob(self._internal.get_bias())

    @bias.setter
    def bias(self, bias):
        """
        """
        self._internal.set_bias(bias._internal)
