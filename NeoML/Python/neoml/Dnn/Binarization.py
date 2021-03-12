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


class EnumBinarization(Layer):
    """
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, enum_size, name=None):

        if type(input_layer) is PythonWrapper.EnumBinarization:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.EnumBinarization(str(name), layers[0], int(outputs[0]), enum_size)
        super().__init__(internal)

    @property
    def enum_size(self):
        """
        """
        return self._internal.get_enum_size()

    @enum_size.setter
    def enum_size(self, enum_size):
        """
        """
        self._internal.set_enum_size(enum_size)

# ----------------------------------------------------------------------------------------------------------------------


class BitSetVectorization(Layer):
    """
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, bit_set_size, name=None):

        if type(input_layer) is PythonWrapper.BitSetVectorization:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.BitSetVectorization(str(name), layers[0], int(outputs[0]), bit_set_size)
        super().__init__(internal)

    @property
    def bit_set_size(self):
        """
        """
        return self._internal.get_bit_set_size()

    @bit_set_size.setter
    def bit_set_size(self, bit_set_size):
        """
        """
        self._internal.set_bit_set_size(bit_set_size)
