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


class EnumBinarization(Layer):
    """The layer that converts enumeration values into one-hot encoding.
    
    Layer inputs
    ----------
    #1: a blob with int or float data that contains enumeration values.
    The dimensions:
    - Channels is 1
    - the other dimensions may be of any length
    
    Layer outputs
    ----------
    #1: a blob with the vectors that one-hot encode the enumeration values.
    The dimensions:
    - Channels is enum_size
    - the other dimensions stay the same as in the first input
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    enum_size : int, > 0
        The number of constants in the enumeration.
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
        """Gets the number of constants in the enumeration.
        """
        return self._internal.get_enum_size()

    @enum_size.setter
    def enum_size(self, enum_size):
        """Sets the number of constants in the enumeration.
        """
        self._internal.set_enum_size(enum_size)

# ----------------------------------------------------------------------------------------------------------------------


class BitSetVectorization(Layer):
    """The layer that converts a bitset into vectors of ones and zeros.
    
    Layer inputs
    ----------
    #1: a blob with int data containing bitsets. 
    The dimensions:
    - BatchLength * BatchWidth * ListSize * Height * Width * Depth
        is the number of bitsets
    - Channels is bitset itself
    
    Layer outputs
    ----------
    #1: a blob with the result of vectorization.
    The dimensions:
    - Channels is equal to bit_set_size
    - the other dimensions are the same as for the input
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    bit_set_size : int, > 0
        The size of the bitset.
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
        """Gets the bitset size.
        """
        return self._internal.get_bit_set_size()

    @bit_set_size.setter
    def bit_set_size(self, bit_set_size):
        """Sets the bitset size.
        """
        self._internal.set_bit_set_size(bit_set_size)
