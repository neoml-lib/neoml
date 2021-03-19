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

class AccumulativeLookup(Layer):
    """The layer that trains fixed-length vector representations
    for the values of a discrete feature.
    It can work only with one feature. When several values of the feature
    are passed, the sum of the corresponding vectors is returned.
    
    Layer inputs
    ----------
    #1: a data blob with integer data that contains the feature values.
    The dimensions:
    - BatchLength * BatchWidth * ListSize equal to the number 
    of different values the feature can take
    - Height * Width * Depth * Channels equal to the number of values in the set
    
    Layer outputs
    ----------
    #1: a blob with the sum of vector representations of the given feature values.
    The dimensions:
    - BatchLength, BatchWidth, ListSize equal to these dimensions of the input
    - Height, Width, Depth equal to 1
    - Channels equal to the vector length (size parameter below)
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    count : int
        The number of vectors in the representation table.
    size : int
        The length of each vector in the representation table.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, count, size, name=None):

        if type(input_layer) is PythonWrapper.AccumulativeLookup:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.AccumulativeLookup(str(name), layers[0], int(outputs[0]), int(count), int(size))
        super().__init__(internal)

    @property
    def size(self):
        """Gets the vector length.
        """
        return self._internal.get_size()

    @size.setter
    def size(self, size):
        """Sets the vector length.
        """
        self._internal.set_size(int(size))

    @property
    def count(self):
        """Gets the number of vectors.
        """
        return self._internal.get_count()

    @count.setter
    def count(self, count):
        """Sets the number of vectors.
        """
        self._internal.set_count(int(count))
