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


class Transpose(Layer):
    """The layer that switches two of the blob dimensions,
    moving the data inside accordingly.
    
    Layer inputs
    ----------
    #1: a data blob to be transposed, of any size.
    
    Layer outputs
    ----------
    #1: the result of transposition.
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    first_dim : {"batch_length", "batch_width", "list_size", 
                    "height", "width", "depth", "channels"}
        One of the dimensions that should be switched.
    second_dim : {"batch_length", "batch_width", "list_size", 
                    "height", "width", "depth", "channels"}
        The other dimension that should be switched.
    name : str, default=None
        The layer name.
    """

    dimensions = ["batch_length", "batch_width", "list_size", "height", "width", "depth", "channels"]

    def __init__(self, input_layer, first_dim='height', second_dim='width', name=None):

        if type(input_layer) is PythonWrapper.Transpose:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        first_index = self.dimensions.index(first_dim)
        second_index = self.dimensions.index(second_dim)

        internal = PythonWrapper.Transpose(str(name), layers[0], int(outputs[0]), int(first_index), int(second_index))
        super().__init__(internal)

    @property
    def first_dim(self):
        """Gets the first dimension to be switched.
        """
        return self.dimensions[self._internal.get_first_dim()]

    @first_dim.setter
    def first_dim(self, first_dim):
        """Sets the first dimension to be switched.
        """
        first_index = self.dimensions.index(first_dim)

        self._internal.set_first_dim(first_index)

    @property
    def second_dim(self):
        """Gets the second dimension to be switched.
        """
        return self.dimensions[self._internal.get_second_dim()]

    @second_dim.setter
    def second_dim(self, second_dim):
        """Sets the second dimension to be switched.
        """
        second_index = self.dimensions.index(second_dim)

        self._internal.set_second_dim(second_index)
