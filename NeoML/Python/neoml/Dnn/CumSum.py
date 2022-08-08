""" Copyright (c) 2017-2022 ABBYY Production LLC

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


class CumSum(Layer):
    """The layer that calculates cumulative sum along the given dimension.

    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param dimension: The dimension along which the cumulative sum is to be calculated.
    :type dimension: str, {'batch_length', 'batch_width', 'list_size', 'height', 'width',
        'depth', 'channels'}, default='channels'
    :param reverse: If True then cumulative sums will be calculated in reverse order
    :type reverse: bool, default=False
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size and data type

    .. rubric:: Layer outputs:

    (1) a blob of the same size and data type with the cumulative sums
    """
    dimensions = ["batch_length", "batch_width", "list_size", "height", "width", "depth", "channels"]

    def __init__(self, input_layer, dimension="channels", reverse=False, name=None):

        if type(input_layer) is PythonWrapper.CumSum:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        dimension_index = self.dimensions.index(dimension)

        internal = PythonWrapper.CumSum(str(name), layers[0], int(outputs[0]), dimension_index, bool(reverse))
        super().__init__(internal)

    @property
    def dimension(self):
        """Gets the dimension along which the cumulative sum is to be calculated.
        """
        return self.dimensions[self._internal.get_dimension()]

    @dimension.setter
    def dimension(self, dimension):
        """Sets the dimension along which the cumulative sum is to be calculated.
        """
        dimension_index = self.dimensions.index(dimension)

        self._internal.set_dimension(dimension_index)

    @property
    def reverse(self):
        """Gets the flag of reversed sum order
        """
        return self._internal.is_reverse()

    @reverse.setter
    def reverse(self, reverse):
        """Sets the flag of reversed sum order
        """
        self._internal.set_reverse(bool(reverse))
