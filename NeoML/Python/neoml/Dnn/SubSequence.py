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


class SubSequence(Layer):
    """The layer that extracts a subsequence 
    from each vector sequence of the set.
    
    Layer inputs
    ----------
    #1: a blob with a set of objects, numbered along the BatchWidth dimension.
    
    Layer outputs
    ----------
    #1: a blob with the subsequence of objects.
    The dimensions:
    - BatchWidth is abs(length) or smaller if it doesn't fit 
        after starting at start_pos
    - the other dimensions are the same as for the input
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    start_pos : int
        The first element of the subsequence. Counted from the end if negative.
    length : int
        The length of the subsequence. Reversed order if negative.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, start_pos=0, length=1, name=None):

        if type(input_layer) is PythonWrapper.SubSequence:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.SubSequence(str(name), layers[0], int(outputs[0]), int(start_pos), int(length), False)
        super().__init__(internal)

    @property
    def start_pos(self):
        """Gets the starting position.
        """
        return self._internal.get_start_pos()

    @start_pos.setter
    def start_pos(self, start_pos):
        """Sets the starting position.
        """
        self._internal.set_start_pos(int(start_pos))

    @property
    def length(self):
        """Gets the subsequence length.
        """
        return self._internal.get_length()

    @length.setter
    def length(self, length):
        """Sets the subsequence length.
        """
        self._internal.set_length(int(length))

# ----------------------------------------------------------------------------------------------------------------------


class ReverseSequence(Layer):
    """The layer that reverses sequence order of the input.
    
    Layer inputs
    ----------
    #1: a blob of any size with sequence of objects 
        numbered along the BatchWidth dimension.
    
    Layer outputs
    ----------
    #1: the reverse sequence. The same size as the input.
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.SubSequence:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.SubSequence(str(name), layers[0], int(outputs[0]), int(-1), int(-1), True)
        super().__init__(internal)
