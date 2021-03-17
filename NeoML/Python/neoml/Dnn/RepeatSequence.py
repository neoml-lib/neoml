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


class RepeatSequence(Layer):
    """The layer that repeats the input sequences several times.
    
    Layer inputs
    ---------
    #1: a sequence of objects.
    The dimensions:
    - BatchLength is the sequence length
    - BatchWidth * ListSize is the number of sequences in the set
    - Height * Width * Depth * Channels is the size of each object
    
    Layer outputs
    ---------
    #1: the same sequence repeated repeat_count times.
    The dimensions:
    - BatchLength is repeat_count times larger than the input's BatchLength
    - all other dimensions are the same as for the input
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    repeat_count : int, > 0
        The number of repetitions.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, repeat_count, name=None):

        if type(input_layer) is PythonWrapper.RepeatSequence:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.RepeatSequence(str(name), layers[0], int(outputs[0]), int(repeat_count))
        super().__init__(internal)

    @property
    def repeat_count(self):
        """Gets the number of repetitions.
        """
        return self._internal.get_repeat_count()

    @repeat_count.setter
    def repeat_count(self, repeat_count):
        """Sets the number of repetitions.
        """
        self._internal.set_repeat_count(int(repeat_count))
