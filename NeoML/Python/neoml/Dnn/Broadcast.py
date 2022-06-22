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


class Broadcast(Layer):
    """The layer that broadcasts all of its inputs to the same size.
    Each dimension of each of the inputs may be `1` or fixed non-trivial size.
    E.g. if there is one input with 3 Channels, then every other input must have 1 or 3 Channels.

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:
    
    The layer accepts an arbitrary number of inputs of varied size.
    The only size restriction has been described above.
    
    .. rubric:: Layer outputs:

    The layer has as many outputs as the inputs.
    The sizes of all the outputs are the same.
    Size of each dimension is equal to the maximum size of this dimension among inputs.
    """
    def __init__(self, input_layers, name=None):
        if type(input_layers) is PythonWrapper.Broadcast:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.Broadcast(str(name), layers, outputs)
        super().__init__(internal)
