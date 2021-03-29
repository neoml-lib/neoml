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


class SequenceSum(Layer):
    """The layer that adds up object sequences.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a set of object sequences.
        The dimensions:

        - **BatchLength** is the sequence length
        - **BatchWidth** is the number of sequences in the set
        - **ListSize** is 1
        - **Height** * **Width** * **Depth** * **Channels** is the object size
    
    .. rubric:: Layer outputs:

    (1) the results of adding up each of the sequences.
        The dimensions:

        - **BatchLength** is 1
        - the other dimensions are the same as for the input
    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.SequenceSum:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.SequenceSum(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)
