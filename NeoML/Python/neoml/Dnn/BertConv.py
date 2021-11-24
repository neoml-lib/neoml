""" Copyright (c) 2017-2021 ABBYY Production LLC

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

import neoml.PythonWrapper
from .Dnn import Layer
from neoml.Utils import check_input_layers


class BertConv(Layer):
    """This layer performs a special convolution used in BERT architecture

    This operation extracts the convolution regions as if 1-dimensional kernel of size (kernel size)
    would move along (sequence length) padded by ((kernel size) - 1 / 2) zeros from both sides.
    Then it applies different kernel values for every position along (sequence length), (batch size) and (num heads).
    The only dimension shared betweed different kernels is (head size).
    The kernel values are provided by an additional input.

    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int) of list of them
    :param name: The layer name.
    :type name: str, default=None

    .. rublic:: Layer inputs:

    (1) convolution data
        - BD_BatchLentgh is equal to (sequence length)
        - BD_BatchWidth is equal to (batch size)
        - BD_Channels is equal to (attention heads) * (head size)
        - others are equal to 1

    (2) convolution kernels
        - BD_BatchLength is equal to (sequence length)
        - BD_BatchWidth is equal to (batch size) * (attention heads)
        - BD_Height is equal to (kernel size)
        - others are equal to 1

    .. rubric:: Layer outputs:

    (2) convolution result
        - BD_BatchLength is equal to (sequence length)
        - BD_BatchWidth is equal to (batch size) * (attention heads)
        - BD_Height is equal to (head size)
        - others are equal to 1
    """
    def __init__(self, input_layer, name=None):
        if type(input_layer) is PythonWrapper.BertConv:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 2)

        internal = PythonWrapper.BertConv(str(name), layers, outputs)
        super().__init__(internal)
