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
--------------------------------------------------------------------------------------------------------------
"""

import neoml.PythonWrapper as PythonWrapper
from .Dnn import Layer
from neoml.Utils import check_input_layers


class ScatterND(Layer):
    """The layer that updates certain objects in data:

    - :math:`data[indices[i]] = updates[i]`

    where `indices[...]` is an integer vector of `IndexDims` elements
    containing coordinates in the first `IndexDims` dimensions of the `data` blob

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer must have 3 inputs.

    1. Data. Set of objects of any type. The product of its first `IndexDims`
    dimensions is `ObjectCount`. The product of the rest of the dimensions
    is `ObjectSize`.

    2. Indices. Blob of integer data type. The number of channels of this blob
    is `IndexDims`. The total size must be `UpdateCount * IndexDims`.

    3. Updates. Blob of the same data type as the first. The total size of the
    blob is `UpdateCount * ObjectSize`.

    .. rubric:: Layer outputs:

    A blob of the same data type and size as first input.

    """

    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.ScatterND:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 3)

        internal = PythonWrapper.ScatterND(str(name), layers[0], int(outputs[0]),
            layers[1], int(outputs[1]), layers[2], int(outputs[2]))
        super().__init__(internal)
