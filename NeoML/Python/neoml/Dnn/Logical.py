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

class Not(Layer):
    """The layer that calculates logical not for each
    element of the single input:

    - :math:`f(x) = x == 0 ? 1 : 0`

    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with integer data of any size

    .. rubric:: Layer outputs:

    (1) a blob with integer data of the same size with the result

    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.Not:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.Not(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)


class Less(Layer):
    """The layer that compares 2 inputs element-by-element:

    - :math:`less[i] = first[i] < second[i] ? 1 : 0`

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer must have 2 inputs of the same size and data type.

    .. rubric:: Layer outputs:

    A blob with integer data of the same size as the inputs.

    """

    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.Less:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 2)

        internal = PythonWrapper.Less(str(name), layers[0], int(outputs[0]),
            layers[1], int(outputs[1]))
        super().__init__(internal)


class Equal(Layer):
    """The layer that compares 2 inputs element-by-element:

    - :math:`equal[i] = first[i] == second[i] ? 1 : 0`

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer must have 2 inputs of the same size and data type.

    .. rubric:: Layer outputs:

    A blob with integer data of the same size as the inputs.

    """

    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.Equal:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 2)

        internal = PythonWrapper.Equal(str(name), layers[0], int(outputs[0]),
            layers[1], int(outputs[1]))
        super().__init__(internal)


class Where(Layer):
    """The layer that merges 2 blobs based on a mask:

    - :math:`where[i] = first[i] != 0 ? second[i] : third[i]`

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer must have 3 inputs of the same size.
    First input must be of integer data type.
    Data types of second and third inputs must match.

    .. rubric:: Layer outputs:

    A blob with integer data of the same size and data type as the second input.

    """

    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.Where:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 3)

        internal = PythonWrapper.Where(str(name), layers[0], int(outputs[0]),
            layers[1], int(outputs[1]), layers[2], int(outputs[2]))
        super().__init__(internal)
