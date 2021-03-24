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


class MatrixMultiplication(Layer):
    """The layer that multiplies two sets of matrices.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the first set of matrices.
    The dimensions:
    - **BatchLength** * **BatchWidth** * **ListSize** - the number of matrices in the set
    - **Height** * **Width** * **Depth** - the height of each matrix
    - **Channels** - the width of each matrix
    
    (2) the second set of matrices.
        The dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** - the number of matrices in the set, 
          must be the same as for the first input
        - **Height** * **Width** * **Depth** - the height of each matrix, 
          must be equal to Channels of the first input
        - **Channels** - the width of each matrix
    
    .. rubric:: Layer outputs:

    (1) the set of multiplication results.
        The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize**, **Height**, **Width**, **Depth** 
          the same as for the first input
        - **Channels** the same as for the second input
    """

    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.MatrixMultiplication:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 2)

        internal = PythonWrapper.MatrixMultiplication(str(name), layers[0], layers[1], int(outputs[0]), int(outputs[1]))
        super().__init__(internal)
