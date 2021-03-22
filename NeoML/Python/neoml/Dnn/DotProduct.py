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


class DotProduct(Layer):
    """The layer that calculates the dot product of its two inputs: 
    each object in the first input is multiplied by the object
    with the same index in the second input.
    
    :param input_layer: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layer: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer has two inputs, which must contain blobs of the same dimensions.
    
    .. rubric:: Layer outputs:

    (1) a blob with the dot product.
    The dimensions:
    - **BatchLength**, **BatchWidth**, **ListSize** equal to the inputs' dimensions
    - **Height**, **Width**, **Depth**, **Channels** equal to 1
    """

    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.DotProduct:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 2)

        internal = PythonWrapper.DotProduct(str(name), layers[0], layers[1], int(outputs[0]), int(outputs[1]))
        super().__init__(internal)
