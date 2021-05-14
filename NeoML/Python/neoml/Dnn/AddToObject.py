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


class AddToObject(Layer):
    """The layer that adds the same object to all objects in its first input.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the data blob of the dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** - the number of objects
        - **Height** * **Width** * **Depth** * **Channels** - object size
    
    (2) the data blob with the object to add to the first input.
        The dimensions:

        - **Height**, **Width**, **Depth**, **Channels** are the same as for the first input
        - the other dimensions are 1
    
    .. rubric:: Layer outputs:

    (1) a blob that contains the result of adding 
        the second input to each object of the first.
        The dimensions are the same as for the first input.
    """

    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.AddToObject:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 2)

        internal = PythonWrapper.AddToObject(str(name), layers[0], layers[1], int(outputs[0]), int(outputs[1]))
        super().__init__(internal)
