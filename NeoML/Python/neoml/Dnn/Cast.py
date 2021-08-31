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

import neoml.PythonWrapper as PythonWrapper
from .Dnn import Layer
from neoml.Utils import check_input_layers

class Cast(Layer):
    """The layer that converts data type of its input

    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param output_type: Type of the output.
    :type output_type: str, 'float' or 'int', default='float'
    :param name: The layer name.
    :type name: str, default=None
    """

    def __init__(self, input_layer, output_type='float', name=None):

        if type(input_layer) is PythonWrapper.Cast:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if output_type != 'float' and output_type != 'int':
            raise ValueError('The `output_type` must be `int` or `float`')
        internal = PythonWrapper.Cast(str(name), layers[0], int(outputs[0]), str(output_type))
        super().__init__(internal)

    @property
    def output_type(self):
        """Gets the output type.
        """
        self._internal.get_output_type()

    @output_type.setter
    def output_type(self, output_type):
        """Sets the output type.
        """
        if output_type != 'float' and output_type != 'int':
            raise ValueError('The `output_type` must be `int` or `float`')
        self._internal.set_output_type(str(output_type))
