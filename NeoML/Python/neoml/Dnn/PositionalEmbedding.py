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
import neoml.Blob as Blob


class PositionalEmbedding(Layer):
    """
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    type :
    name : str, default=None
        The layer name.
    """
    types = ["learnable_addition", "transformers"]

    def __init__(self, input_layer, type_name, name=None):

        if type(input_layer) is PythonWrapper.PositionalEmbedding:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        type_index = self.types.index(type_name)

        internal = PythonWrapper.PositionalEmbedding(str(name), layers[0], int(outputs[0]), type_index)
        super().__init__(internal)

    @property
    def type(self):
        """
        """
        return self.types[self._internal.get_type()]

    @type.setter
    def type(self, type):
        """
        """
        type_index = self.types.index(type)

        self._internal.set_type(type_index)

    @property
    def addends(self):
        """
        """
        return Blob.Blob(self._internal.get_addends())

    @addends.setter
    def addends(self, addends):
        """
        """
        self._internal.set_addends(addends._internal)
