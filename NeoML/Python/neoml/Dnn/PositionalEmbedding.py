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
import neoml.Blob as Blob


class PositionalEmbedding(Layer):
    """The layer that maps positions in sequence into vectors, 
    optionally trainable. The exact formula depends on the type_name parameter.
    
    The formula for "transformers":
    result[i][j][k] = input[i][j][k] + sin(j / pow(10000, (k / vectorLength))),
    where
    - i is the index of sequence in batch (from 0 to BatchWidth - 1)
    - j is the position of vector in sequence (from 0 to ListSize - 1)
    - k is the index of the element in the vector (from 0 to vectorLength - 1)
    
    The formula for "learnable_addition":
    result[i][j] = input[i][j] + addends[j],
    where
    - i is the index of sequence in batch (from 0 to BatchWidth - 1)
    - j is the position of vector in sequence (from 0 to ListSize - 1)
    - addends is the trainable vector to be added
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param type_name: The operation type.    
    :type type_name: str, {"learnable_addition", "transformers"}
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with vector sequences.
    The dimensions:
    - **BatchLength** is 1
    - **BatchWidth** is the number of sequences in the set
    - **ListSize** is the sequence length
    - **Height** * **Width** * **Depth** * **Channels** is the vector length;
        for "transformers", **Height**, **Width**, and **Depth** should be 1,
        vector length is equal to **Channels**
    
    .. rubric:: Layer outputs:

    (1) the transformation result, of the same dimensions as the input.
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
        """Gets the type of operation the layer performs.
        """
        return self.types[self._internal.get_type()]

    @type.setter
    def type(self, type):
        """Sets the type of operation the layer performs.
        """
        type_index = self.types.index(type)

        self._internal.set_type(type_index)

    @property
    def addends(self):
        """Gets the trainable vectors added. The blob dimensions:
        - BatchLength, BatchWidth are 1
        - the other dimensions are the same as for the input
        """
        return Blob.Blob(self._internal.get_addends())

    @addends.setter
    def addends(self, addends):
        """Sets the trainable vectors added. The blob dimensions:
        - BatchLength, BatchWidth are 1
        - the other dimensions are the same as for the input
        """
        self._internal.set_addends(addends._internal)
