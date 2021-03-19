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
from .BatchNormalization import BatchNormalization
import neoml.Blob as Blob


class FullyConnected(Layer):
    """The fully connected layer.
    It multiplies each of the input vectors by the weight matrix
    and adds the free term vector to the result.

    Layer inputs
    -----------
    The layer can have any number of inputs.
    The dimensions:
    - BatchLength * BatchWidth * ListSize is the number of vectors
    - Height * Width * Depth * Channels is the vector size; 
        should be the same for all inputs

    Layer outputs
    -----------
    The layer returns one output for each input.
    The dimensions:
    - BatchLength, BatchWidth, ListSize the same as for the input
    - Height, Width, Depth are 1
    - Channels is element_count

    Parameters
    -----------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    element_count : int, > 0
        The length of each vector in the output.
    is_zero_free_term : bool, default=False
        If True, the free term vector is set to all zeros and not trained.
        If False, the free term is trained together with the weights.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, element_count, is_zero_free_term=False, name=None):

        if type(input_layers) is PythonWrapper.FullyConnected:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        if element_count < 1:
            raise ValueError('The `element_count` must be > 0.')

        internal = PythonWrapper.FullyConnected(str(name), layers, outputs, int(element_count), bool(is_zero_free_term))
        super().__init__(internal)

    @property
    def element_count(self):
        """Gets the length of each vector in the output.
        """
        return self._internal.get_element_count()

    @property
    def zero_free_term(self):
        """Sets the length of each vector in the output.
        """
        return self._internal.get_zero_free_term()

    @zero_free_term.setter
    def zero_free_term(self, zero_free_term):
        """Checks if the free term is all zeros.
        """
        self._internal.set_zero_free_term(bool(zero_free_term))

    def apply_batch_normalization(self, layer):
        """Applies batch normalization to this layer.
        Batch normalization must be deleted from the dnn afterwards
        and layers which were connected to the batch norm must be connected to this layer.

        :param neoml.Dnn.BatchNormalization layer: batch norm to be applied
        """
        if type(layer) is not BatchNormalization:
            raise ValueError('The `layer` must be neoml.Dnn.BatchNormalization.')

        self._internal.apply_batch_normalization(layer._internal)

    @property
    def weights(self):
        """Gets the trained weights as a blob of the dimensions:
        - BatchLength * BatchWidth * ListSize equal to element_count
        - Height, Width, Depth, Channels the same as for the first input
        """
        return Blob.Blob(self._internal.get_weights())

    @weights.setter
    def weights(self, blob):
        """Sets the trained weights as a blob of the dimensions:
        - BatchLength * BatchWidth * ListSize equal to element_count
        - Height, Width, Depth, Channels the same as for the first input
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_weights(blob._internal)

    @property
    def free_term(self):
        """Gets the free term vector, of element_count length.
        """
        return Blob.Blob(self._internal.get_free_term())

    @free_term.setter
    def free_term(self, blob):
        """Sets the free term vector, of element_count length.
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_free_term(blob._internal)
