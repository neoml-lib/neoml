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


class FullyConnected(Layer):
    """
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
        """
        """
        return self._internal.get_element_count()

    @property
    def zero_free_term(self):
        """
        """
        return self._internal.get_zero_free_term()

    @zero_free_term.setter
    def zero_free_term(self, zero_free_term):
        """
        """
        self._internal.set_zero_free_term(bool(zero_free_term))


    def apply_batch_normalization(self, layer):
        if not type(layer) is BatchNormalization:
            raise ValueError('The `layer` must be neoml.BatchNormalization.')

        self._internal.apply_batch_normalization(layer._internal)	

    @property
    def weights(self):
        """
        """
        return Blob.Blob(self._internal.get_weights())

    @weights.setter
    def weights(self, blob):
        """
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_weights(blob._internal)

    @property
    def free_term(self):
        """
        """
        return Blob.Blob(self._internal.get_free_term())

    @free_term.setter
    def free_term(self, blob):
        """
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_free_term(blob._internal)
