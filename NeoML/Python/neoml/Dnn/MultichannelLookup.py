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
--------------------------------------------------------------------------------------------------------------
"""

import numpy
import neoml.PythonWrapper as PythonWrapper
from .Dnn import Layer
from .Utils import check_input_layers
import neoml.Blob as Blob


class MultichannelLookup(Layer):
    """
    """
    def __init__(self, input_layers, dimensions=None, name=None):

        if dimensions is None:
            dimensions = [(1, 1)]

        if not type(dimensions) is list:
            raise ValueError('The `dimensions` must be list with elements like this (VectorCount, VectorSize).')

        if any(not type(d) is tuple for d in dimensions):
            raise ValueError('The `dimensions` must be list with elements like this (VectorCount, VectorSize).')

        if any(len(d) != 2 for d in dimensions):
            raise ValueError('The `dimensions` must be list with elements like this (VectorCount, VectorSize).')

        if any(d[0] < 0 or d[1] < 1 for d in dimensions):
            raise ValueError('The `dimensions` must be list with elements like this (VectorCount, VectorSize).')

        if type(input_layers) is PythonWrapper.MultichannelLookup:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.MultichannelLookup(str(name), layers, outputs, dimensions)
        super().__init__(internal)

    @property
    def dimensions(self):
        """
        """
        return self._internal.get_dimensions()

    @dimensions.setter
    def dimensions(self, dimensions):
        """
        """
        if not type(dimensions) is list:
            raise ValueError('The `dimensions` must be list with elements like this (VectorCount, VectorSize).')

        if any(not type(d) is tuple for d in dimensions):
            raise ValueError('The `dimensions` must be list with elements like this (VectorCount, VectorSize).')

        if any(len(d) != 2 for d in dimensions):
            raise ValueError('The `dimensions` must be list with elements like this (VectorCount, VectorSize).')

        if any(d[0] < 0 or d[1] < 1 for d in dimensions):
            raise ValueError('The `dimensions` must be list with elements like this (VectorCount, VectorSize).')

        self._internal.set_dimensions(dimensions)

    def get_embeddings(self, index):
        """
        """
        return Blob.Blob(self._internal.get_embeddings(index))

    def set_embeddings(self, index, blob):
        """
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        return self._internal.set_embeddings(index, blob._internal)

    def initialize(self, initializer):
        if initializer is None:
            return self._internal.clear()
    
        if not isinstance(initializer, Dnn.Initializer):
            raise ValueError('The `initializer` must be an Initializer.')

        self._internal.initialize(initializer._internal)
