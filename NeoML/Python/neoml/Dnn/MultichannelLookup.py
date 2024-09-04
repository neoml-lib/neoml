""" Copyright (c) 2017-2024 ABBYY

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
from .Initializer import Initializer
from neoml.Utils import check_input_layers
import neoml.Blob as Blob


class MultichannelLookup(Layer):
    """The layer that trains fixed-length vector representation for the values
    of several discrete features.
    See https://en.wikipedia.org/wiki/Word2vec, 
    https://en.wikipedia.org/wiki/GloVe_(machine_learning)
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: object, tuple(object, int) or list of them
    :param dimensions: Each of the elements specifies the number and length of vectors 
        in the representation table with the element's index.
    :type dimensions: list of tuple(int, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with feature values (float or int).
        The dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** * **Height** * **Width** * **Depth**
          is the number of features
        - **Channels** is the dimension along which the feature values for
          different sets are stored. Not smaller than the number of 
          feature sets.
        
    .. rubric:: Layer outputs:

    (1) the blob with vectors for each input feature.
        The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize**, **Height**, **Width**, **Depth**
          are the same as for the input
        - **Channels** is the sum of vector lengths of all sets and additional channels
          if the input **Channels** is more than the number of tables.
    """
    def __init__(self, input_layers, dimensions=None, name=None):

        if dimensions is None:
            dimensions = [(1, 1)]

        if not type(dimensions) is list:
            raise ValueError('`dimensions` must be a list of elements like (VectorCount, VectorSize).')

        if any(not type(d) is tuple for d in dimensions):
            raise ValueError('`dimensions` must be a list of elements like (VectorCount, VectorSize).')

        if any(len(d) != 2 for d in dimensions):
            raise ValueError('`dimensions` must be a list of elements like (VectorCount, VectorSize).')

        if any(d[0] < 0 or d[1] < 1 for d in dimensions):
            raise ValueError('`dimensions` must be a list of elements like (VectorCount, VectorSize).')

        if type(input_layers) is PythonWrapper.MultichannelLookup:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.MultichannelLookup(str(name), layers, outputs, dimensions)
        super().__init__(internal)

    @property
    def dimensions(self):
        """Gets the list of representation table sizes.
        """
        return self._internal.get_dimensions()

    @dimensions.setter
    def dimensions(self, dimensions):
        """Sets the list of representation table sizes.
        """
        if not type(dimensions) is list:
            raise ValueError('`dimensions` must be a list of elements like (VectorCount, VectorSize).')

        if any(not type(d) is tuple for d in dimensions):
            raise ValueError('`dimensions` must be a list of elements like (VectorCount, VectorSize).')

        if any(len(d) != 2 for d in dimensions):
            raise ValueError('`dimensions` must be a list of elements like (VectorCount, VectorSize).')

        if any(d[0] < 0 or d[1] < 1 for d in dimensions):
            raise ValueError('`dimensions` must be a list of elements like (VectorCount, VectorSize).')

        self._internal.set_dimensions(dimensions)

    def get_embeddings(self, index):
        """Gets the representation table with the given index 
        as a blob of the dimensions:

            - BatchLength * BatchWidth * ListSize is dimensions[i].VectorCount
            - Height * Width * Depth * Channels is dimensions[i].VectorSize
        """
        return Blob.Blob(self._internal.get_embeddings(index))

    def set_embeddings(self, index, blob):
        """Sets the representation table with the given index 
        as a blob of the dimensions:

            - BatchLength * BatchWidth * ListSize is dimensions[i].VectorCount
            - Height * Width * Depth * Channels is dimensions[i].VectorSize
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        return self._internal.set_embeddings(index, blob._internal)

    def initialize(self, initializer):
        """Specifies a different initializer for this layer
        than the one set for the whole network in general.
        """
        if initializer is None:
            self._internal.clear()
    
        if not isinstance(initializer, Initializer):
            raise ValueError('The `initializer` must be an Initializer.')

        self._internal.initialize(initializer._internal)
