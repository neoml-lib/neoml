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


class TiedEmbeddings(Layer):
    """
    """
    def __init__(self, input_layers, embeddings_layer_name, channel, name=None):

        if type(input_layers) is PythonWrapper.TiedEmbeddings:
            super().__init__(input_layers)
            return

        if channel < 0:
            raise ValueError('`channel` must be >= 0.')

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.TiedEmbeddings(str(name), layers, outputs, str(embeddings_layer_name), int(channel))
        super().__init__(internal)

    @property
    def embeddings_layer_name(self):
        """
        """
        return self._internal.get_embeddings_layer_name()

    @embeddings_layer_name.setter
    def embeddings_layer_name(self, embeddings_layer_name):
        """
        """
        self._internal.set_embeddings_layer_name(embeddings_layer_name)

    @property
    def channel(self):
        """
        """
        return self._internal.get_channel()

    @channel.setter
    def channel(self, channel):
        """
        """
        if channel < 0:
            raise ValueError('`channel` must be >= 0.')

        self._internal.set_channel(channel)
