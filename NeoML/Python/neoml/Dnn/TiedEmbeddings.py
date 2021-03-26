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
from neoml.Utils import check_input_layers
import neoml.Blob as Blob


class TiedEmbeddings(Layer):
    """The tied embeddings layer. 
    See https://arxiv.org/pdf/1608.05859.pdf
    The representations table is taken from a MultichannelLookup layer.
    
    Layer inputs
    -----------
    The layer may have any number of inputs, of the dimensions:
    - BatchLength * BatchWidth * ListSize is the number of objects
    - Height, Width, Depth are 1
    - Channels is the embedding size
    
    Layer outputs
    -----------
    For each input the layer has one output of the same dimensions.
    
    Parameters
    -----------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    embeddings_layer_name : str
        The name of the layer used for embeddings. 
        Needs to be a MultichannelLookup layer.
    channel : int, >=0
        The channel index in the embeddings layer.
    name : str, default=None
        The layer name.
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
        """Gets the name of the layer used for representation table.
        """
        return self._internal.get_embeddings_layer_name()

    @embeddings_layer_name.setter
    def embeddings_layer_name(self, embeddings_layer_name):
        """Sets the name of the layer used for representation table.
        """
        self._internal.set_embeddings_layer_name(embeddings_layer_name)

    @property
    def channel(self):
        """Gets the channel index in the embeddings layer.
        """
        return self._internal.get_channel()

    @channel.setter
    def channel(self, channel):
        """Sets the channel index in the embeddings layer.
        """
        if channel < 0:
            raise ValueError('`channel` must be >= 0.')

        self._internal.set_channel(channel)
