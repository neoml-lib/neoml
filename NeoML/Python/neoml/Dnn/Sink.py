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

import neoml.PythonWrapper as PythonWrapper
from .Dnn import Layer
from .Utils import check_input_layers
import neoml.Blob as Blob


class Sink(Layer):
    """The sink layer that serves to pass a data blob out of the network.
    Use the ``get_blob`` method to retrieve the blob.
    
    Layer inputs
    ---------------
    The layer has one input that accepts a data blob of any size.
    
    Layer outputs
    ---------------
    The layer has no outputs.
    
    Parameters
    ---------------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.Sink:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.Sink(str(name), layers[0], int(outputs[0]))
        super().__init__(internal)
        
    def get_blob(self):
        """Gets the data blob.
        """
        return Blob.Blob(self._internal.get_blob())
