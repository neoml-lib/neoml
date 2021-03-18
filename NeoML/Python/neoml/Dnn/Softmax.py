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


class Softmax(Layer):
    """The layer that calculates softmax function on each vector of a set:
    softmax(x[0], ... , x[n-1])[i] = exp(x[i]) / (exp(x[0]) + ... + exp(x[n-1]))
    
    Layer inputs
    ----------
    #1: a data blob of any size. The area setting determines 
    which dimensions would be considered to constitute vector length.
    
    Layer outputs
    ----------
    #1: a blob of the same size with softmax applied to every vector.
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    area : ["object_size", "batch_length", "list_size", "channel"], default=object_size
        Specifies which dimensions constitute the vector length:
        - object_size: there are 
            BatchLength * BatchWidth * ListSize vectors, of
            Height * Width * Depth * Channels length each
        - batch_length: there are
            BatchWidth * ListSize * Height * Width * Depth * Channels vectors, of
            BatchLength length each
        - list_size: there are 
            BatchLength * BatchWidth * Height * Width * Depth * Channels vectors, of
            ListSize length each
        - channel: there are
            BatchLength * BatchWidth * ListSize * Height * Width * Depth vectors, of
            Channels length each
    name : str, default=None
        The layer name.
    """
    areas = ["object_size", "batch_length", "list_size", "channel"]

    def __init__(self, input_layer, area="object_size", name=None):

        if type(input_layer) is PythonWrapper.Softmax:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        area_index = self.areas.index(area)

        internal = PythonWrapper.Softmax(str(name), layers[0], int(outputs[0]), area_index)
        super().__init__(internal)

    @property
    def area(self):
        """Checks which dimensions constitute the vector length.
        """
        return self.areas[self._internal.get_area()]

    @area.setter
    def area(self, area):
        """Specifies which dimensions constitute the vector length.
        """
        area_index = self.areas.index(area)

        self._internal.set_area(area_index)
