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
import neoml.Dnn as Dnn
import neoml.Utils as Utils


class Argmax(Dnn.Layer):
    """The layer that finds the maximum element along the given dimension. 
    If there are several maximum elements, all their coordinates are returned.
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    dimension : {'batch_length', 'batch_width', 'list_size', 'height', 'width',
                'depth', 'channels'}, default='channels'
        The dimension along which the maximum is to be found.
    name : str, default=None
        The layer name.
    """
    dimensions = ["batch_length", "batch_width", "list_size", "height", "width", "depth", "channels"]

    def __init__(self, input_layer, dimension="channels", name=None):

        if type(input_layer) is PythonWrapper.Argmax:
            super().__init__(input_layer)
            return

        layers, outputs = Utils.check_input_layers(input_layer, 1)

        dimension_index = self.dimensions.index(dimension)

        internal = PythonWrapper.Argmax(str(name), layers[0], int(outputs[0]), dimension_index)
        super().__init__(internal)

    @property
    def dimension(self):
        """Gets the dimension along which the maximum is to be found.
        """
        return self.dimensions[self._internal.get_dimension()]

    @dimension.setter
    def dimension(self, dimension):
        """Sets the dimension along which the maximum is to be found.
        """
        dimension_index = self.dimensions.index(dimension)

        self._internal.set_dimension(dimension_index)
