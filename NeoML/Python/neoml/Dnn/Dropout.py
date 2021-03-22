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


class Dropout(Layer):
    """The layer that randomly sets some elements of the single input to 0.
    If the input BatchLength is greater than 1, all elements 
    along the same BatchLength coordinate will use the same mask.
    When the network is not being trained (for example, during a test run),
    the dropout will not happen.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer:object, tuple(object, int)
    :param rate: The proportion of elements that will be set to 0.
    :type rate: float, [0..1]
    :param spatial: Turns on and off the spatial dropout mode. 
        When True, the whole contents of a channel will be filled with zeros,
        instead of elements one by one.
        Useful for convolutional networks.
    :type spatial: bool, default=False
    :param batchwise: Turns on and off the batchwise dropout mode.
        When True, the same mask will be used along the same BatchWidth.
        Useful for large input size.
    :type batchwise: bool, default=False
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any dimensions.
    
    .. rubric:: Layer outputs:

    (1) a blob of the same dimensions, with some of the elements set to 0,
    during training only.
    When you run the network, this layer does nothing.
    """

    def __init__(self, input_layer, rate=0.5, spatial=False, batchwise=False, name=None):

        if type(input_layer) is PythonWrapper.Dropout:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if rate < 0 or rate >= 1:
            raise ValueError('The `rate` must be in [0, 1).')

        internal = PythonWrapper.Dropout(str(name), layers[0], int(outputs[0]), float(rate), bool(spatial), bool(batchwise))
        super().__init__(internal)

    @property
    def rate(self):
        """Gets the dropout rate.
        """
        return self._internal.get_rate()

    @rate.setter
    def rate(self, rate):
        """Sets the dropout rate.
        """
        if rate < 0 or rate >= 1:
            raise ValueError('The `rate` must be in [0, 1).')

        self._internal.set_rate(rate)

    @property
    def spatial(self):
        """Checks if the spatial mode is on.
        """
        return self._internal.get_spatial()

    @spatial.setter
    def spatial(self, spatial):
        """Turns the spatial mode on and off.
        """
        self._internal.set_spatial(bool(spatial))

    @property
    def batchwise(self):
        """Checks if the batchwise mode is on.
        """
        return self._internal.get_batchwise()

    @batchwise.setter
    def batchwise(self, batchwise):
        """Turns the batchwise mode on and off.
        """
        self._internal.set_batchwise(bool(batchwise))
