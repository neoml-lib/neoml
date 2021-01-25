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


class ChannelwiseConv(Dnn.Layer):
    """The layer that performs channel-wise convolution.
    Each channel of the input is convolved 
    with the corresponding channel of the filter.
    
    Parameters
    ----------------
    input_layers : array of two (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    filter_count : int, default=1
        The number of channels in the filter. 
        Should be the same as the number of channels in the input.
    filter_size : (int, int), default=(3, 3)
        Filter size (height, width).
    stride_size : (int, int), default=(1, 1)
        Convolution stride (vertical, horizontal).
    padding_size : (int, int), default=(0, 0)
        The size of the padding (vertical, horizontal).
    free_term : bool, default=True
        Specifies if the free term should be non-zero.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layers, filter_count=1, filter_size=(3, 3), stride_size=(1, 1), padding_size=(0, 0),
                 free_term=True, name=None):

        if type(input_layers) is PythonWrapper.ChannelwiseConv:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, 1)

        if filter_count <= 0:
            raise ValueError('`filter_count` must be > 0.')

        if len(filter_size) != 2:
            raise ValueError('`filter_size` must contain two values (h, w).')

        if len(stride_size) != 2:
            raise ValueError('`stride_size` must contain two values (h, w).')

        if len(padding_size) != 2:
            raise ValueError('`padding_size` must contain two values (h, w).')

        internal = PythonWrapper.ChannelwiseConv(str(name), layers[0], outputs[0], int(filter_count),
                                                 int(filter_size[0]), int(filter_size[1]), int(stride_size[0]),
                                                 int(stride_size[1]), int(padding_size[0]), int(padding_size[1]),
                                                 bool(free_term))
        super().__init__(internal)

    @property
    def filter_count(self):
        """Gets the number of channels in the filter.
        """
        return self._internal.get_filter_count()

    @filter_count.setter
    def filter_count(self, new_filter_count):
        """Sets the number of channels in the filter.
        """
        self._internal.set_filter_count(int(new_filter_count))

    @property
    def filter_size(self):
        """Gets the filter size.
        """
        return self._internal.get_filter_height(), self._internal.get_filter_width()

    @filter_size.setter
    def filter_size(self, filter_size):
        """Sets the filter size.
        """
        if len(filter_size) != 2:
            raise ValueError('`filter_size` must contain two values (h, w).')

        self._internal.set_filter_height(int(filter_size[0]))
        self._internal.set_filter_width(int(filter_size[1]))

    @property
    def stride_size(self):
        """Gets the convolution stride.
        """
        return self._internal.get_stride_height(), self._internal.get_stride_width()

    @stride_size.setter
    def stride_size(self, stride_size):
        """Sets the convolution stride.
        """
        if len(stride_size) != 2:
            raise ValueError('`stride_size` must contain two values (h, w).')

        self._internal.set_stride_height(int(stride_size[0]))
        self._internal.set_stride_width(int(stride_size[1]))

    @property
    def padding_size(self):
        """Gets the padding size.
        """
        return self._internal.get_padding_height(), self._internal.get_padding_width()

    @padding_size.setter
    def padding_size(self, padding_size):
        """Sets the padding size.
        """
        if len(padding_size) != 2:
            raise ValueError('`padding_size` must contain two values (h, w).')

        self._internal.set_padding_height(int(padding_size[0]))
        self._internal.set_padding_width(int(padding_size[1]))

    @property
    def filter(self):
        """Gets the filter.
        """
        return self._internal.get_filter()

    @property
    def free_term(self):
        """Gets the free term.
        """
        return self._internal.get_free_term()
