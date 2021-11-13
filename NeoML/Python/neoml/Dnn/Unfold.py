""" Copyright (c) 2017-2021 ABBYY Production LLC

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


class Unfold(Layer):
    """The layer that extracts data from the regions,
    that would be affected by the convolution

    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, typle(object, int)
    :param filter_size: Filter size (height, width).
    :type filter_size: tuple(int, int), default=(1, 1)
    :param stride_size: Convolution stride (vertical, horizontal).
    :type stride_size: tuple(int, int), default=(1, 1)
    :param padding_size: The size of the padding (vertical, horizontal).
    :type padding_size: tuple(int, int), default=(0, 0) 
    :param dilation_size: The step values for dilated convolution (vertical, horizontal).
        The default value of (1, 1) means no dilation.
    :type dilation_size: tuple(int, int), default=(1, 1)

    .. rubric:: Layer inputs:

    (1) a data blob with input images, from which data should be extracted, of the dimensions:

    - **BatchLength** * **BatchWidth** * **ListSize** - the number of images in the set
    - **Height** - the images' height
    - **Width** - the images' width
    - **Depth** * **Channels** - the number of channels the image format uses

    .. rubrice:: Layer outputs:

    (1) a data blob with the operation result, of the dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize** are equal to the input dimensions
    - **Height** is equal to the product of the convolution output's height and width.
      Convolution output height is equal to
      (2 * **PaddingHeight** + **Height** - (1 + **DilationHeight** * (**FilterHeight** - 1))) / **StrideHeight** + 1 .
      Convolution output width is equal to
      (2 * **PaddingWidth** + **Width** - (1 + **DilationWidth** * (**FilterWidth** - 1))) / **StrideWidth** + 1
    - **Depth** is equal to 1
    - **Channels** is equal to the input's **Channels** multiplied by filter_height * filter_width
    """

    def __init__(self, input_layer, filter_size=(1, 1), stride_size=(1, 1), padding_size=(0, 0),
                 dilation_size=(1, 1), name=None):

        if type(input_layer) is PythonWrapper.Unfold:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if len(filter_size) != 2:
            raise ValueError('`filter_size` must contain two values (h, w).')
        if filter_size[0] < 1 or filter_size[1] < 1:
            raise ValueError('filter sizes must be >= 1')

        if len(stride_size) != 2:
            raise ValueError('`stride_size` must contain two values (h, w).')
        if stride_size[0] < 1 or stride_size[1] < 1:
            raise ValueError('strides must be >= 1')

        if len(padding_size) != 2:
            raise ValueError('`padding_size` must contain two values (h, w).')
        if padding_size[0] < 0 or padding_size[1] < 0:
            raise ValueError('paddings must be >= 0')

        if len(dilation_size) != 2:
            raise ValueError('`dilation_size` must contain two values (h, w).')
        if dilation_size[0] < 1 or dilation_size[1] < 1:
            raise ValueError('dilations must be >= 1')

        internal = PythonWrapper.Unfold(str(name), layers[0], int(outputs[0]), int(filter_size[0]), int(filter_size[1]),
                                        int(stride_size[0]), int(stride_size[1]), int(padding_size[0]), int(padding_size[1]),
                                        int(dilation_size[0]), int(dilation_size[1]))
        super().__init__(internal)

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
        if filter_size[0] < 1 or filter_size[1] < 1:
            raise ValueError('filter sizes must be >= 1')

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
        if stride_size[0] < 1 or stride_size[1] < 1:
            raise ValueError('strides must be >= 1')

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
        if padding_size[0] < 0 or padding_size[1] < 0:
            raise ValueError('paddings must be >= 0')

        self._internal.set_padding_height(int(padding_size[0]))
        self._internal.set_padding_width(int(padding_size[1]))

    @property
    def dilation_size(self):
        """Gets the step values for dilated convolution.
        """
        return self._internal.get_dilation_height(), self._internal.get_dilation_width()

    @dilation_size.setter
    def dilation_size(self, dilation_size):
        """Sets the step values for dilated convolution.
        """
        if len(dilation_size) != 2:
            raise ValueError('`dilation_size` must contain two values (h, w).')
        if dilation_size[0] < 1 or dilation_size[1] < 1:
            raise ValueError('dilations must be >= 1')

        self._internal.set_dilation_height(int(dilation_size[0]))
        self._internal.set_dilation_width(int(dilation_size[1]))
