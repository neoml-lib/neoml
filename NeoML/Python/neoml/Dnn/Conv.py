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
from .BatchNormalization import BatchNormalization
import neoml.Blob as Blob


class Conv(Layer):
    """The layer that performs convolution 
    on a set of two-dimensional multi-channel images.

    Layer inputs
    ------------
    Can have several inputs, of the dimensions:
    - BatchLength * BatchWidth * ListSize - the number of images in the set
    - Height - the images' height
    - Width - the images' width
    - Depth * Channels - the number of channels the image format uses

    Layer outputs
    -------------
    The layer has as many outputs as the inputs, of the dimensions:
    - BatchLength, BatchWidth, ListSize are equal to the input dimensions
    - Height can be calculated from the input Height as
        (2 * PaddingHeight + Height - (1 + DilationHeight * (FilterHeight - 1))) / StrideHeight + 1
    - Width can be calculated from the input Width as
        (2 * PaddingWidth + Width - (1 + DilationWidth * (FilterWidth - 1))) / StrideWidth + 1
    - Depth is equal to 1
    - Channels is equal to the number of filters

    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    filter_count : int, default=1
        The number of filters.
    filter_size : (int, int), default=(3, 3)
        Filter size (height, width).
    stride_size : (int, int), default=(1, 1)
        Convolution stride (vertical, horizontal).
    padding_size : (int, int), default=(0, 0)
        The size of the padding (vertical, horizontal).
    dilation_size : (int, int), default=(1, 1)
        The step values for dilated convolution (vertical, horizontal).
        The default value of (1, 1) means no dilation.
    is_zero_free_term : bool, default=True
        Specifies if the free term should be zero.
    name : str, default=None
        The layer name. 
    """

    def __init__(self, input_layers, filter_count=1, filter_size=(3, 3), stride_size=(1, 1), padding_size=(0, 0),
                 dilation_size=(1, 1), is_zero_free_term=True, name=None):

        if type(input_layers) is PythonWrapper.Conv:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        if filter_count <= 0:
            raise ValueError('`filter_count` must be > 0.')

        if len(filter_size) != 2:
            raise ValueError('`filter_size` must contain two values (h, w).')

        if len(stride_size) != 2:
            raise ValueError('`stride_size` must contain two values (h, w).')

        if len(padding_size) != 2:
            raise ValueError('`padding_size` must contain two values (h, w).')

        if len(dilation_size) != 2:
            raise ValueError('`dilation_size` must contain two values (h, w).')

        internal = PythonWrapper.Conv(str(name), layers, outputs, int(filter_count), int(filter_size[0]),
                                      int(filter_size[1]), int(stride_size[0]), int(stride_size[1]),
                                      int(padding_size[0]), int(padding_size[1]), int(dilation_size[0]),
                                      int(dilation_size[1]), bool(is_zero_free_term))
        super().__init__(internal)

    @property
    def filter_count(self):
        """Gets the number of filters.
        """
        return self._internal.get_filter_count()

    @filter_count.setter
    def filter_count(self, new_filter_count):
        """Sets the number of filters.
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

        self._internal.set_dilation_height(int(dilation_size[0]))
        self._internal.set_dilation_width(int(dilation_size[1]))

    @property
    def filter(self):
        """Gets the filters. The dimensions:
        - BatchLength * BatchWidth * ListSize is filter_count
        - Height, Width are taken from filter_size
        - Depth, Channels are equal to the inputs' dimensions
        """
        return Blob(self._internal.get_filter())

    @property
    def free_term(self):
        """Gets the free term. The blob size is filter_count
        """
        return Blob(self._internal.get_free_term())

    def apply_batch_normalization(self, layer):
        if type(layer) is not BatchNormalization:
            raise ValueError('The `layer` must be neoml.Dnn.BatchNormalization.')

        self._internal.apply_batch_normalization(layer._internal)

# ----------------------------------------------------------------------------------------------------------------------


class Conv3D(Layer):
    """The layer that performs convolution 
    on a set of three-dimensional multi-channel images.

    Layer inputs
    ------------
    Can have several inputs, of the dimensions:
    - BatchLength * BatchWidth * ListSize - the number of images in the set
    - Height - the images' height
    - Width - the images' width
    - Depth - the images' depth
    - Channels - the number of channels the image format uses

    Layer outputs
    -------------
    The layer has as many outputs as the inputs, of the dimensions:
    - BatchLength, BatchWidth, ListSize are equal to the input dimensions
    - Height can be calculated from the input Height as
        (2 * PaddingHeight + Height - FilterHeight) / StrideHeight + 1
    - Width can be calculated from the input Width as
        (2 * PaddingWidth + Width - FilterWidth) / StrideWidth + 1
    - Depth can be calculated from the input Depth as
        (2 * PaddingDepth + Depth - FilterDepth) / StrideDepth + 1
    - Channels is equal to the number filters
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    filter_count : int, default=1
        The number of filters.
    filter_size : (int, int, int), default=(3, 3, 3)
        Filter size (height, width, depth).
    stride_size : (int, int, int), default=(1, 1, 1)
        Convolution stride (vertical, horizontal, depth).
    padding_size : (int, int, int), default=(0, 0, 0)
        The size of the padding (vertical, horizontal, depth).
    is_zero_free_term : bool, default=False
        Specifies if the free term should be zero.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layers, filter_count=1, filter_size=(3, 3, 3), stride_size=(1, 1, 1),
                 padding_size=(0, 0, 0), is_zero_free_term=False, name=None):

        if type(input_layers) is PythonWrapper.Conv3D:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        if filter_count <= 0:
            raise ValueError('`filter_count` must be > 0.')

        if len(filter_size) != 3:
            raise ValueError('`filter_size` must contain three values (h, w, d).')

        if len(stride_size) != 3:
            raise ValueError('`stride_size` must contain three values (h, w, d).')

        if len(padding_size) != 3:
            raise ValueError('`padding_size` must contain three values (h, w, d).')


        internal = PythonWrapper.Conv3D(str(name), layers, outputs, int(filter_count), int(filter_size[0]),
                                      int(filter_size[1]), int(filter_size[2]), int(stride_size[0]), int(stride_size[1]),
                                      int(stride_size[2]), int(padding_size[0]), int(padding_size[1]), int(padding_size[2]), bool(is_zero_free_term))
        super().__init__(internal)

    @property
    def filter_count(self):
        """Gets the number of filters.
        """
        return self._internal.get_filter_count()

    @filter_count.setter
    def filter_count(self, new_filter_count):
        """Sets the number of filters.
        """
        self._internal.set_filter_count(int(new_filter_count))

    @property
    def filter_size(self):
        """Gets the filter size.
        """
        return self._internal.get_filter_height(), self._internal.get_filter_width(), self._internal.get_filter_depth()

    @filter_size.setter
    def filter_size(self, filter_size):
        """Sets the filter size.
        """
        if len(filter_size) != 3:
            raise ValueError('`filter_size` must contain two values (h, w, d).')

        self._internal.set_filter_height(int(filter_size[0]))
        self._internal.set_filter_width(int(filter_size[1]))
        self._internal.set_filter_depth(int(filter_size[2]))

    @property
    def stride_size(self):
        """Gets the convolution stride.
        """
        return self._internal.get_stride_height(), self._internal.get_stride_width(), self._internal.get_stride_depth()

    @stride_size.setter
    def stride_size(self, stride_size):
        """Sets the convolution stride.
        """
        if len(stride_size) != 3:
            raise ValueError('`stride_size` must contain three values (h, w, d).')

        self._internal.set_stride_height(int(stride_size[0]))
        self._internal.set_stride_width(int(stride_size[1]))
        self._internal.set_stride_depth(int(stride_size[2]))

    @property
    def padding_size(self):
        """Gets the padding size.
        """
        return self._internal.get_padding_height(), self._internal.get_padding_width(), self._internal.get_padding_depth()

    @padding_size.setter
    def padding_size(self, padding_size):
        """Sets the padding size.
        """
        if len(padding_size) != 3:
            raise ValueError('`padding_size` must contain three values (h, w, d).')

        self._internal.set_padding_height(int(padding_size[0]))
        self._internal.set_padding_width(int(padding_size[1]))
        self._internal.set_padding_depth(int(padding_size[2]))

    @property
    def filter(self):
        """Gets the filters. The dimensions:
        - BatchLength * BatchWidth * ListSize is filter_count
        - Height, Width, Depth are taken from filter_size
        - Channels is equal to the inputs' Channels
        """
        return Blob(self._internal.get_filter())

    @property
    def free_term(self):
        """Gets the free term. The blob is of filter_count size.
        """
        return Blob(self._internal.get_free_term())

    def apply_batch_normalization(self, layer):
        if type(layer) is not BatchNormalization:
            raise ValueError('The `layer` must be neoml.Dnn.BatchNormalization.')

        self._internal.apply_batch_normalization(layer._internal)

# ----------------------------------------------------------------------------------------------------------------------


class TransposedConv3D(Layer):
    """The layer that performs transposed convolution 
    on a set of three-dimensional multi-channel images.

    Layer inputs
    ------------
    Can have several inputs, of the dimensions:
    - BatchLength * BatchWidth * ListSize - the number of images in the set
    - Height - the images' height
    - Width - the images' width
    - Depth - the images' depth
    - Channels - the number of channels the image format uses

    Layer outputs
    -------------
    The layer has as many outputs as the inputs, of the dimensions:
    - BatchLength, BatchWidth, ListSize are equal to the input dimensions
    - Height can be calculated from the input Height as
        StrideHeight * (Height - 1) + FilterHeight - 2 * PaddingHeight
    - Width can be calculated from the input Width as
        StrideWidth * (Width - 1) + FilterWidth - 2 * PaddingWidth
    - Depth can be calculated from the input Depth as
        StrideDepth * (Depth - 1) + FilterDepth - 2 * PaddingDepth
    - Channels is equal to the number of filters
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    filter_count : int, default=1
        The number of filters.
    filter_size : (int, int, int), default=(3, 3, 3)
        Filter size (height, width, depth).
    stride_size : (int, int, int), default=(1, 1, 1)
        Convolution stride (vertical, horizontal, depth).
    padding_size : (int, int, int), default=(0, 0, 0)
        The size of the padding (vertical, horizontal, depth).
    is_zero_free_term : bool, default=False
        Specifies if the free term should be zero.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layers, filter_count=1, filter_size=(3, 3, 3), stride_size=(1, 1, 1),
                 padding_size=(0, 0, 0), is_zero_free_term=False, name=None):

        if type(input_layers) is PythonWrapper.TransposedConv3D:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        if filter_count <= 0:
            raise ValueError('`filter_count` must be > 0.')

        if len(filter_size) != 3:
            raise ValueError('`filter_size` must contain three values (h, w, d).')

        if len(stride_size) != 3:
            raise ValueError('`stride_size` must contain three values (h, w, d).')

        if len(padding_size) != 3:
            raise ValueError('`padding_size` must contain three values (h, w, d).')


        internal = PythonWrapper.TransposedConv3D(str(name), layers, outputs, int(filter_count), int(filter_size[0]),
                                      int(filter_size[1]), int(filter_size[2]), int(stride_size[0]), int(stride_size[1]),
                                      int(stride_size[2]), int(padding_size[0]), int(padding_size[1]), int(padding_size[2]), bool(is_zero_free_term))
        super().__init__(internal)

    @property
    def filter_count(self):
        """Gets the number of filters.
        """
        return self._internal.get_filter_count()

    @filter_count.setter
    def filter_count(self, new_filter_count):
        """Sets the number of filters.
        """
        self._internal.set_filter_count(int(new_filter_count))

    @property
    def filter_size(self):
        """Gets the filter size.
        """
        return self._internal.get_filter_height(), self._internal.get_filter_width(), self._internal.get_filter_depth()

    @filter_size.setter
    def filter_size(self, filter_size):
        """Sets the filter size.
        """
        if len(filter_size) != 3:
            raise ValueError('`filter_size` must contain two values (h, w, d).')

        self._internal.set_filter_height(int(filter_size[0]))
        self._internal.set_filter_width(int(filter_size[1]))
        self._internal.set_filter_depth(int(filter_size[2]))

    @property
    def stride_size(self):
        """Gets the convolution stride.
        """
        return self._internal.get_stride_height(), self._internal.get_stride_width(), self._internal.get_stride_depth()

    @stride_size.setter
    def stride_size(self, stride_size):
        """Sets the convolution stride.
        """
        if len(stride_size) != 3:
            raise ValueError('`stride_size` must contain three values (h, w, d).')

        self._internal.set_stride_height(int(stride_size[0]))
        self._internal.set_stride_width(int(stride_size[1]))
        self._internal.set_stride_depth(int(stride_size[2]))

    @property
    def padding_size(self):
        """Gets the padding size.
        """
        return self._internal.get_padding_height(), self._internal.get_padding_width(), self._internal.get_padding_depth()

    @padding_size.setter
    def padding_size(self, padding_size):
        """Sets the padding size.
        """
        if len(padding_size) != 3:
            raise ValueError('`padding_size` must contain three values (h, w, d).')

        self._internal.set_padding_height(int(padding_size[0]))
        self._internal.set_padding_width(int(padding_size[1]))
        self._internal.set_padding_depth(int(padding_size[2]))

    @property
    def filter(self):
        """Gets the filters. The dimensions:
        - BatchLength, ListSize are 1
        - BatchWidth is equal to the inputs' Channels
        - Height, Width, Depth are taken from filter_size
        - Channels is filter_count
        """
        return Blob(self._internal.get_filter())

    @property
    def free_term(self):
        """Gets the free term. The blob size is filter_count.
        """
        return Blob(self._internal.get_free_term())

    def apply_batch_normalization(self, layer):
        if type(layer) is not BatchNormalization:
            raise ValueError('The `layer` must be neoml.Dnn.BatchNormalization.')

        self._internal.apply_batch_normalization(layer._internal)

# ----------------------------------------------------------------------------------------------------------------------


class TransposedConv(Layer):
    """The layer that performs transposed convolution 
    on a set of two-dimensional multi-channel images.

    Layer inputs
    ------------
    Can have several inputs, of the dimensions:
    - BatchLength * BatchWidth * ListSize - the number of images in the set
    - Height - the images' height
    - Width - the images' width
    - Depth * Channels - the number of channels the image format uses

    Layer outputs
    -------------
    The layer has as many outputs as the inputs, of the dimensions:
    - BatchLength, BatchWidth, ListSize are equal to the input dimensions
    - Height can be calculated from the input Height as
        StrideHeight * (Height - 1) + (FilterHeight - 1) * DilationHeight + 1 - 2 * PaddingHeight
    - Width can be calculated from the input Width as
        StrideWidth * (Width - 1) + (FilterWidth - 1) * DilationWidth + 1 - 2 * PaddingWidths
    - Depth is 1
    - Channels is equal to the number of filters
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    filter_count : int, default=1
        The number of filters.
    filter_size : (int, int), default=(3, 3)
        Filter size (height, width).
    stride_size : (int, int), default=(1, 1)
        Convolution stride (vertical, horizontal).
    padding_size : (int, int), default=(0, 0)
        The size of the padding (vertical, horizontal).
    dilation_size : (int, int), default=(1, 1)
        The step values for dilated convolution (vertical, horizontal).
        The default value of (1, 1) means no dilation.
    is_zero_free_term : bool, default=False
        Specifies if the free term should be zero.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layers, filter_count=1, filter_size=(3, 3), stride_size=(1, 1), padding_size=(0, 0),
                 dilation_size=(1, 1), is_zero_free_term=False, name=None):

        if type(input_layers) is PythonWrapper.TransposedConv:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        if filter_count <= 0:
            raise ValueError('`filter_count` must be > 0.')

        if len(filter_size) != 2:
            raise ValueError('`filter_size` must contain two values (h, w).')

        if len(stride_size) != 2:
            raise ValueError('`stride_size` must contain two values (h, w).')

        if len(padding_size) != 2:
            raise ValueError('`padding_size` must contain two values (h, w).')

        if len(dilation_size) != 2:
            raise ValueError('`dilation_size` must contain two values (h, w).')

        internal = PythonWrapper.TransposedConv(str(name), layers, outputs, int(filter_count), int(filter_size[0]),
                                      int(filter_size[1]), int(stride_size[0]), int(stride_size[1]),
                                      int(padding_size[0]), int(padding_size[1]), int(dilation_size[0]),
                                      int(dilation_size[1]), bool(is_zero_free_term))
        super().__init__(internal)

    @property
    def filter_count(self):
        """Gets the number of filters.
        """
        return self._internal.get_filter_count()

    @filter_count.setter
    def filter_count(self, new_filter_count):
        """Sets the number of filters.
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

        self._internal.set_dilation_height(int(dilation_size[0]))
        self._internal.set_dilation_width(int(dilation_size[1]))

    @property
    def filter(self):
        """Gets the filters. The dimensions:
        - BatchLength, ListSize are 1
        - BatchWidth is equal to the inputs' Channels * Depth
        - Height, Width are taken from filter_size
        - Channels is filter_count
        """
        return Blob(self._internal.get_filter())

    @property
    def free_term(self):
        """Gets the free term. The blob size is filter_count.
        """
        return Blob(self._internal.get_free_term())

    def apply_batch_normalization(self, layer):
        if type(layer) is not BatchNormalization:
            raise ValueError('The `layer` must be neoml.Dnn.BatchNormalization.')

        self._internal.apply_batch_normalization(layer._internal)

# ----------------------------------------------------------------------------------------------------------------------


class ChannelwiseConv(Layer):
    """The layer that performs channel-wise convolution.
    Each channel of the input is convolved
    with the corresponding channel of the filter.

    Layer inputs
    ------------
    Can have several inputs, of the dimensions:
    - BatchLength * BatchWidth * ListSize - the number of images in the set
    - Height - the images' height
    - Width - the images' width
    - Depth * Channels - the number of channels the image format uses

    Layer outputs
    -------------
    The layer has as many outputs as the inputs, of the dimensions:
    - BatchLength, BatchWidth, ListSize are equal to the input dimensions
    - Height can be calculated from the input Height as
        (2 * PaddingHeight + Height - FilterHeight)/StrideHeight + 1
    - Width can be calculated from the input Width as
        (2 * PaddingWidth + Width - FilterWidth)/StrideWidth + 1
    - Depth is equal to 1
    - Channels is equal to the number of channels in the filter and the input


    Parameters
    ----------------
    input_layers : array of (object, int) tuples or objects
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

    def __init__(self, input_layers, filter_count, filter_size=(3, 3), stride_size=(1, 1), padding_size=(0, 0),
                 is_zero_free_term=False, name=None):

        if type(input_layers) is PythonWrapper.ChannelwiseConv:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        if filter_count <= 0:
            raise ValueError('`filter_count` must be > 0.')

        if len(filter_size) != 2:
            raise ValueError('`filter_size` must contain two values (h, w).')

        if len(stride_size) != 2:
            raise ValueError('`stride_size` must contain two values (h, w).')

        if len(padding_size) != 2:
            raise ValueError('`padding_size` must contain two values (h, w).')

        internal = PythonWrapper.ChannelwiseConv(str(name), layers, outputs, int(filter_count),
                                                 int(filter_size[0]), int(filter_size[1]), int(stride_size[0]),
                                                 int(stride_size[1]), int(padding_size[0]), int(padding_size[1]),
                                                 bool(is_zero_free_term))
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
        """Gets the filter. The dimensions:
        - BatchLength, BatchWidth, ListSize, Depth are 1
        - Height, Width are taken from filter_size
        - Channels is equal to the inputs' Channels
        """
        return self._internal.get_filter()

    @property
    def free_term(self):
        """Gets the free term. The blob size is inputs' Channels.
        """
        return self._internal.get_free_term()

    def apply_batch_normalization(self, layer):
        if type(layer) is not BatchNormalization:
            raise ValueError('The `layer` must be neoml.Dnn.BatchNormalization.')

        self._internal.apply_batch_normalization(layer._internal)

# ----------------------------------------------------------------------------------------------------------------------


class TimeConv(Layer):
    """The layer that performs time convolution on a set of sequences.
    
    Layer inputs
    ------------
    Can have several inputs, of the dimensions:
    - BatchLength is the sequence length
    - BatchWidth * ListSize - the number of sequences in the set
    - Height * Width * Depth * Channels - the size of each element

    Layer outputs
    -------------
    The layer has as many outputs as the inputs, of the dimensions:
    - BatchLength can be calculated from the input BatchLength as
        (padding_front + padding_back + BatchLength - 
            - (1 + dilation * (filter_size - 1)))/stride + 1
    - BatchWidth, ListSize are equal to the input dimensions
    - Height, Width, Depth are equal to 1
    - Channels is equal to the number of filters


    Parameters
    ----------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected.
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    filter_count : int, > 0
        The number of filters.
    filter_size : int, > 0
        Filter size.
    padding_front : int, >= 0
        The number of zeros to be added at the sequence start.
    padding_back : int, >= 0
        The number of zeros to be added at the sequence end.
    dilation : int, default=1
        The step value for dilated convolution. 1 means no dilation.
    stride : int, > 0, default=1
        Convolution stride.
    name : str, default=None
        The layer name.    
    """

    def __init__(self, input_layers, filter_count, filter_size, padding_front=0, padding_back=0, dilation=1, stride=1, name=None):

        if type(input_layers) is PythonWrapper.TimeConv:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        if filter_count <= 0:
            raise ValueError('`filter_count` must be > 0.')

        if filter_size <= 0:
            raise ValueError('`filter_size` must be > 0.')

        if padding_front < 0:
            raise ValueError('`padding_front` must be >= 0.')

        if padding_back < 0:
            raise ValueError('`padding_back` must be >= 0.')

        if stride <= 0:
            raise ValueError('`stride` must be > 0.')

        if dilation <= 0:
            raise ValueError('`dilation` must be > 0.')

        internal = PythonWrapper.TimeConv(str(name), layers, outputs, int(filter_count), int(filter_size),
                                      int(padding_front), int(padding_back), int(stride), int(dilation))
        super().__init__(internal)

    @property
    def filter_count(self):
        """Gets the number of filters.
        """
        return self._internal.get_filter_count()

    @filter_count.setter
    def filter_count(self, new_filter_count):
        """Sets the number of filters.
        """
        self._internal.set_filter_count(int(new_filter_count))

    @property
    def filter_size(self):
        """Gets the filter size.
        """
        return self._internal.get_filter_size()

    @filter_size.setter
    def filter_size(self, filter_size):
        """Sets the filter size.
        """
        if filter_size <= 0:
            raise ValueError('`filter_size` must be > 0.')

        self._internal.set_filter_size(int(filter_size))

    @property
    def stride(self):
        """Gets the convolution stride.
        """
        return self._internal.get_stride()

    @stride.setter
    def stride(self, stride):
        """Sets the convolution stride.
        """
        if stride <= 0:
            raise ValueError('`stride` must be > 0.')

        self._internal.set_stride(int(stride))

    @property
    def padding_front(self):
        """Gets the padding size.
        """
        return self._internal.get_padding_front()

    @padding_front.setter
    def padding_front(self, padding):
        """Sets the padding size.
        """
        if padding < 0:
            raise ValueError('`padding_front` must be >= 0.')

        self._internal.set_padding_front(int(padding))

    @property
    def padding_back(self):
        """Gets the padding size.
        """
        return self._internal.get_padding_back()

    @padding_back.setter
    def padding_back(self, padding):
        """Sets the padding size.
        """
        if padding < 0:
            raise ValueError('`padding_back` must be >= 0.')

        self._internal.set_padding_back(int(padding))

    @property
    def dilation(self):
        """Gets the step value for dilated convolution.
        """
        return self._internal.get_dilation()

    @dilation.setter
    def dilation(self, dilation):
        """Sets the step value for dilated convolution.
        """
        if dilation <= 0:
            raise ValueError('`dilation` must be > 0.')

        self._internal.set_dilation(int(dilation))

    @property
    def filter(self):
        """Gets the filters. The dimensions:
        - BatchLength is 1
        - BatchWidth is filter_count
        - Height is filter_size
        - Width, Depth are 1
        - Channels is the inputs' Height * Width * Depth * Channels
        """
        return Blob(self._internal.get_filter())

    @property
    def free_term(self):
        """Gets the free term. The blob size is filter_count.
        """
        return Blob(self._internal.get_free_term())
