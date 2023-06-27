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


class Pooling(Layer):
    """The base class for pooling layers.
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.Pooling):
            raise ValueError('The `internal` must be PythonWrapper.Pooling')

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
            raise ValueError('The `filter_size` must contain two values (h, w).')

        if int(filter_size[0]) < 1 or int(filter_size[1]) < 1:
            raise ValueError('`filter_size` must contain positive values.')

        self._internal.set_filter_height(int(filter_size[0]))
        self._internal.set_filter_width(int(filter_size[1]))

    @property
    def stride_size(self):
        """Gets the filter stride: vertical and horizontal.
        """
        return self._internal.get_stride_height(), self._internal.get_stride_width()

    @stride_size.setter
    def stride_size(self, stride_size):
        """Sets the filter stride: vertical and horizontal.
        """
        if len(stride_size) != 2:
            raise ValueError('`stride_size` must contain two values (h, w).')

        if int(stride_size[0]) < 1 or int(stride_size[1]) < 1:
            raise ValueError('`stride_size` must contain positive values.')

        self._internal.set_stride_height(int(stride_size[0]))
        self._internal.set_stride_width(int(stride_size[1]))

# ----------------------------------------------------------------------------------------------------------------------


class MaxPooling(Pooling):
    """The pooling layer that finds maximum in a window.

    :param input_layers: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layers: (object, int)
    :param filter_size: The size of the window: (height, width).    
    :type filter_size: tuple(int, int), default=(3, 3)
    :param stride_size: Window stride (vertical, horizontal).  
    :type stride_size: tuple(int, int), default=(1, 1)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the set of images, of dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** - the number of images in the set
        - **Height** - the images' height
        - **Width** - the images' width
        - **Depth** * **Channels** - the number of channels the image format uses
        
    .. rubric:: Layer outputs:

    (1) the result of pooling
        The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize** are equal to the input dimensions
        - **Height** can be calculated from the input as (**Height** - **FilterHeight**)/**StrideHeight** + 1
        - **Width** can be calculated from the input as (**Width** - **FilterWidth**)/**StrideWidth** + 1
        - **Depth** and **Channels** are equal to the input dimensions
    """
    def __init__(self, input_layers, filter_size=(3, 3), stride_size=(1, 1), name=None):

        if type(input_layers) is PythonWrapper.MaxPooling:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 1)

        if len(filter_size) != 2:
            raise ValueError('`filter_size` must contain two values (h, w).')

        if int(filter_size[0]) < 1 or int(filter_size[1]) < 1:
            raise ValueError('`filter_size` must contain positive values.')

        if len(stride_size) != 2:
            raise ValueError('`stride_size` must contain two values (h, w).')

        if int(stride_size[0]) < 1 or int(stride_size[1]) < 1:
            raise ValueError('`stride_size` must contain positive values.')

        internal = PythonWrapper.MaxPooling(str(name), layers[0], outputs[0], int(filter_size[0]),
                                            int(filter_size[1]), int(stride_size[0]), int(stride_size[1]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class MeanPooling(Pooling):
    """The pooling layer that takes average over the window.

    :param input_layers: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layers: (object, int)
    :param filter_size: The size of the window: (height, width).    
    :type filter_size: tuple(int, int), default=(3, 3)
    :param stride_size: Window stride (vertical, horizontal).  
    :type stride_size: tuple(int, int), default=(1, 1)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the set of images, of dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** - the number of images in the set
        - **Height** - the images' height
        - **Width** - the images' width
        - **Depth** * **Channels** - the number of channels the image format uses
        
    .. rubric:: Layer outputs:

    (1) the result of pooling
        The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize** are equal to the input dimensions
        - **Height** can be calculated from the input as (**Height** - **FilterHeight**)/**StrideHeight** + 1
        - **Width** can be calculated from the input as (**Width** - **FilterWidth**)/**StrideWidth** + 1
        - **Depth** and **Channels** are equal to the input dimensions
    """
    def __init__(self, input_layers, filter_size=(3, 3), stride_size=(1, 1), name=None):

        if type(input_layers) is PythonWrapper.MeanPooling:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 1)

        if len(filter_size) != 2:
            raise ValueError('`filter_size` must contain two values (h, w).')

        if len(stride_size) != 2:
            raise ValueError('`stride_size` must contain two values (h, w).')

        internal = PythonWrapper.MeanPooling(str(name), layers[0], outputs[0], int(filter_size[0]),
                                             int(filter_size[1]), int(stride_size[0]), int(stride_size[1]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class GlobalMaxPooling(Layer):
    """The layer that finds maximum over the whole three-dimensional image,
    allowing for multiple largest elements to be found.
    If you set the number of largest elements to 1, it will function
    exactly as MaxPooling3d with the filter size equal to the input image size.
    
    :param input_layers: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layers: object, tuple(object, int)
    :param max_count: The number of largest elements to be found. Note that these 
        do not have to be equal to each other; the top max_count elements
        will be returned.    
    :type max_count: int, > 0
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the set of images, of dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** - the number of images in the set
        - **Height** - the images' height
        - **Width** - the images' width
        - **Depth** - the images' depth
        - **Channels** - the number of channels the image format uses
        
    .. rubric:: Layer outputs:

    (1) the maximum values found.
        The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize**, **Channels** are equal to the input dimensions
        - **Height**, **Depth** are 1
        - **Width** is max_count
    
    (2) (optional): the indices of the values found in the input blob.
        The dimensions are the same.
    """
    def __init__(self, input_layers, max_count, name=None):

        if type(input_layers) is PythonWrapper.GlobalMaxPooling:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 1)

        if int(max_count) < 1:
            raise ValueError('The `max_count` must be > 0.')

        internal = PythonWrapper.GlobalMaxPooling(str(name), layers[0], outputs[0], max_count)
        super().__init__(internal)

    @property
    def max_count(self):
        """Gets the number of largest elements to be found.
        """
        return self._internal.get_max_count()

    @max_count.setter
    def max_count(self, max_count):
        """Sets the number of largest elements to be found.
        """
        if int(max_count) < 1:
            raise ValueError('The `max_count` must be > 0.')

        self._internal.set_max_count(int(max_count))

# ----------------------------------------------------------------------------------------------------------------------


class GlobalMeanPooling(Layer):
    """The layer that finds the average over the whole three-dimensional image.
    It functions exactly as MeanPooling3d with the filter size 
    equal to the input image size.
    
    :param input_layers: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layers: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the set of images, of dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** - the number of images in the set
        - **Height** - the images' height
        - **Width** - the images' width
        - **Depth** - the images' depth
        - **Channels** - the number of channels the image format uses
        
    .. rubric:: Layer outputs:

    (1) the average values over each image.
        The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize** are equal to the input dimensions
        - **Height**, **Width**, **Depth** are 1
        - **Channels** is equal to the input **Channels**
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.GlobalMeanPooling:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 1)

        internal = PythonWrapper.GlobalMeanPooling(str(name), layers[0], outputs[0])
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class GlobalSumPooling(Layer):
    """The layer that finds the sum over the whole three-dimensional image.
    
    :param input_layers: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layers: object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the set of images, of dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** - the number of images in the set
        - **Height** - the images' height
        - **Width** - the images' width
        - **Depth** - the images' depth
        - **Channels** - the number of channels the image format uses
        
    .. rubric:: Layer outputs:

    (1) the sums over each image.
        The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize** are equal to the input dimensions
        - **Height**, **Width**, **Depth** are 1
        - **Channels** is equal to the input **Channels**
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.GlobalSumPooling:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 1)

        internal = PythonWrapper.GlobalSumPooling(str(name), layers[0], outputs[0])
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class MaxOverTimePooling(Layer):
    """The layer that finds maximums on the set of sequences,
    with the window taken over BatchLength axis.
    
    :param input_layers: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layers: object, tuple(object, int)
    :param filter_len: The window size. Set to <= 0 if the maximum over the whole 
        sequence length should be found.   
    :type filter_len: int
    :param stride_len: The window stride. Meaningful only for filter_len > 0.   
    :type stride_len: int, > 0
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the set of sequences, of dimensions:

        - **BatchLength** is the sequence length
        - **BatchWidth** * **ListSize** is the number of sequences in the set
        - **Height** * **Width** * **Depth** * **Channels** is the length of each vector
        
    .. rubric:: Layer outputs:

    (1) the pooling result.
        The dimensions:

        - **BatchLength** is:

            - 1 if filter_len is <= 0
            - (**BatchLength** - filter_len) / stride_len + 1 otherwise
        - the other dimensions are the same as for the input
    """
    def __init__(self, input_layers, filter_len, stride_len, name=None):

        if type(input_layers) is PythonWrapper.MaxOverTimePooling:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 1)

        if int(stride_len) < 1:
            raise ValueError('The `stride_len` must be > 0.')

        internal = PythonWrapper.MaxOverTimePooling(str(name), layers[0], outputs[0], int(filter_len), int(stride_len))
        super().__init__(internal)

    @property
    def filter_len(self):
        """Gets the window size.
        """
        return self._internal.get_filter_len()

    @filter_len.setter
    def filter_len(self, filter_len):
        """Sets the window size.
        """
        self._internal.set_filter_len(int(filter_len))

    @property
    def stride_len(self):
        """Gets the window stride.
        """
        return self._internal.get_stride_len()

    @stride_len.setter
    def stride_len(self, stride_len):
        """Sets the window stride.
        """
        if int(stride_len) < 1:
            raise ValueError('The `stride_len` must be > 0.')

        self._internal.set_stride_len(int(stride_len))

# ----------------------------------------------------------------------------------------------------------------------


class ProjectionPooling(Layer):
    """The layer that takes the average over one of the dimensions.
    This dimension is either compressed to a point when original_size=False,
    or stays the same length with all elements equal to the average 
    when original_size=True.
    
    :param input_layers: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layers: object, tuple(object, int)
    :param dimension: The dimension along which the average is taken.    
    :type dimension: str, {"batch_length", "batch_width", "list_size", 
        "height", "width", "depth", "channels"}
    :param original_size: Specifies if the blob should stay the same size, with the average 
        value broadcast along the pooling dimension.
    :type original_size: bool, default=False
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size.
    
    .. rubric:: Layer outputs:

    (1) the result of pooling.
        The dimensions:

        - all stay the same if original_size is True
        - if original_size is False, the pooling dimension is 1 
          and other dimensions stay the same
    """

    dimensions = ["batch_length", "batch_width", "list_size", "height", "width", "depth", "channels"]

    def __init__(self, input_layers, dimension, original_size, name=None):

        if type(input_layers) is PythonWrapper.ProjectionPooling:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 1)

        dimension_index = self.dimensions.index(dimension)

        internal = PythonWrapper.ProjectionPooling(str(name), layers[0], outputs[0], int(dimension_index), bool(original_size))
        super().__init__(internal)

    @property
    def dimension(self):
        """Gets the dimension along which the average is to be calculated.
        """
        return self.dimensions[self._internal.get_dimension()]

    @dimension.setter
    def dimension(self, dimension):
        """Sets the dimension along which the average is to be calculated.
        """
        dimension_index = self.dimensions.index(dimension)

        self._internal.set_dimension(dimension_index)

    @property
    def original_size(self):
        """Checks if the blob will stay the same size, with the average
        value broadcast along the pooling dimension.
        """
        return self._internal.get_original_size()

    @original_size.setter
    def original_size(self, original_size):
        """Specifies if the blob should stay the same size, with the average
        value broadcast along the pooling dimension.
        """
        self._internal.set_original_size(bool(original_size))

# ----------------------------------------------------------------------------------------------------------------------


class Pooling3D(Layer):
    """The base class for 3d pooling layers.
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.Pooling3D):
            raise ValueError('The `internal` must be PythonWrapper.Pooling3D')

        super().__init__(internal)

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
            raise ValueError('The `filter_size` must contain three values (h, w, d).')

        if int(filter_size[0]) < 1 or int(filter_size[1]) < 1 or int(filter_size[2]) < 1:
            raise ValueError('`filter_size` must contain positive values.')

        self._internal.set_filter_height(int(filter_size[0]))
        self._internal.set_filter_width(int(filter_size[1]))
        self._internal.set_filter_depth(int(filter_size[2]))

    @property
    def stride_size(self):
        """Gets the filter stride.
        """
        return self._internal.get_stride_height(), self._internal.get_stride_width(), self._internal.get_stride_depth()

    @stride_size.setter
    def stride_size(self, stride_size):
        """Sets the filter stride.
        """
        if len(stride_size) != 3:
            raise ValueError('`filter_size` must contain three values (h, w, d).')

        if int(stride_size[0]) < 1 or int(stride_size[1]) < 1 or int(stride_size[2]) < 1:
            raise ValueError('`stride_size` must contain positive values.')

        self._internal.set_stride_height(int(stride_size[0]))
        self._internal.set_stride_width(int(stride_size[1]))
        self._internal.set_stride_depth(int(stride_size[2]))

# ----------------------------------------------------------------------------------------------------------------------


class MaxPooling3D(Pooling3D):
    """The pooling layer that finds maximum in a window 
    for three-dimensional images.

    :param input_layers: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layers: object, tuple(object, int)
    :param filter_size: The size of the window: (height, width, depth).     
    :type filter_size: tuple(int, int, int), default=(3, 3, 3)
    :param stride_size: Window stride (vertical, horizontal, depth).  
    :type stride_size: tuple(int, int, int), default=(1, 1, 1)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the set of images, of dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** - the number of images in the set
        - **Height** - the images' height
        - **Width** - the images' width
        - **Depth** - the images' depth
        - **Channels** - the number of channels the image format uses
        
    .. rubric:: Layer outputs:

    (1) the result of pooling
        The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize** are equal to the input dimensions
        - **Height** can be calculated from the input as (**Height** - **FilterHeight**)/**StrideHeight** + 1
        - **Width** can be calculated from the input as (**Width** - **FilterWidth**)/**StrideWidth** + 1
        - **Depth** can be calculated from the input as (**Depth** - **FilterDepth**)/**StrideDepth** + 1
        - **Channels** is equal to the input **Channels**
    """
    def __init__(self, input_layers, filter_size=(3, 3, 3), stride_size=(1, 1, 1), name=None):

        if type(input_layers) is PythonWrapper.MaxPooling3D:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 1)

        if len(filter_size) != 3:
            raise ValueError('`filter_size` must contain three values (h, w, d).')

        if len(stride_size) != 3:
            raise ValueError('`stride_size` must contain three values (h, w, d).')

        internal = PythonWrapper.MaxPooling3D(str(name), layers[0], outputs[0], int(filter_size[0]),
                                            int(filter_size[1]), int(filter_size[2]), int(stride_size[0]), int(stride_size[1]), int(stride_size[2]))
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class MeanPooling3D(Pooling3D):
    """The pooling layer that takes average over a window 
    for three-dimensional images.

    :param input_layers: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layers: object, tuple(object, int)
    :param filter_size: The size of the window: (height, width, depth).     
    :type filter_size: tuple(int, int, int), default=(3, 3, 3)
    :param stride_size: Window stride (vertical, horizontal, depth).  
    :type stride_size: tuple(int, int, int), default=(1, 1, 1)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the set of images, of dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** - the number of images in the set
        - **Height** - the images' height
        - **Width** - the images' width
        - **Depth** - the images' depth
        - **Channels** - the number of channels the image format uses
        
    .. rubric:: Layer outputs:

    (1) the result of pooling
        The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize** are equal to the input dimensions
        - **Height** can be calculated from the input as (**Height** - **FilterHeight**)/**StrideHeight** + 1
        - **Width** can be calculated from the input as (**Width** - **FilterWidth**)/**StrideWidth** + 1
        - **Depth** can be calculated from the input as (**Depth** - **FilterDepth**)/**StrideDepth** + 1
        - **Channels** is equal to the input **Channels**
    """
    def __init__(self, input_layers, filter_size=(3, 3, 3), stride_size=(1, 1, 1), name=None):

        if type(input_layers) is PythonWrapper.MeanPooling3D:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 1)

        if len(filter_size) != 3:
            raise ValueError('`filter_size` must contain three values (h, w, d).')

        if len(stride_size) != 3:
            raise ValueError('`stride_size` must contain three values (h, w, d).')

        internal = PythonWrapper.MeanPooling3D(str(name), layers[0], outputs[0], int(filter_size[0]),
                                             int(filter_size[1]), int(filter_size[2]), int(stride_size[0]), int(stride_size[1]), int(stride_size[2]))
        super().__init__(internal)
