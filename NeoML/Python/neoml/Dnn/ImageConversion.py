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


class ImageResize(Layer):
    """The layer that resizes a set of two-dimensional multi-channel images.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param deltas: The differences between the original and the resized image, 
        on each side. If the difference is negative, rows or columns are
        removed from the specified side. If it is positive, rows or
        columns are added and filled with the default_value pixels.
    :type deltas: str, ("left", "right", "top", "bottom")
    :param default_value: The value for the added pixels.
    :type default_value: float, default=0.0 
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a set of images, of the dimensions:
    - **BatchLength** * **BatchWidth** * **ListSize** - the number of images
    - **Height** - the images' height
    - **Width** - the images' width
    - **Depth** * **Channels** - the number of channels the image format uses
    
    .. rubric:: Layer outputs:

    (1) a blob with the resized images, of the dimensions:
    - **BatchLength**, **BatchWidth**, **ListSize**, **Depth**, **Channels** are 
        equal to the input dimensions
    - **Height** is the input Height plus the sum of top and bottom deltas
    - **Width** is the input Width plus the sum of right and left deltas
    """

    def __init__(self, input_layer, deltas, default_value=0.0, name=None):

        if type(input_layer) is PythonWrapper.ImageResize:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if len(deltas) != 4:
            raise ValueError('The `deltas` must contain 4 elements.')

        internal = PythonWrapper.ImageResize(str(name), layers[0], int(outputs[0]), int(deltas[0]), int(deltas[1]), int(deltas[2]), int(deltas[3]), default_value)
        super().__init__(internal)

    @property
    def deltas(self):
        """Gets the size differences on each side.
        """
        return self._internal.get_deltas()

    @deltas.setter
    def deltas(self, deltas):
        """Sets the size differences on each side.
        """
        if len(deltas) != 4:
            raise ValueError('The `deltas` must contain 4 elements.')
        self._internal.set_deltas(deltas)

    @property
    def default_value(self):
        """Gets the default value for new pixels.
        """
        return self._internal.get_default_value()

    @default_value.setter
    def default_value(self, default_value):
        """Sets the default value for new pixels.
        """
        self._internal.set_default_value(default_value)

# ----------------------------------------------------------------------------------------------------------------------


class PixelToImage(Layer):
    """The layer that creates a set of two-dimensional images using a set of
    pixel sequences with specified coordinates.
    
    Layer inputs
    ----------
    #1: a blob with pixel sequences.
    The dimensions:
    - BatchLength is 1
    - BatchWidth is the number of sequences in the set
    - ListSize is the length of each sequence
    - Height, Width, Depth are 1
    - Channels is the number of channels for the pixel sequences 
        and the output images.
    
    #2: a blob with integer data that contains lists of pixel coordinates.
    The dimensions:
    - BatchWidth, ListSize are the same as for the first input
    - the other dimensions are 1
    
    Layer outputs
    ----------
    #1: a blob with images.
    The dimensions:
    - BatchLength is 1
    - BatchWidth is the same as for the first input
    - ListSize is 1
    - Height is the specified image height
    - Width is the specified image width
    - Depth is 1
    - Channels is the same as for the first input
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    height : int
        The height of the resulting images.
    width : int
        The width of the resulting images.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, height, width, name=None):

        if type(input_layer) is PythonWrapper.PixelToImage:
            super().__init__(input_layer)
            return

        if height < 1:
            raise ValueError('The `height` must be > 0.')

        if width < 1:
            raise ValueError('The `width` must be > 0.')

        layers, outputs = check_input_layers(input_layer, 2)

        internal = PythonWrapper.PixelToImage(str(name), layers[0], int(outputs[0]), layers[1], int(outputs[1]), int(height), int(width))
        super().__init__(internal)

    @property
    def height(self):
        """Gets the output image height.
        """
        return self._internal.get_height()

    @height.setter
    def height(self, height):
        """Sets the output image height.
        """
        if height < 1:
            raise ValueError('The `height` must be > 0.')
        self._internal.set_height(height)

    @property
    def width(self):
        """Gets the output image width.
        """
        return self._internal.get_width()

    @width.setter
    def width(self, width):
        """Sets the output image width.
        """
        if width < 1:
            raise ValueError('The `width` must be > 0.')
        self._internal.set_width(width)

# ----------------------------------------------------------------------------------------------------------------------


class ImageToPixel(Layer):
    """The layer that extracts a set of pixel sequences along the specified 
    coordinates from a set of two-dimensional images.
    
    Layer inputs
    ----------
    #1: a set of two-dimensional images.
    The blob dimensions:
    - BatchLength is 1
    - BatchWidth is the number of sequences in the set
    - ListSize 1
    - Height is the images' height
    - Width is the images' width
    - Depth is 1
    - Channels is the number of channels the image format uses
    
    #2: a blob with integer data that contains the pixel sequences.
    The dimensions: 
    - BatchWidth is the same as for the first input
    - ListSize is the length of each sequence
    - all other dimensions are 1
    
    Layer outputs
    ----------
    #1: a blob with the pixel sequences.
    The dimensions:
    - BatchLength is 1
    - BatchWidth is the inputs' BatchWidth
    - ListSize is the same as for the second input
    - Height, Width, Depth are 1
    - Channels is the same as for the first input
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, name=None):

        if type(input_layer) is PythonWrapper.ImageToPixel:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 2)

        internal = PythonWrapper.ImageToPixel(str(name), layers[0], layers[1], int(outputs[0]), int(outputs[1]))
        super().__init__(internal)
