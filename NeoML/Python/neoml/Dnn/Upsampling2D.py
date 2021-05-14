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


class Upsampling2D(Layer):
    """The layer that scales up a set of two-dimensional multi-channel images.
    The new pixels are filled up by repeating the existing pixels' values,
    without any interpolation.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: object, tuple(object, int) or list of them
    :param height_copy_count: The height multiplier.
    :type height_copy_count: int, > 0
    :param width_copy_count: The width multiplier.
    :type width_copy_count: int, > 0
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer can have any amount of inputs, each with a set of images.
    The dimensions:

    - **BatchLength** * **BatchWidth** * **ListSize** is the number of images
    - **Height** is the images' height
    - **Width** is the images' width
    - **Depth** * **Channels** is the number of channels the image format uses
    
    .. rubric:: Layer outputs:

    For each input, there is a corresponding output with upscaled images.
    The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize**, **Depth**, **Channels** equal 
      the correspoonding input dimensions
    - **Height** is height_copy_count times larger than the input **Height**
    - **Width** is width_copy_count times larger than the input **Width**

    """

    def __init__(self, input_layers, height_copy_count, width_copy_count, name=None):

        if type(input_layers) is PythonWrapper.Upsampling2D:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        if int(height_copy_count) < 1:
            raise ValueError('The `height_copy_count` must be > 0.')

        if int(width_copy_count) < 1:
            raise ValueError('The `width_copy_count` must be > 0.')

        internal = PythonWrapper.Upsampling2D(str(name), layers, outputs, int(height_copy_count), int(width_copy_count))
        super().__init__(internal)

    @property
    def height_copy_count(self):
        """Gets the height multiplier.
        """
        return self._internal.get_height_copy_count()

    @height_copy_count.setter
    def height_copy_count(self, height_copy_count):
        """Sets the height multiplier.
        """
        if int(height_copy_count) < 1:
            raise ValueError('The `height_copy_count` must be > 0.')

        self._internal.set_height_copy_count(int(height_copy_count))

    @property
    def width_copy_count(self):
        """Gets the width multiplier.
        """
        return self._internal.get_width_copy_count()

    @width_copy_count.setter
    def width_copy_count(self, width_copy_count):
        """Sets the width multiplier.
        """
        if int(width_copy_count) < 1:
            raise ValueError('The `width_copy_count` must be > 0.')

        self._internal.set_width_copy_count(int(width_copy_count))
