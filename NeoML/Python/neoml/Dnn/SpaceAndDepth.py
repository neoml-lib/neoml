""" Copyright (c) 2017-2024 ABBYY

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


class SpaceToDepth(Layer):
    """The layer that splits images into square blocks of size `k x k x Ch` and
    writes the contents of these blocks to the corresponding pixels (`1 x 1 x Ch*k*k`) of the
    output images in channel-last ordering.

    As a result image of size `H x W x Ch` is transformed into images of size `H/k x W/k x Ch*k*k`.

    This operation is the inverse function of DepthToSpace.

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: object, tuple(object, int) or list of them
    :param block_size: The size of the block (`k` from the formula).
    :type block_size: int, default=1
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    Has single input, of the dimensions:

    - **BatchLength** * **BatchWidth** * **ListSize** is equal to the number of images
    - **Height** is the image height; should be a multiple of `block_size`
    - **Width** is the image width; should be a multiple of `block_size`
    - **Depth** is equal to `1`
    - **Channels** is the number of channels in the image format

    .. rubric:: Layer outputs:

    The layer has single output, of the dimensions:

    - **BatchLength** is equal to the input `BatchLength`
    - **BatchWidth** is equal to the input `BatchWidth`
    - **ListSize** is equal to the input `ListSize`
    - **Height** is equal to the input `Height / block_size`
    - **Width** is equal to the input `Width / block_size`
    - **Depth** is equal to `1`
    - **Channels** is equal to the input `Channels * block_size * block_size`
    """

    def __init__(self, input_layer, block_size=1, name=None):
        if type(input_layer) is PythonWrapper.SpaceToDepth:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if block_size < 1:
            raise ValueError('`block_size` must be < 0.')

        internal = PythonWrapper.SpaceToDepth(str(name), layers[0], int(outputs[0]), block_size)
        super().__init__(internal)

    @property
    def block_size(self):
        """Gets the block size.
        """
        return self._internal.get_block_size()

    @block_size.setter
    def block_size(self, new_block_size):
        """Sets the block size.
        """
        if new_block_size < 1:
            raise ValueError('`block_size` must be < 0.')
        self._internal.set_block_size(int(new_block_size))

# ----------------------------------------------------------------------------------------------------------------------


class DepthToSpace(Layer):
    """The layer that transforms each pixel (`1 x 1 x Ch`) of 2-dimensional
    images into square blocks of size `k x k x Ch/(k*k)`.
    The elements of pixel are interpreted as an image of size `k x k x Ch/(k*k)` in channel-last ordering.
    As a result `H x W x Ch` image is transformed into `H*k x W*k x Ch/(k*k)` image.

    This operation is the inverse function of SpaceToDepth.

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: object, tuple(object, int) or list of them
    :param block_size: The size of the block (`k` from the formula).
    :type block_size: int, default=1
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    Has single input, of the dimensions:

    - **BatchLength** * **BatchWidth** * **ListSize** is equal to the number of images
    - **Height** is the image height
    - **Width** is the image width
    - **Depth** is equal to `1`
    - **Channels** is the number of channels in the image format; should be a multiple of `block_size * block_size`

    .. rubric:: Layer outputs:

    The layer has single output, of the dimensions:

    - **BatchLength** is equal to the input `BatchLength`
    - **BatchWidth** is equal to the input `BatchWidth`
    - **ListSize** is equal to the input `ListSize`
    - **Height** is equal to the input `Height * block_size`
    - **Width** is equal to the input `Width * block_size`
    - **Depth** is equal to `1`
    - **Channels** is equal to the input `Channels / ( block_size * block_size )`
    """

    def __init__(self, input_layer, block_size=1, name=None):
        if type(input_layer) is PythonWrapper.DepthToSpace:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if block_size < 1:
            raise ValueError('`block_size` must be < 0.')

        internal = PythonWrapper.DepthToSpace(str(name), layers[0], int(outputs[0]), block_size)
        super().__init__(internal)

    @property
    def block_size(self):
        """Gets the block size.
        """
        return self._internal.get_block_size()

    @block_size.setter
    def block_size(self, new_block_size):
        """Sets the block size.
        """
        if new_block_size < 1:
            raise ValueError('`block_size` must be < 0.')
        self._internal.set_block_size(int(new_block_size))
