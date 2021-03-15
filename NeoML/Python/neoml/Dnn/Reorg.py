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
from .Utils import check_input_layers


class Reorg(Layer):
    """The layer that transforms a set of two-dimensional multi-channel images
    into a set of images of smaller size but with more channels.
    This operation is used in YOLO architecture: https://pjreddie.com/darknet/yolo/
    
    Layer inputs
    ----------
    #1: a blob with images.
    The dimensions:
    - BatchLength * BatchWidth * ListSize is the number of images
    - Height is the image height; should be a multiple of stride
    - Width is the image width; should be a multiple of stride
    - Depth is 1
    - Channels is the number of channels the image format uses
    
    Layer outputs
    ----------
    #1: the result of image transformation.
    The dimensions:
    - BatchLength, BatchWidth, ListSize are the same as for the input
    - Height is input Height / stride
    - Width is input Width / stride
    - Depth is 1
    - Channels is input Channels * stride^2
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    stride : int, default=1
        The value by which the image size will be divided in the final result.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, stride=1, name=None):

        if type(input_layer) is PythonWrapper.Reorg:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if stride < 1:
            raise ValueError('The `stride` must be > 0.')

        internal = PythonWrapper.Reorg(str(name), layers[0], int(outputs[0]), int(stride))
        super().__init__(internal)

    @property
    def stride(self):
        """Gets the divider value.
        """
        return self._internal.get_stride()

    @stride.setter
    def stride(self, stride):
        """Sets the divider value.
        """
        if stride < 1:
            raise ValueError('The `stride` must be > 0.')

        self._internal.set_stride(int(stride))
