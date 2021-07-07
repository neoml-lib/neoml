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


class ConcatChannels(Layer):
    """The layer that concatenates several blobs into one
    along the Channels dimension.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:
    
    The layer accepts an arbitrary number of inputs.
    The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize**, 
      **Height**, **Width**, **Depth** equal for all inputs
    - **Channels** dimension may vary
    
    .. rubric:: Layer outputs:

    (1) a blob with the result of concatenation.
        The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize**, 
      **Height**, **Width**, **Depth** equal to the inputs' dimensions
    - **Channels** equal to the sum of all inputs' **Channels**
    """
    def __init__(self, input_layers, name=None):
        if type(input_layers) is PythonWrapper.ConcatChannels:
            super().__init__(input_layers)
            return

        if len(input_layers) > 32:
            raise ValueError('The `ConcatChannels` can merge no more than 32 blobs.')

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.ConcatChannels(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class ConcatDepth(Layer):
    """The layer that concatenates several blobs into one
    along the Depth dimension.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer accepts an arbitrary number of inputs.
    The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize**, 
      **Height**, **Width**, **Channels** equal for all inputs
    - **Depth** dimension may vary
    
    .. rubric:: Layer outputs:

    (1) a blob with the result of concatenation.
        The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize**, 
      **Height**, **Width**, **Channels** equal to the inputs' dimensions
    - **Depth** equal to the sum of all inputs' **Depth**
    """
    def __init__(self, input_layers, name=None):
        if type(input_layers) is PythonWrapper.ConcatDepth:
            super().__init__(input_layers)
            return

        if len(input_layers) > 32:
            raise ValueError('The `ConcatDepth` can merge no more than 32 blobs.')

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.ConcatDepth(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class ConcatWidth(Layer):
    """The layer that concatenates several blobs into one
    along the Width dimension.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer accepts an arbitrary number of inputs.
    The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize**, 
      **Height**, **Depth**, **Channels** equal for all inputs
    - **Width** dimension may vary
    
    .. rubric:: Layer outputs:

    (1) a blob with the result of concatenation.
        The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize**, 
      **Height**, **Depth**, **Channels** equal to the inputs' dimensions
    - **Width** equal to the sum of all inputs' **Width**
    """
    def __init__(self, input_layers, name=None):
        if type(input_layers) is PythonWrapper.ConcatWidth:
            super().__init__(input_layers)
            return

        if len(input_layers) > 32:
            raise ValueError('The `ConcatWidth` can merge no more than 32 blobs.')

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.ConcatWidth(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class ConcatHeight(Layer):
    """The layer that concatenates several blobs into one
    along the Height dimension.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer accepts an arbitrary number of inputs.
    The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize**, 
      **Width**, **Depth**, **Channels** equal for all inputs
    - **Height** dimension may vary
    
    .. rubric:: Layer outputs:

    (1) a blob with the result of concatenation.
        The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize**, 
      **Width**, **Depth**, **Channels** equal to the inputs' dimensions
    - **Height** equal to the sum of all inputs' **Height**
    """
    def __init__(self, input_layers, name=None):
        if type(input_layers) is PythonWrapper.ConcatHeight:
            super().__init__(input_layers)
            return

        if len(input_layers) > 32:
            raise ValueError('The `ConcatHeight` can merge no more than 32 blobs.')

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.ConcatHeight(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class ConcatBatchWidth(Layer):
    """The layer that concatenates several blobs into one
    along the BatchWidth dimension.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer accepts an arbitrary number of inputs.
    The dimensions:

    - **BatchLength**, **ListSize**, **Height**,
      **Width**, **Depth**, **Channels** equal for all inputs
    - **BatchWidth** dimension may vary
    
    .. rubric:: Layer outputs:

    (1) a blob with the result of concatenation.
        The dimensions:

    - **BatchLength**, **ListSize**, **Height**,
      **Width**, **Depth**, **Channels** equal to the inputs' dimensions
    - **BatchWidth** equal to the sum of all inputs' **BatchWidth**
    """
    def __init__(self, input_layers, name=None):
        if type(input_layers) is PythonWrapper.ConcatBatchWidth:
            super().__init__(input_layers)
            return

        if len(input_layers) > 32:
            raise ValueError('The `BatchWidth` can merge no more than 32 blobs.')

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.ConcatBatchWidth(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class ConcatBatchLength(Layer):
    """The layer that concatenates several blobs into one
    along the BatchLength dimension.

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer accepts an arbitrary number of inputs.
    The dimensions:

    - **BatchWidth**, **ListSize**, **Height**,
      **Width**, **Depth**, **Channels** equal for all inputs
    - **BatchLength** dimension may vary

    .. rubric:: Layer outputs:

    (1) a blob with the result of concatenation.
        The dimensions:

    - **BatchWidth**, **ListSize**, **Height**,
      **Width**, **Depth**, **Channels** equal to the inputs' dimensions
    - **BatchLength** equal to the sum of all inputs' **BatchLength**
    """
    def __init__(self, input_layers, name=None):
        if type(input_layers) is PythonWrapper.ConcatBatchLength:
            super().__init__(input_layers)
            return

        if len(input_layers) > 32:
            raise ValueError('The `BatchLength` can merge no more than 32 blobs.')

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.ConcatBatchLength(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class ConcatListSize(Layer):
    """The layer that concatenates several blobs into one
    along the ListSize dimension.

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer accepts an arbitrary number of inputs.
    The dimensions:

    - **BatchLength**, **BatchWidth**, **Height**,
      **Width**, **Depth**, **Channels** equal for all inputs
    - **ListSize** dimension may vary

    .. rubric:: Layer outputs:

    (1) a blob with the result of concatenation.
        The dimensions:

    - **BatchLength**, **BatchWidth**, **Height**,
      **Width**, **Depth**, **Channels** equal to the inputs' dimensions
    - **ListSize** equal to the sum of all inputs' **ListSize**
    """
    def __init__(self, input_layers, name=None):
        if type(input_layers) is PythonWrapper.ConcatListSize:
            super().__init__(input_layers)
            return

        if len(input_layers) > 32:
            raise ValueError('The `ListSize` can merge no more than 32 blobs.')

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.ConcatListSize(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class ConcatObject(Layer):
    """The layer that concatenates several blobs into one
    along the Height, Width, Depth, and Channels dimensions.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    The layer accepts an arbitrary number of inputs.
    The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize** equal for all inputs
    - **Height**, **Width**, **Depth**, **Channels** dimensions may vary
    
    .. rubric:: Layer outputs:

    (1) a blob with the result of concatenation.
        The dimensions:

    - **BatchLength**, **BatchWidth**, **ListSize** equal to the inputs' dimensions
    - **Height**, **Width**, **Depth** equal to 1
    - **Channels** equal to the sum of 
      **Height** * **Width** * **Depth** * **Channels** over all inputs
    """
    def __init__(self, input_layers, name=None):
        if type(input_layers) is PythonWrapper.ConcatObject:
            super().__init__(input_layers)
            return

        if len(input_layers) > 32:
            raise ValueError('The `ConcatObject` can merge no more than 32 blobs.')

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.ConcatObject(str(name), layers, outputs)
        super().__init__(internal)
