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
import numpy as np


class SplitChannels(Layer):
    """The layer that splits an input blob along the Channels dimension.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param sizes: The sizes of the first one, two, or three parts. 
        The final part size is what's left.  
    :type sizes: array of int, up to 3 elements
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with input data.
        The dimensions:

        - **Channels** should not be less than the sum of sizes array elements.
    
    .. rubric:: Layer outputs:

    The layer has at least len(sizes) outputs.
    The dimensions:

    - **Channels** equals the corresponding element of sizes array, 
      for the last output it is input **Channels** minus the sum of sizes
    - all other dimensions are the same as for the input
    """
    def __init__(self, input_layer, sizes, name=None):
        if type(input_layer) is PythonWrapper.SplitChannels:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        s = np.asarray(sizes, dtype=np.int32)

        if s.size > 3:
            raise ValueError('The `sizes` must contain not more than 3 elements.')

        if np.any(s < 0):
            raise ValueError('The `sizes` must contain only positive values.')

        internal = PythonWrapper.SplitChannels(str(name), layers[0], int(outputs[0]), s)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class SplitDepth(Layer):
    """The layer that splits an input blob along the Depth dimension.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param sizes: The sizes of the first one, two, or three parts. 
        The final part size is what's left.  
    :type sizes: array of int, up to 3 elements
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with input data.
        The dimensions:

        - **Depth** should not be less than the sum of sizes array elements.
    
    .. rubric:: Layer outputs:

    The layer has at least len(sizes) outputs.
    The dimensions:

        - **Depth** equals the corresponding element of sizes array, 
          for the last output it is input **Depth** minus the sum of sizes
        - all other dimensions are the same as for the input
    """
    def __init__(self, input_layer, sizes, name=None):
        if type(input_layer) is PythonWrapper.SplitDepth:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        s = np.asarray(sizes, dtype=np.int32)

        if s.size > 3:
            raise ValueError('The `sizes` must contain not more than 3 elements.')

        if np.any(s < 0):
            raise ValueError('The `sizes` must contain only positive values.')

        internal = PythonWrapper.SplitDepth(str(name), layers[0], int(outputs[0]), s)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class SplitWidth(Layer):
    """The layer that splits an input blob along the Width dimension.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param sizes: The sizes of the first one, two, or three parts. 
        The final part size is what's left.  
    :type sizes: array of int, up to 3 elements
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with input data.
        The dimensions:

        - **Width** should not be less than the sum of sizes array elements.
    
    .. rubric:: Layer outputs:

    The layer has at least len(sizes) outputs.
    The dimensions:

        - **Width** equals the corresponding element of sizes array, 
          for the last output it is input **Width** minus the sum of sizes
        - all other dimensions are the same as for the input
    """
    def __init__(self, input_layer, sizes, name=None):
        if type(input_layer) is PythonWrapper.SplitWidth:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        s = np.asarray(sizes, dtype=np.int32)

        if s.size > 3:
            raise ValueError('The `sizes` must contain not more than 3 elements.')

        if np.any(s < 0):
            raise ValueError('The `sizes` must contain only positive values.')

        internal = PythonWrapper.SplitWidth(str(name), layers[0], int(outputs[0]), s)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class SplitHeight(Layer):
    """The layer that splits an input blob along the Height dimension.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param sizes: The sizes of the first one, two, or three parts. 
        The final part size is what's left.  
    :type sizes: array of int, up to 3 elements
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with input data.
        The dimensions:

        - **Height** should not be less than the sum of sizes array elements.
    
    .. rubric:: Layer outputs:

    The layer has at least len(sizes) outputs.
    The dimensions:

        - **Height** equals the corresponding element of sizes array, 
          for the last output it is input **Height** minus the sum of sizes
        - all other dimensions are the same as for the input
    """
    def __init__(self, input_layer, sizes, name=None):
        if type(input_layer) is PythonWrapper.SplitHeight:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        s = np.asarray(sizes, dtype=np.int32)

        if s.size > 3:
            raise ValueError('The `sizes` must contain not more than 3 elements.')

        if np.any(s < 0):
            raise ValueError('The `sizes` must contain only positive values.')

        internal = PythonWrapper.SplitHeight(str(name), layers[0], int(outputs[0]), s)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class SplitListSize(Layer):
    """The layer that splits an input blob along the ListSize dimension.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param sizes: The sizes of the first one, two, or three parts. 
        The final part size is what's left.
    :type sizes: array of int, up to 3 elements
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with input data.
        The dimensions:

        - **ListSize** should not be less than the sum of sizes array elements.
    
    .. rubric:: Layer outputs:

    The layer has at least len(sizes) outputs.
    The dimensions:

        - **ListSize** equals the corresponding element of sizes array, 
          for the last output it is input **ListSize** minus the sum of sizes
        - all other dimensions are the same as for the input
    """
    def __init__(self, input_layer, sizes, name=None):
        if type(input_layer) is PythonWrapper.SplitListSize:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        s = np.asarray(sizes, dtype=np.int32)

        if s.size > 3:
            raise ValueError('The `sizes` must contain not more than 3 elements.')

        if np.any(s < 0):
            raise ValueError('The `sizes` must contain only positive values.')

        internal = PythonWrapper.SplitListSize(str(name), layers[0], int(outputs[0]), s)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class SplitBatchWidth(Layer):
    """The layer that splits an input blob along the BatchWidth dimension.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param sizes: The sizes of the first one, two, or three parts. 
        The final part size is what's left.  
    :type sizes: array of int, up to 3 elements
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with input data.
        The dimensions:

        - **BatchWidth** should not be less than the sum of sizes array elements.
    
    .. rubric:: Layer outputs:

    The layer has at least len(sizes) outputs.
    The dimensions:

        - **BatchWidth** equals the corresponding element of sizes array, 
          for the last output it is input **BatchWidth** minus the sum of sizes
        - all other dimensions are the same as for the input
    """
    def __init__(self, input_layer, sizes, name=None):
        if type(input_layer) is PythonWrapper.SplitBatchWidth:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        s = np.asarray(sizes, dtype=np.int32)

        if s.size > 3:
            raise ValueError('The `sizes` must contain not more than 3 elements.')

        if np.any(s < 0):
            raise ValueError('The `sizes` must contain only positive values.')

        internal = PythonWrapper.SplitBatchWidth(str(name), layers[0], int(outputs[0]), s)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class SplitBatchLength(Layer):
    """The layer that splits an input blob along the BatchLength dimension.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param sizes: The sizes of the first one, two, or three parts. 
        The final part size is what's left.
    :type sizes: array of int, up to 3 elements
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with input data.
        The dimensions:

        - **BatchLength** should not be less than the sum of sizes array elements.
    
    .. rubric:: Layer outputs:

    The layer has at least len(sizes) outputs.
    The dimensions:

        - **BatchLength** equals the corresponding element of sizes array, 
          for the last output it is input **BatchLength** minus the sum of sizes
        - all other dimensions are the same as for the input
    """
    def __init__(self, input_layer, sizes, name=None):
        if type(input_layer) is PythonWrapper.SplitBatchLength:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        s = np.asarray(sizes, dtype=np.int32)

        if s.size > 3:
            raise ValueError('The `sizes` must contain not more than 3 elements.')

        if np.any(s < 0):
            raise ValueError('The `sizes` must contain only positive values.')

        internal = PythonWrapper.SplitBatchLength(str(name), layers[0], int(outputs[0]), s)
        super().__init__(internal)
