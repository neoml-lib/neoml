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
--------------------------------------------------------------------------------------------------------------
"""

import neoml.PythonWrapper as PythonWrapper
from .Dnn import Layer
from neoml.Utils import check_input_layers


class EltwiseSum(Layer):
    """The layer that adds up its inputs element by element.

    Layer inputs
    -----------
    The layer should have at least two inputs. All inputs should be the same size.
    
    Layer outputs
    -----------
    The layer has one output of the same size as the inputs.
    
    Parameters
    -----------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.EltwiseSum:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.EltwiseSum(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class EltwiseMul(Layer):
    """The layer that multiplies its inputs element by element.

    Layer inputs
    -----------
    The layer should have at least two inputs. All inputs should be the same size.
    
    Layer outputs
    -----------
    The layer has one output of the same size as the inputs.
    
    Parameters
    -----------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.EltwiseMul:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.EltwiseMul(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class EltwiseNegMul(Layer):
    """The layer that that calculates the element-wise product of `1 - x`,
    where `x` is the element of the first input, 
    and the corresponding elements of all other inputs.

    Layer inputs
    -----------
    The layer should have at least two inputs. All inputs should be the same size.
    
    Layer outputs
    -----------
    The layer has one output of the same size as the inputs.
    
    Parameters
    -----------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.EltwiseNegMul:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.EltwiseNegMul(str(name), layers, outputs)
        super().__init__(internal)

# ----------------------------------------------------------------------------------------------------------------------


class EltwiseMax(Layer):
    """The layer that finds the maximum among the elements that are at the same position in all input blobs.

    Layer inputs
    -----------
    The layer should have at least two inputs. All inputs should be the same size.
    
    Layer outputs
    -----------
    The layer has one output of the same size as the inputs.
    
    Parameters
    -----------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.EltwiseMax:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 0)

        internal = PythonWrapper.EltwiseMax(str(name), layers, outputs)
        super().__init__(internal)
