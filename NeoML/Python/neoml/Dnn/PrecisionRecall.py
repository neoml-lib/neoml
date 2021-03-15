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


class PrecisionRecall(Layer):
    """The layer that calculates the number of objects classified
    correctly for either class in a binary classification scenario.
    Using these statistics, you can easily calculate the precision 
    and recall for the trained network.
    
    Layer inputs
    ----------
    #1: the network response.
    The dimensions:
    - BatchLength * BatchWidth * ListSize is the number of objects classified
    - Height, Width, Depth, Channels are 1
   
    #2: the correct class labels (1 or -1).
    The dimensions are the same as for the first input.
   
    Layer outputs
    ----------
    #1: the 4-element array along the Channels dimension that contains:
    0 - the number of correctly classified objects in class 1
    1 - the total number of objects in class 1
    2 - the number of correctly classified objects in class -1
    3 - the total number of objects in class -1
    
    Parameters
    ----------
    input_layers : array of (object, int) tuples and objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
    reset : bool, default=True
        Specifies if the statistics should be reset with each run.
        Set to False to accumulate statistics for subsequent runs.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layers, reset=True, name=None):

        if type(input_layers) is PythonWrapper.PrecisionRecall:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 2)

        internal = PythonWrapper.PrecisionRecall(str(name), layers[0], layers[1], int(outputs[0]), int(outputs[1]), bool(reset))
        super().__init__(internal)

    @property
    def reset(self):
        """Checks if the statistics will be reset after each run.
        """
        return self._internal.get_reset()

    @reset.setter
    def reset(self, reset):
        """Specifies if the statistics should be reset after each run.
        """
        self._internal.set_reset(bool(reset))

    @property
    def result(self):
        """Returns the result 4-element array, 
            in the same order as the output blob.
        """
        return self._internal.get_result()
