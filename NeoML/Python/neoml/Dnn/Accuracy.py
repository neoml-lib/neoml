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


class Accuracy(Layer):
    """The layer that calculates classification accuracy, that is,
    the proportion of objects classified correctly in the set.

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param reset: Specifies if the statistics should be reset with each run.
        Set to False to accumulate statistics for subsequent runs.
    :type reset: bool, default=True
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with the network response
    The dimensions:
    - **BatchLength** * **BatchWidth** * **ListSize** equal to the number of objects 
        that were classified
    - **Height**, **Width**, **Depth** equal to 1
    - **Channels** equal to 1 for binary classification and to the number of classes 
        if there are more than 2
    
    (2) a blob with the correct class labels
    The dimensions should be the same as for the first input
    
    .. rubric:: Layer outputs:

    (1) a blob with only one element, which contains the proportion of
    correctly classified objects among all objects
    """

    def __init__(self, input_layers, reset=True, name=None):

        if type(input_layers) is PythonWrapper.Accuracy:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 2)

        internal = PythonWrapper.Accuracy(str(name), layers[0], layers[1], int(outputs[0]), int(outputs[1]), bool(reset))
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

# ----------------------------------------------------------------------------------------------------------------------


class ConfusionMatrix(Layer):
    """The layer that calculates the confusion matrix for classification
    results.
    The columns correspond to the network response, the rows - to 
    the correct labels. Each element of the matrix contains the number 
    of objects that belong to the "row" class and were classified 
    as the "column" class.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param reset: Specifies if the statistics should be reset with each run.
        Set to False to accumulate statistics for subsequent runs.
    :type reset: bool, default=True
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with the network response
    The dimensions:
    - **BatchLength** * **BatchWidth** * **ListSize** equal to the number of objects 
        that were classified
    - **Height**, **Width**, **Depth** equal to 1
    - **Channels** equal to the number of classes and should be greater than 1
    
    (2) a blob with the correct class labels
    The dimensions should be the same as for the first input
    
    .. rubric:: Layer outputs:

    (1) the confusion matrix.
    The dimensions:
    - **BatchLength**, **BatchWidth**, **ListSize**, **Depth**, **Channels** are 1
    - **Height** and **Width** are equal to the input **Channels**
    """

    def __init__(self, input_layers, reset=True, name=None):

        if type(input_layers) is PythonWrapper.ConfusionMatrix:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 2)

        internal = PythonWrapper.ConfusionMatrix(str(name), layers[0], layers[1], int(outputs[0]), int(outputs[1]), bool(reset))
        super().__init__(internal)

    @property
    def reset(self):
        """Checks if the calculations will be reset on each run.
        """
        return self._internal.get_reset()

    @reset.setter
    def reset(self, reset):
        """Specifies if the calculations should be reset on each run.
        """
        self._internal.set_reset(bool(reset))

    @property
    def matrix(self):
        """Gets the confusion matrix. The dimensions:
        - **BatchLength**, **BatchWidth**, **ListSize**, **Depth**, **Channels** are 1
        - **Height** and **Width** are equal to the input **Channels**
        """
        return self._internal.get_matrix()

    def reset_matrix(self):
        """Resets the confusion matrix values. The dimensions:
        - **BatchLength**, **BatchWidth**, **ListSize**, **Depth**, **Channels** are 1
        - **Height** and **Width** are equal to the input **Channels**
        """
        return self._internal.reset_matrix()
