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
import numpy

class Transform(Layer):
    """The layer that changes the input blob dimensions without 
    moving any of the data. The total number of elements in the blob 
    stays the same, and therefore the product of all dimensions 
    should not be changed by the transformation.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected. 
    :type input_layer: object, tuple(object, int)
    :param transforms: Specifies the transformation to be made to each of the 7 dimensions:

        - "set" its length to the int value
        - "multiply" by the int value
        - "divide" by the int value
        - "remainder" may only be set for one dimension; it will be set 
          so that the total size of the blob stays the same
    :type transforms: array of 7 tuples (operation, value), 
        operation: one of "remainder", "set", "multiply", "divide"
        value: int > 0
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of any size.
    
    .. rubric:: Layer outputs:

    (1) a blob of the dimensions determined by the rules:

        - the dimensions in "set" mode will be equal to the specified value
        - the dimensions in "multiply" mode will be value times larger
        - the dimensions in "divide" mode will be value times smaller
        - the dimension in "remainder" mode will be such that 
          the total size of the input and the output are the same
    """

    rules = ["remainder", "set", "multiply", "divide"]

    def __init__(self, input_layer, transforms, name=None):

        if type(input_layer) is PythonWrapper.Transform:
            super().__init__(input_layer)
            return
 
        layers, outputs = check_input_layers(input_layer, 1)

        if len(transforms) != 7:
            raise ValueError('The `transforms` array must have 7 elements.')

        operations = numpy.ones(7, numpy.int32)
        parameters = numpy.ones(7, numpy.int32)
        for i in range(len(transforms)):
            if len(transforms[i]) != 2:
                raise ValueError('The `transforms` array must contain pairs (operation, value).')

            operations[i] = self.rules.index(transforms[i][0])
            if transforms[i][1] < 0:
                raise ValueError('All values in `transforms` must be >= 0.')
            parameters[i] = transforms[i][1]

        internal = PythonWrapper.Transform(str(name), layers[0], int(outputs[0]), operations, parameters)
        super().__init__(internal)

    @property
    def transforms(self):
        """Gets the array of transformations.
        """
        operations = self._internal.get_operations()
        parameters = self._internal.get_parameters()

        result = []
        for i in range(operations.size):
            result.append((self.rules[operations[i]], parameters[i]))
        return result

    @transforms.setter
    def transforms(self, transforms):
        """Sets the array of transformations.
        """
        if len(transforms) != 7:
            raise ValueError('The `transforms` array must have 7 elements.')

        operations = numpy.ones(7, numpy.int32)
        parameters = numpy.ones(7, numpy.int32)
        for i in range(len(transforms)):
            if len(transforms[i]) != 2:
                raise ValueError('The `transforms` array must contain pairs (operation, value).')

            operations[i] = self.rules.index(transforms[i][0])
            if transforms[i][1] < 0:
                raise ValueError('All values in `transforms` must be >= 0.')
            parameters[i] = transforms[i][1]

        self._internal.set_transforms(operations, parameters)
