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
import neoml.Dnn as Dnn
import neoml.Utils as Utils
import neoml.Initializer as Initializer


class MultichannelLookup(Dnn.Layer):
    """
    """
    def __init__(self, input_layers, dimensions=None, name=None):

        if dimensions is None:
            dimensions = [(1, 1)]
        if type(input_layers) is PythonWrapper.MultichannelLookup:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, 0)

        internal = PythonWrapper.MultichannelLookup(str(name), layers, outputs, dimensions)
        super().__init__(internal)

    def initialize(self, initializer):
        if initializer is None:
            return self._internal.clear()
    
        if not isinstance(initializer, Initializer.Initializer):
            raise ValueError('The `initializer` must be an Initializer.')

        self._internal.initialize(initializer._internal)
