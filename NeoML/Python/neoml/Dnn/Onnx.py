""" Copyright (c) 2017-2023 ABBYY Production LLC

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


class OnnxTranspose(Layer):
    """Special transpose helper layer which is used during import from ONNX
    Read-only for Python
    """

    dimensions = ["batch_length", "batch_width", "list_size", "height", "width", "depth", "channels"]

    def __init__(self, base_layer):
        if type(base_layer) is not PythonWrapper.OnnxTranspose:
            raise ValueError('`base_layer` must be PythonWrapper.OnnxTranspose')
        super().__init__(base_layer)

    @property
    def first_dim(self):
        """Gets the first dimension to be switched
        """
        return self.dimensions[self._internal.get_first_dim()]

    @property
    def second_dim(self):
        """Gets the second dimension to be switched
        """
        return self.dimensions[self._internal.get_second_dim()]


class OnnxTransform(Layer):
    """Special transpose helper layer which is used during import from ONNX
    Read-only for Python
    """

    dimensions = ["batch_length", "batch_width", "list_size", "height", "width", "depth", "channels", "NOT_SET"]

    def __init__(self, base_layer):
        if type(base_layer) is not PythonWrapper.OnnxTransform:
            raise ValueError('`base_layer` must be PythonWrapper.OnnxTransform')
        super().__init__(base_layer)

    @property
    def rules(self):
        """Gets the rules of this transform layer
        """
        return [self.dimensions[i] for i in self._internal.get_rules()]
