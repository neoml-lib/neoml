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
import neoml.Random


# -------------------------------------------------------------------------------------------------------------


class Dnn(PythonWrapper.Dnn):
    """
    """
    def __init__(self, math_engine, random=None):
        if not isinstance(math_engine, neoml.MathEngine.MathEngine):
            raise ValueError('The `math_engine` must be a neoml.MathEngine.MathEngine.')

        if random is None:
            random = neoml.Random.Random(42)

        if not isinstance(random, neoml.Random.Random):
            raise ValueError('The `random_generator` must be a neoml.Random.Random.')

        super().__init__(random, math_engine._internal)

    def store(self, path):
        """
        """
        self._store(str(path))

    def load(self, path):
        """
        """
        self._load(str(path))

    def store_checkpoint(self, path):
        """
        """
        self._store_checkpoint(str(path))

    def load_checkpoint(self, path):
        """
        """
        self._load_checkpoint(str(path))

    @property
    def math_engine(self):
        """
        """
        return self.get_math_engine()

    @property
    def solver(self):
        """
        """
        return self.get_solver()

    @solver.setter
    def solver(self, new_solver):
        """
        """
        self.set_solver(new_solver._internal)

    @property
    def initializer(self):
        """
        """
        return self.get_initializer()

    @initializer.setter
    def initializer(self, new_initializer):
        """
        """
        self.set_initializer(new_initializer._internal)

    @property
    def input_layers(self):
        """
        """
        return self.get_inputs()

    @property
    def output_layers(self):
        """
        """
        return self.get_outputs()

    @property
    def layers(self):
        """
        """
        return self.get_layers()

    def add_layer(self, layer):
        """
        """
        if type(layer) is not Layer:
            raise ValueError('The `layer` is expected to be neoml.Dnn.Layer`')
        self._add_layer(layer.internal)
    
    def delete_layer(self, layer):
        if type(layer) is str:
            self._delete_layer(layer)
        elif type(layer) is Layer:
            self._delete_layer(layer.name)
        else:
            raise ValueError('The `layer` is expected to be `str` or `neoml.Dnn.Layer`')

    def run(self, inputs):
        """Runs the network.
        Parameters
        ----------
        inputs : The dictionary of input blobs.

        Returns
        -------
            The list of output blobs.
        """
        dnn_inputs = self.get_inputs()

        if len(inputs) < len(dnn_inputs):
            raise ValueError('The `inputs` contains less layers than the dnn.')

        if len(inputs) > len(dnn_inputs):
            raise ValueError('The `inputs` contains more layers than the dnn.')

        input_list = []
        for layer_name in dnn_inputs:
            input_list.append(inputs[layer_name]._internal)

        dnn_outputs = self._run(input_list)

        outputs = {}
        for layer_name in dnn_outputs:
            outputs[layer_name] = neoml.Blob.Blob(dnn_outputs[layer_name])

        return outputs

    def run_and_backward(self, inputs):
        """
        """
        dnn_inputs = self.get_inputs()

        if len(inputs) < len(dnn_inputs):
            raise ValueError('The `inputs` contains less layers than the dnn.')

        if len(inputs) > len(dnn_inputs):
            raise ValueError('The `inputs` contains more layers than the dnn.')

        input_list = []
        for layer_name in dnn_inputs:
            input_list.append(inputs[layer_name]._internal)

        return self._run_and_backward(input_list)

    def learn(self, inputs):
        """
        """
        dnn_inputs = self.get_inputs()

        if len(inputs) < len(dnn_inputs):
            raise ValueError('The `inputs` contains less layers than the dnn.')

        if len(inputs) > len(dnn_inputs):
            raise ValueError('The `inputs` contains more layers than the dnn.')

        input_list = []
        for layer_name in dnn_inputs:
            input_list.append(inputs[layer_name]._internal)

        self._learn(input_list)

# -------------------------------------------------------------------------------------------------------------


class Layer:
    """
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.Layer):
            raise ValueError('The `internal` must be PythonWrapper.Layer')

        self._internal = internal

    @property
    def name(self):
        """
        """
        return self._internal.get_name()
    
    def connect(self, layer, output_index=0, input_index=0):
        if not isinstance(layer, Layer):
            raise ValueError('The `layer` is expected to be neoml.Dnn.Layer')
        if output_index < 0:
            raise ValueError('The `output_index` must be >= 0')
        if input_index < 0:
            raise ValueError('The `input_index` must be >= 0')
        self._internal.connect(layer._internal, output_index, input_index)
