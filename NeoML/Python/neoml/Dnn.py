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


class Dnn(PythonWrapper.Dnn):
    """
    """
    def __init__(self, math_engine, random=None, path=None):
        if random is None:
            random = neoml.Random.Random(42)

        if not isinstance(random, neoml.Random.Random):
            raise ValueError('The `random_generator` must be a neoml.Random.Random.')

        if path is None:
            super().__init__(random, math_engine._internal)
        else:
            super().__init__(random, math_engine._internal, path)

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

    def run(self, inputs):
        """Runs the network.
        Parameters
        ----------
        inputs : The dictionary of input blobs.

        Returns
        -------
            The list of output blobs.
        """
        dnn_inputs = super()._get_inputs()

        if len(inputs) < len(dnn_inputs):
            raise ValueError('The `inputs` contains less layers than the dnn.')

        if len(inputs) > len(dnn_inputs):
            raise ValueError('The `inputs` contains more layers than the dnn.')

        input_list = []
        for layer_name in dnn_inputs:
            input_list.append(inputs[layer_name].internal)

        return self._run(input_list)

    def run_and_backward(self, inputs):
        """
        """
        dnn_inputs = self._get_inputs()

        if len(inputs) < len(dnn_inputs):
            raise ValueError('The `inputs` contains less layers than the dnn.')

        if len(inputs) > len(dnn_inputs):
            raise ValueError('The `inputs` contains more layers than the dnn.')

        input_list = []
        for layer_name in dnn_inputs:
            input_list.append(inputs[layer_name].internal)

        return self._run_and_backward(input_list)

    def learn(self, inputs):
        """
        """
        dnn_inputs = self._get_inputs()

        if len(inputs) < len(dnn_inputs):
            raise ValueError('The `inputs` contains less layers than the dnn.')

        if len(inputs) > len(dnn_inputs):
            raise ValueError('The `inputs` contains more layers than the dnn.')

        input_list = []
        for layer_name in dnn_inputs:
            input_list.append(inputs[layer_name].internal)

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
