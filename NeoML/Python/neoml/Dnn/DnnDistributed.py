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

import uuid
import os
from neoml.MathEngine import MathEngine
import neoml.PythonWrapper as PythonWrapper


class DnnDistributed(PythonWrapper.DnnDistributed):
    """Single process, multiple threads distributed training.
    
    :param dnn: The dnn to learn distributed.
    :type dnn: neoml.Dnn or str
    :param type: Learn on cpu or gpu.
    :type type: str, ("cpu", "gpu"), default="cpu"
    :param count: Count of models to use.
    :type count: int, default=0
    :param devs: Numbers of gpus to use
    :type devs: list, default=None
    :param path: The archive filename using for internal purposes.
    :type path: str, default="distributed.arch"
    """
    def __init__(self, dnn, type='cpu', count=0, devs=None):
        is_dnn = isinstance(dnn, neoml.Dnn)
        path = str(uuid.uuid4()) if is_dnn else dnn
        if type == 'cpu':
            if count < 1:
                raise ValueError('`count` must be a positive number.')
            if is_dnn:
                dnn.store(path)
            super().__init__(path, count)
        elif type == 'cuda':
            if devs is None:
                if count < 1:
                    raise ValueError('`devs` or `count` must be specified.')
                devs = list(range(count))
            if is_dnn:
                dnn.store(path)
            super().__init__(path, devs)
        else:
            raise ValueError('`type` must be one of: "cpu", "cuda".')
        if is_dnn:
            os.remove(path)

    def run(self, set_data):
        """Runs the network.

        :param set_data: A callback that takes a math_engine and thread number
            as an argument. It must return a dictionary of input blobs for the
            dnn on a given thread, this dictionary must be the same as for the
            learn method of a dnn.
        :type set_data: callable
        """
        self._run(set_data)

    def run_and_backward(self, set_data):
        """Runs the network and performs a backward pass with the input data.

        :param set_data: A callback that takes a math_engine and thread number
            as an argument. It must return a dictionary of input blobs for the
            dnn on a given thread, this dictionary must be the same as for the
            learn method of a dnn.
        :type set_data: callable
        """
        self._run_and_backward(set_data)

    def learn(self, set_data):
        """Runs the network, performs a backward pass 
        and updates the trainable weights.

        :param set_data: A callback that takes a math_engine and thread number
            as an argument. It must return a dictionary of input blobs for the
            dnn on a given thread, this dictionary must be the same as for the
            learn method of a dnn.
        :type set_data: callable
        """
        self._learn(set_data)

    def train(self):
        """Updates the trainable weights of all models (after run_and_backward).
        """
        self._train()

    def last_losses(self, layer_name):
        """Gets values of the loss function on the last step for all models.

        :param layer_name: The name of the loss layer for which last losses will
            be returned. The class of the layer with that name must be `Loss`.
        :type layer_name: str
        """
        return self._last_losses(layer_name)

    def get_output(self, layer_name):
        """Returns last blobs of `layer_name` for all models.
        """
        return self._get_output(layer_name)

    def save(self, path):
        """Serializes the trained network.

        :param path: The full path to the location where the network should be stored.
        :type path: str
        """
        return self._save(path)

    @solver.setter
    def solver(self, new_solver):
        """Sets the optimizer for the layer's trainable parameters.
        """
        self.set_solver(new_solver._internal)

    @property
    def get_model_count(self):
        """Gets the number of models in disitrbuted traning.
        """
        return self._get_model_count()