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
from neoml.Blob import Blob
from neoml.Dnn import Dnn


class DnnDistributed(PythonWrapper.DnnDistributed):
    """Single process, multiple threads distributed training.
    
    :param dnn: The dnn or the archive with the dnn to learn distributed.
    :type dnn: neoml.Dnn.Dnn or str
    :param type: Learn on cpu or gpu.
    :type type: str, ("cpu", "gpu"), default="cpu"
    :param count: Count of models to use.
    :type count: int, default=0
    :param devs: Numbers of gpus to use
    :type devs: list, default=None
    :param initializer: The initializer that will fill initial weight values.
    :type path: str, ("xavier", "xavier_uniform", "uniform"), default="xavier"
    :param seed: Random seed number.
    :type seed: int, default=42
    """
    def __init__(self, dnn, type='cpu', count=0, devs=None, initializer='xavier', seed=42):
        if not isinstance(dnn, Dnn) and not isinstance(dnn, str):
            raise ValueError('`dnn` must be neoml.Dnn.Dnn or str.')
        if initializer not in ('xavier', 'xavier_uniform', 'uniform'):
            raise ValueError('`initializer` must be one of: "xavier", "xavier_uniform", "uniform".')
        if type == 'cpu':
            if count < 1:
                raise ValueError('`count` must be a positive number.')
            super().__init__(dnn, count, initializer, seed)
        elif type == 'cuda':
            if devs is None:
                if count < 1:
                    raise ValueError('`devs` or `count` must be specified.')
                devs = list(range(count))
            super().__init__(dnn, devs, initializer, seed)
        else:
            raise ValueError('`type` must be one of: "cpu", "cuda".')

    def run(self, set_data_callback):
        """Runs the network.

        :param set_data_callback: A callback that takes a math_engine and thread number
            as its arguments. It must return a tuple(int, dict) where
            int is the size of this batch and dict contains input blobs for the
            dnn on a given thread, this dict must be the same as for the
            run method of a dnn. Zero batch size (no data for this thread during this run)
            is supported on any run except the first.
        :type set_data_callback: callable
        """
        self._run(set_data_callback)

    def run_and_backward(self, set_data_callback):
        """Runs the network and performs a backward pass with the input data.

        :param set_data_callback: A callback that takes a math_engine and thread number
            as its arguments. It must return a tuple(int, dict) where
            int is the size of this batch and dict contains input blobs for the
            dnn on a given thread, this dict must be the same as for the
            run_and_backward method of a dnn. Zero batch size
            (no data for this thread during this run)
            is supported on any run except the first.
        :type set_data_callback: callable
        """
        self._run_and_backward(set_data_callback)

    def learn(self, set_data_callback):
        """Runs the network, performs a backward pass 
        and updates the trainable weights.

        :param set_data_callback: A callback that takes a math_engine and thread number
            as its arguments. It must return a tuple(int, dict) where
            int is the size of this batch and dict contains input blobs for the
            dnn on a given thread, this dict must be the same as for the
            learn method of a dnn. Zero batch size (no data for this thread during this run)
            is supported on any run except the first.
        :type set_data_callback: callable
        """
        self._learn(set_data_callback)

    def train(self):
        """Updates the trainable weights of all models (after run_and_backward).
        """
        self._train()

    def last_losses(self, layer_name):
        """Gets values of the loss function on the last step for all models.

        :param layer_name: The name of the loss layer for which last losses will
            be returned. The class of the layer with that name must be `neoml.Dnn.Loss`.
        :type layer_name: str

        :return: The array of losses for all models.
        :rtype: *ndarray of shape (model_count,)*
        """
        return self._last_losses(str(layer_name))

    def get_output(self, layer_name):
        """Returns last blobs of `layer_name` for all models.

        :param layer_name: The name of the layer for which last output will be returned.
            `layer_name` should correspond to neoml.Sink.
        :type layer_name: str

        :return: The list of output blobs for all models. Default cpu math engine is used for blobs.
        :rtype: *list of size model_count*
        """
        return [Blob(blob) for blob in self._get_output(str(layer_name))]

    def save(self, path):
        """Serializes the trained network.

        :param path: The full path to the location where the network should be stored.
        :type path: str
        """
        return self._save(str(path))

    def set_solver(self, path):
        """Sets the optimizer for the layer's trainable parameters.

        :param path: The full path to the location where the solver should be stored.
        :type path: str
        """
        self._set_solver(str(path))

    @property
    def model_count(self):
        """Gets the number of models in distributed traning.
        """
        return self._get_model_count()