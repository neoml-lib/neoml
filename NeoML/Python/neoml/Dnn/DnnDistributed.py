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
    :type dnn: neoml.Dnn
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
        path = str(uuid.uuid4())
        if type == 'cpu':
            if count < 1:
                raise ValueError('`count` must be a positive number.')
            dnn.store(path)
            super().__init__(path, count)
        elif type == 'cuda':
            if devs is None:
                if count < 1:
                    raise ValueError('`devs` or `count` must be specified.')
                devs = list(range(count))
            dnn.store(path)
            super().__init__(path, devs)
        else:
            raise ValueError('`type` must be one of: "cpu", "cuda".')
        os.remove(path)

    def learn(self, set_data):
        """Performs one iteration of learning.

        :param set_data: A callback that takes a math_engine and thread number
            as an argument. It must return a dictionary of input blobs for the
            dnn on a given thread, this dictionary must be the same as for the
            learn method of a dnn.
        :type set_data: callable
        """
        self._learn(set_data)

    def last_losses(self, layer_name):
        """Gets values of the loss function on the last step for all models.

        :param layer_name: The name of the loss layer for which last losses will
            be returned. The class of the layer with that name must be `Loss`.
        :type layer_name: str
        """
        return self._last_losses(layer_name)