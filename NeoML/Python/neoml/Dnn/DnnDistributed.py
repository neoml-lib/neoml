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

class DnnDistributed(PythonWrapper.DnnDistributed):
    """Single process, multiple threads distributed training.
    
    :param dnn: .
    :type neoml.Dnn
    :param type: .
    :type random: , default=None 
    """
    def __init__(self, dnn, type='cpu', count=0, devs=None, path='distributed.arch'):
        if type == 'cpu':
            if count < 1:
                raise ValueError('Count must be a positive number.')
            dnn.store(path)
            super().__init__(path, count)
        elif type == 'cuda':
            if devs is None:
                raise ValueError('`devs` must be specified.')
            dnn.store(path)
            super().__init__(path, devs)
        else:
            raise ValueError('type must be one of: `cpu`, `cuda`.')
        