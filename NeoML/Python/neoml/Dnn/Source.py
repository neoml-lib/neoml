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
from .Dnn import Layer
import neoml.Blob as Blob


class Source(Layer):
    """The source layer that serves to pass a data blob into the network.
    Layer inputs
    ---------------
    The layer has no inputs.
    
    Layer outputs
    ---------------
    The layer has one output that contains the data blob 
    passed into the last call of set_blob.
      
    Parameters
    -------------
    dnn : object
        The neural network.
    name : str, default=None
        The layer name.
    """
    def __init__(self, dnn, name=None):
        if type(dnn) is PythonWrapper.Source:
            super().__init__(dnn)
            return

        internal = PythonWrapper.Source(dnn, str(name))
        super().__init__(internal)

    def set_blob(self, blob):
        """Sets the blob with source data.
        """
        self._internal.set_blob(blob._internal)
            
    def get_blob(self):
        """Gets the blob with source data.
        """
        return Blob(self._internal.get_blob())
