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
from .Utils import check_input_layers


class Gru(Layer):
    """The gated recurrent unit (GRU) layer that works with a set
    of vector sequences.
    
    Layer inputs
    ----------
    #1: the set of vector sequences.
    The dimensions:
    - BatchLength is sequence length
    - BatchWidth * ListSize is the number of sequences
    - Height * Width * Depth * Channels is vector size
    
    #2 (optional): the initial previous step result. If you do not connect
    this input all zeros will be used on the first step.
    The dimensions are the same as for the first input.
    
    Layer outputs
    ----------
    #1: a vector sequence of the same length.
    The dimensions:
    - BatchLength, BatchWidth, ListSize equal to the input's dimensions
    - Height, Width, Depth are 1
    - Channels is equal to hidden layer size
    
    Parameters
    ----------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    hidden_size : int
        The size of the hidden layer.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layer, hidden_size, name=None):

        if type(input_layer) is PythonWrapper.Gru:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, (1, 2))

        if hidden_size < 1:
            raise ValueError('The `hidden_size` must be > 0.')

        internal = PythonWrapper.Gru(str(name), layers, outputs, int(hidden_size))
        super().__init__(internal)

    @property
    def main_weights(self):
        """Gets the output weights as a 2d matrix of the size:
        - BatchLength * BatchWidth * ListSize equal to hidden_size
        - Height * Width * Depth * Channels equal to 
            this dimension of the input plus hidden_size
        """
        return self._internal.get_main_weights()

    @main_weights.setter
    def main_weights(self, main_weights):
        """Sets the output weights as a 2d matrix of the size:
        - BatchLength * BatchWidth * ListSize equal to hidden_size
        - Height * Width * Depth * Channels equal to 
            this dimension of the input plus hidden_size
        """
        self._internal.set_main_weights(main_weights)

    @property
    def main_free_term(self):
        """Gets the output free terms as a blob of total hidden_size size.
        """
        return self._internal.get_main_free_term()

    @main_free_term.setter
    def main_free_term(self, main_free_term):
        """Sets the output free terms as a blob of total hidden_size size.
        """
        self._internal.set_main_free_term(main_free_term)

    @property
    def gate_weights(self):
        """Gets the gate weights as a 2d matrix of the size:
        - BatchLength * BatchWidth * ListSize equal to 2* hidden_size
        - Height * Width * Depth * Channels equal to 
            this dimension of the input plus hidden_size
        """
        return self._internal.get_gate_weights()

    @gate_weights.setter
    def gate_weights(self, gate_weights):
        """Sets the gate weights as a 2d matrix of the size:
        - BatchLength * BatchWidth * ListSize equal to 2* hidden_size
        - Height * Width * Depth * Channels equal to 
            this dimension of the input plus hidden_size
        """
        self._internal.set_gate_weights(gate_weights)

    @property
    def gate_free_term(self):
        """Gets the gate free terms as a blob of total 2 * hidden_size size.
        """
        return self._internal.get_gate_free_term()

    @gate_free_term.setter
    def gate_free_term(self, gate_free_term):
        """Sets the gate free terms as a blob of total 2 * hidden_size size.
        """
        self._internal.set_gate_free_term(gate_free_term)

    @property
    def hidden_size(self):
        """Gets the hidden layer size.
        """
        return self._internal.get_hidden_size()

    @hidden_size.setter
    def hidden_size(self, hidden_size):
        """Sets the hidden layer size.
        """
        self._internal.set_hidden_size(hidden_size)
