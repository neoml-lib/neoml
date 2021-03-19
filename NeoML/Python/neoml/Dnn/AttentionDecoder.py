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
from neoml.Utils import check_input_layers


class AttentionDecoder(Layer):
    """The layer that converts the input sequence 
    into the output sequence, not necessarily of the same length

    Layer inputs
    ------------
    #1: a data blob of any size with the input sequence
    
    #2: the special character used to initialize the output sequence
    All dimensions are equal to 1
    
    Layer outputs
    -------------
    #1: the output sequence
    The dimensions:
    - BatchLength equal to output_seq_len
    - Channels equal to output_object_size
    - all other dimensions are equal to 1

       
    Parameters
    ---------------
    input_layers : array of (object, int) tuples and objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    score : {"additive", "dot_product"}
        The type of estimate function used to check alignment 
        of input and output sequences.
        ``additive`` is tanh(x*Wx + y*Wy)*v
        ``dot_product`` is x*W*y
    hidden_size : int
        The size of the hidden layer.
    output_object_size : int
        The number of channels in the output object.
    output_seq_len : int
        The length of the output sequence.
    name : str, default=None
        The layer name.
    """
    scores = ["additive", "dot_product"]

    def __init__(self, input_layers, score, hidden_size, output_object_size, output_seq_len, name=None):

        if type(input_layers) is PythonWrapper.AttentionDecoder:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 2)

        score_index = self.scores.index(score)

        if output_object_size <= 0:
            raise ValueError('`output_object_size` must be > 0.')

        if output_seq_len <= 0:
            raise ValueError('`output_seq_len` must be > 0.')

        if hidden_size <= 0:
            raise ValueError('`hidden_size` must be > 0.')

        internal = PythonWrapper.AttentionDecoder(str(name), layers[0], int(outputs[0]), layers[1], int(outputs[1]),
                                                  score_index, int(output_object_size), int(output_seq_len),
                                                  int(hidden_size))
        super().__init__(internal)

    @property
    def score(self):
        """Gets the estimate function.
        """
        return self.scores[self._internal.get_score()]

    @score.setter
    def score(self, new_score):
        """Sets the estimate function.
        """
        score_index = self.scores.index(new_score)
        self._internal.set_score(score_index)

    @property
    def output_seq_len(self):
        """Gets the length of the output sequence.
        """
        return self._internal.get_output_seq_len()

    @output_seq_len.setter
    def output_seq_len(self, output_seq_len):
        """Sets the length of the output sequence.
        """
        self._internal.set_output_seq_len(int(output_seq_len))
        
    @property
    def output_object_size(self):
        """Gets the number of channels in the output.
        """
        return self._internal.get_output_object_size()

    @output_object_size.setter
    def output_object_size(self, output_object_size):
        """Sets the number of channels in the output.
        """
        self._internal.set_output_object_size(int(output_object_size))

    @property
    def hidden_layer_size(self):
        """Gets the size of the hidden layer.
        """
        return self._internal.get_hidden_layer_size()

    @hidden_layer_size.setter
    def hidden_layer_size(self, hidden_layer_size):
        """Sets the size of the hidden layer.
        """
        self._internal.set_hidden_layer_size(int(hidden_layer_size))
