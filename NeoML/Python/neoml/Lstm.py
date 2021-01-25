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
import neoml.Dnn as Dnn
import neoml.Utils as Utils


class Lstm(Dnn.Layer):
    """A long short-term memory (LSTM) layer that can be applied to a set of 
    vector sequences. The output is a sequence containing the same number 
    of vectors, each of the layer hidden size length.
    
    Layer inputs
    ------------
    #1: the set of vector sequences, of dimensions:
    - BatchLength - the length of a sequence
    - BatchWidth * ListSize - the number of vector sequences in the input set
    - Height * Width * Depth * Channels - the size of each vector in the sequence
    
    #2 (optional): the initial state of the LSTM before the first step. 
    If this input is not specified, the initial state is all zeros.
    The dimensions:
    - BatchLength should be 1
    - the other dimensions should be the same as for the first input
    
    #3 (optional): the initial value of "previous output" to be used 
    on the first step. If this input is not specified, all zeros will be used.
    The dimensions:
    - BatchLength should be 1
    - the other dimensions should be the same as for the first input

    Layer outputs
    -------------
    #1: the result of the current step
    #2: the layer history
    The dimensions:
    - BatchLength, BatchWidth, ListSize are equal to the first input dimensions
    - Height, Width, Depth are equal to 1
    - Channels is equal to layer hidden size
    
    Parameters
    --------------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    hidden_size : int, default=1
        The size of hidden layer. 
        Affects the output size and the LSTM state vector size.
    dropout_rate : float, default=0.0
        The dropout probability for the input and the recurrent data.
    recurrent_activation : {"linear", "elu", "relu", "leaky_relu", "abs", 
                            "sigmoid", "tanh", "hard_tanh", "hard_sigmoid", 
                            "power", "hswish", "gelu"}, default="sigmoid"
        The activation function that is used in forget, reset, and input gates.
    reverse_seq : bool, default=False
        Indicates if the input sequence should be taken in the reverse order.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layer, hidden_size=1, dropout_rate=0.0, recurrent_activation="sigmoid", reverse_seq=False,
                 name=None):

        if type(input_layer) is PythonWrapper.Lstm:
            super().__init__(input_layer)
            return

        layers, outputs = Utils.check_input_layers(input_layer, 1)

        if hidden_size <= 0:
            raise ValueError('The `hidden_size` must be > 0.')

        recurrent_activation_index = ["linear", "elu", "relu", "leaky_relu", "abs", "sigmoid", "tanh", "hard_tanh",
                                      "hard_sigmoid", "power", "hswish", "gelu"].index(recurrent_activation)

        internal = PythonWrapper.Lstm(str(name), layers[0], int(outputs[0]), int(hidden_size), float(dropout_rate),
                                      recurrent_activation_index, bool(reverse_seq))
        super().__init__(internal)
