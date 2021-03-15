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
from neoml.Utils import check_input_layers
import neoml.Blob as Blob


class Lstm(Layer):
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
    activations = ["linear", "elu", "relu", "leaky_relu", "abs", "sigmoid", "tanh", "hard_tanh", "hard_sigmoid", "power", "hswish", "gelu"]

    def __init__(self, input_layer, hidden_size=1, dropout_rate=0.0, activation="sigmoid", reverse_seq=False,
                 name=None):

        if type(input_layer) is PythonWrapper.Lstm:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, (1, 3))

        if hidden_size <= 0:
            raise ValueError('The `hidden_size` must be > 0.')

        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError('The `dropout_rate` must be in [0, 1).')

        recurrent_activation_index = self.activations.index(activation)

        internal = PythonWrapper.Lstm(str(name), layers, outputs, int(hidden_size), float(dropout_rate),
                                      recurrent_activation_index, bool(reverse_seq))
        super().__init__(internal)

    @property
    def hidden_size(self):
        """Gets the hidden layer size.
        """
        return self._internal.get_hidden_size()

    @hidden_size.setter
    def hidden_size(self, hidden_size):
        """Sets the hidden layer size.
        """
        self._internal.set_hidden_size(int(hidden_size))

    @property
    def activation(self):
        """Gets the activation function.
        """
        return self.activations[self._internal.get_activation()]

    @activation.setter
    def activation(self, activation):
        """Sets the activation function.
        """
        activation_index = self.activations.index(activation)
        self._internal.set_activation(activation_index)

    @property
    def dropout(self):
        """Gets the dropout rate.
        """
        return self._internal.get_dropout()

    @dropout.setter
    def dropout(self, rate):
        """Sets the dropout rate.
        """
        if rate < 0 or rate >= 1:
            raise ValueError('The `rate` must be in [0, 1).')

        self._internal.set_dropout(float(rate))

    @property
    def reverse_sequence(self):
        """Checks if the input sequence will be reverted.
        """
        return self._internal.get_reverse_sequence()

    @reverse_sequence.setter
    def reverse_sequence(self, reverse_sequence):
        """Specifies if the input sequence should be reverted.
        """
        self._internal.set_reverse_sequence(bool(reverse_sequence))

    @property
    def input_weights(self):
        """Gets the input hidden layer weights.
        The blob size is (4*HiddenSize)x1x1xInputSize.
        """
        return Blob.Blob(self._internal.get_input_weights())

    @input_weights.setter
    def input_weights(self, blob):
        """Sets the input hidden layer weights.
        The blob size is (4*HiddenSize)x1x1xInputSize.
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_input_weights(blob._internal)

    @property
    def input_free_term(self):
        """Gets the input free term.
        """
        return Blob.Blob(self._internal.get_input_free_term())

    @input_free_term.setter
    def input_free_term(self, blob):
        """Sets the input free term.
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_input_free_term(blob._internal)

    @property
    def recurrent_weights(self):
        """Gets the recurrent hidden layer weights.
        The blob size is (4*HiddenSize)x1x1xHiddenSize.
        """
        return Blob.Blob(self._internal.get_recurrent_weights())

    @recurrent_weights.setter
    def recurrent_weights(self, blob):
        """Sets the recurrent hidden layer weights.
        The blob size is (4*HiddenSize)x1x1xHiddenSize.
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_recurrent_weights(blob._internal)

    @property
    def recurrent_free_term(self):
        """Gets the recurrent free term.
        """
        return Blob.Blob(self._internal.get_recurrent_free_term())

    @recurrent_free_term.setter
    def recurrent_free_term(self, blob):
        """Sets the recurrent free term.
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_recurrent_free_term(blob._internal)
