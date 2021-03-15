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
from .Utils import check_input_layers
import neoml.Blob as Blob


class Irnn(Layer):
    """IRNN implementation from this article: https://arxiv.org/pdf/1504.00941.pdf

    It's a simple recurrent unit with the following formula:
        Y_t = ReLU( FC_input( X_t ) + FC_recur( Y_t-1 ) )
    Where FC_* are fully-connected layers

    The crucial point of this layer is weights initialization
    The weight matrix of FC_input is initialized from N(0, input_weight_std) where input_weight_std is a layer param
    The weight matrix of FC_recur is an identity matrix multiplied by identity_scale param
    
    Layer inputs
    ------------
    #1: the set of vector sequences, of dimensions:
    - BatchLength - the length of a sequence
    - BatchWidth * ListSize - the number of vector sequences in the input set
    - Height * Width * Depth * Channels - the size of each vector in the sequence

    Layer outputs
    -------------
    #1: the result of the current step
    
    Parameters
    --------------
    input_layer : (object, int)
        The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    hidden_size : int, default=1
        The size of hidden layer. 
        Affects the output size.
    identity_scale : float, default=1.
        The scale of identity matrix, used for the initialization of recurrent weights
    input_weight_std : float, default=1e-3
        The standard deviation for input weights
    reverse_seq : bool, default=False
        Indicates if the input sequence should be taken in the reverse order.
    name : str, default=None
        The layer name.
    """

    def __init__(self, input_layer, hidden_size=1, identity_scale=1., input_weight_std=1e-3, reverse_seq=False,
                 name=None):

        if type(input_layer) is PythonWrapper.Irnn:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if hidden_size <= 0:
            raise ValueError('The `hidden_size` must be > 0.')

        if identity_scale < 0:
            raise ValueError('The `identity_scale` must be > 0.')

        if input_weight_std < 0:
            raise ValueError('The `input_weight_std` must be > 0.')

        internal = PythonWrapper.Irnn(str(name), layers, outputs, int(hidden_size), float(identity_scale),
                                      float(input_weight_std), bool(reverse_seq))
        super().__init__(internal)

    @property
    def hidden_size(self):
        """
        """
        return self._internal.get_hidden_size()

    @hidden_size.setter
    def hidden_size(self, hidden_size):
        """
        """
        self._internal.set_hidden_size(int(hidden_size))

    @property
    def identity_scale(self):
        """
        """
        return self._internal.get_identity_scale()

    @identity_scale.setter
    def identity_scale(self, identity_scale):
        """
        """
        self._internal.set_identity_scale(identity_scale)

    @property
    def input_weight_std(self):
        """
        """
        return self._internal.get_input_weight_std()

    @input_weight_std.setter
    def input_weight_std(self, input_weight_std):
        """
        """
        self._internal.set_input_weight_std(input_weight_std)

    @property
    def reverse_sequence(self):
        """
        """
        return self._internal.get_reverse_sequence()

    @reverse_sequence.setter
    def reverse_sequence(self, reverse_sequence):
        """
        """
        self._internal.set_reverse_sequence(bool(reverse_sequence))

    @property
    def input_weights(self):
        """
        """
        return Blob.Blob(self._internal.get_input_weights())

    @input_weights.setter
    def input_weights(self, blob):
        """
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_input_weights(blob._internal)

    @property
    def input_free_term(self):
        """
        """
        return Blob.Blob(self._internal.get_input_free_term())

    @input_free_term.setter
    def input_free_term(self, blob):
        """
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_input_free_term(blob._internal)

    @property
    def recurrent_weights(self):
        """
        """
        return Blob.Blob(self._internal.get_recurrent_weights())

    @recurrent_weights.setter
    def recurrent_weights(self, blob):
        """
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_recurrent_weights(blob._internal)

    @property
    def recurrent_free_term(self):
        """
        """
        return Blob.Blob(self._internal.get_recurrent_free_term())

    @recurrent_free_term.setter
    def recurrent_free_term(self, blob):
        """
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_recurrent_free_term(blob._internal)
