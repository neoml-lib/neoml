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


class Irnn(Layer):
    """IRNN implementation from this article: https://arxiv.org/pdf/1504.00941.pdf

    It's a simple recurrent unit with the following formula:
    :math:`Y_t = ReLU( FC_input( X_t ) + FC_recur( Y_t-1 ) )`
    Where :math:`FC` are fully-connected layers.

    The crucial point of this layer is weights initialization.
    The weight matrix of :math:`FC_input` is initialized from N(0, input_weight_std),
    where input_weight_std is a layer parameter.
    The weight matrix of :math:`FC_recur` is an identity matrix, 
    multiplied by identity_scale parameter.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param hidden_size: The size of hidden layer. 
        Affects the output size.
    :type hidden_size: int, default=1
    :param identity_scale: The scale of identity matrix, used for the initialization 
        of recurrent weights.
    :type identity_scale: float, default=1.
    :param input_weight_std: The standard deviation for input weights.
    :type input_weight_std: float, default=1e-3
    :param reverse_seq: Indicates if the input sequence should be taken in the reverse order.
    :type reverse_seq: bool, default=False
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the set of vector sequences, of dimensions:

        - **BatchLength** - the length of a sequence
        - **BatchWidth** * **ListSize** - the number of vector sequences in the input set
        - **Height** * **Width** * **Depth** * **Channels** - the size of each vector in the sequence

    .. rubric:: Layer outputs:

    (1) the result of the current step. The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize** are the same as for the input
        - **Height**, **Width**, **Depth** are 1
        - **Channels** is hidden_size
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
        """Gets the hidden layer size.
        """
        return self._internal.get_hidden_size()

    @hidden_size.setter
    def hidden_size(self, hidden_size):
        """Sets the hidden layer size.
        """
        self._internal.set_hidden_size(int(hidden_size))

    @property
    def identity_scale(self):
        """Gets the multiplier for the identity matrix for initialization.
        """
        return self._internal.get_identity_scale()

    @identity_scale.setter
    def identity_scale(self, identity_scale):
        """Sets the multiplier for the identity matrix for initialization.
        """
        self._internal.set_identity_scale(identity_scale)

    @property
    def input_weight_std(self):
        """Gets the standard deviation for input weights.
        """
        return self._internal.get_input_weight_std()

    @input_weight_std.setter
    def input_weight_std(self, input_weight_std):
        """Sets the standard deviation for input weights.
        """
        self._internal.set_input_weight_std(input_weight_std)

    @property
    def reverse_sequence(self):
        """Checks if the input sequence should be taken in reverse order.
        """
        return self._internal.get_reverse_sequence()

    @reverse_sequence.setter
    def reverse_sequence(self, reverse_sequence):
        """Specifies if the input sequence should be taken in reverse order.
        """
        self._internal.set_reverse_sequence(bool(reverse_sequence))

    @property
    def input_weights(self):
        """Gets the FC_input weights. The dimensions:

            - **BatchLength** * **BatchWidth** * **ListSize** is hidden_size
            - **Height** * **Width** * **Depth** * **Channels** is the same as for the input
        """
        return Blob.Blob(self._internal.get_input_weights())

    @input_weights.setter
    def input_weights(self, blob):
        """Sets the FC_input weights. The dimensions:

            - **BatchLength** * **BatchWidth** * **ListSize** is hidden_size
            - **Height** * **Width** * **Depth** * **Channels** is the same as for the input
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_input_weights(blob._internal)

    @property
    def input_free_term(self):
        """Gets the FC_input free term, of hidden_size length.
        """
        return Blob.Blob(self._internal.get_input_free_term())

    @input_free_term.setter
    def input_free_term(self, blob):
        """Sets the FC_input free term, of hidden_size length.
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_input_free_term(blob._internal)

    @property
    def recurrent_weights(self):
        """Gets the FC_recur weights. The dimensions:

            - **BatchLength** * **BatchWidth** * **ListSize** is hidden_size
            - **Height** * **Width** * **Depth** * **Channels** is hidden_size
        """
        return Blob.Blob(self._internal.get_recurrent_weights())

    @recurrent_weights.setter
    def recurrent_weights(self, blob):
        """Sets the FC_recur weights. The dimensions:

            - **BatchLength** * **BatchWidth** * **ListSize** is hidden_size
            - **Height** * **Width** * **Depth** * **Channels** is hidden_size
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_recurrent_weights(blob._internal)

    @property
    def recurrent_free_term(self):
        """Gets the FC_recur free term, of hidden_size length.
        """
        return Blob.Blob(self._internal.get_recurrent_free_term())

    @recurrent_free_term.setter
    def recurrent_free_term(self, blob):
        """Sets the FC_recur free term, of hidden_size length.
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_recurrent_free_term(blob._internal)
