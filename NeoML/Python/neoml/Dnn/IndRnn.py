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

class IndRnn(Layer):
    """Independently Recurrent Neural Network (IndRNN): https://arxiv.org/pdf/1803.04831.pdf

    It's a simple recurrent unit with the following formula:
    :math:`Y_t = sigmoid(W * X_t + B + U * dropout(Y_{t-1}))`
    where :math:`W` and :math:`B` are weights and free terms of the fully-connected layer
    (:math:`W * X_t` is a matrix multiplication) and :math:`U` is a vector
    (:math:`U * Y_{t-1}` is an eltwise multiplication of 2 vectors of the same length)

    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param hidden_size: The size of hidden layer. 
        Affects the output size.
    :type hidden_size: int, default=1
    :param dropout_rate: The rate of the dropout, applied to both input and recurrent data
    :type dropout_rate: float, default=0.
    :param reverse_sequence: Indicates if the input sequence should be taken in the reverse order.
    :type reverse_sequence: bool, default=False

    .. rubric:: Layer inputs:

    (1) the set of vector sequences, of dimensions:

        - **BatchLngth** - the length of sequence
        - **BatchWidth** * **ListSize** - the number of vector sequences in the input set
        - **Height** * **Width** * **Depth** * **Channels** - the size of each vector in the sequence
    
    .. rubric:: Layer outputs:

    (1) the result of the layer. The dimensions:

        - **BatchLength**, **BatchWidth**, **ListSize** are the same as for the input
        - **Height**, **Width**, **Depth** are 1
        - **Channels** is hidden_size
    """

    def __init__(self, input_layer, hidden_size=1, dropout_rate=0., reverse_sequence=False, name=None):

        if type(input_layer) is PythonWrapper.IndRnn:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if hidden_size <= 0:
            raise ValueError('The `hidden_size` must be > 0.')
        if dropout_rate >= 1.:
            raise ValueError('The `dropout_rate` must be < 1.')

        internal = PythonWrapper.IndRnn(str(name), layers, outputs, int(hidden_size), float(dropout_rate),
                                        bool(reverse_sequence))
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
    def dropout_rate(self):
        """Gets the dropout rate.
        """
        return self._internal.get_dropout_rate()

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate):
        """Sets the dropout rate.
        """
        self._internal.set_dropout_rate(float(dropout_rate))

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
        """Gets the input weights matrix. The dimensions:

            - **BatchLength** * **BatchWidth** * **ListSize** is hidden_size
            - **Height** * **Width** * **Depth** * **Channels** is the same as for the input
        """
        return Blob.Blob(self._internal.get_input_weights())

    @input_weights.setter
    def input_weights(self, blob):
        """Sets the input weights matrix. The dimensions:

            - **BatchLength** * **BatchWidth** * **ListSize** is hidden_size
            - **Height** * **Width** * **Depth** * **Channels** is the same as for the input
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_input_weights(blob._internal)

    @property
    def recurrent_weights(self):
        """Gets the recurrent weights vector, of hidden_size length.
        """
        return Blob.Blob(self._internal.get_recurrent_weights())

    @recurrent_weights.setter
    def recurrent_weights(self, blob):
        """Sets the recurrent weights vector, of hidden_size length.
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_recurrent_weights(blob._internal)

    @property
    def bias(self):
        """Gets the bias vector, of hidden_size length.
        """
        return Blob.Blob(self._internal.get_recurrent_weights())

    @bias.setter
    def bias(self, blob):
        """Sets the bias vector, of hidden_size length.
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_bias(blob._internal)
