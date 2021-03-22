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


class MultiheadAttention(Layer):
    """The multihead self-attention layer that transforms a set of matrices
    according to a formula:

    Q = W_Q * Q
    K = W_K * K
    V = W_V * V
    where W_* are trainable matrices of size (Channels_* x GetHiddenSize())
   
    Attention(Q, K, V) = softmax( Q * K_t / sqrt(d_K) ) * V
    where d_k - dimension of k
    
    MultiHeadAttention = dropout_if_needed(concat( head_1, ..., head_N )) * W_O
    where head_i = Attention( W_Q_i * X, W_K_i * X, W_V_i * X ) 
    W_* - trainable parameters and W_O is an additional trainable matrix of size (GetHiddenSize() x GetOutputSize())

    See the papers: https://arxiv.org/pdf/1706.03762.pdf
    https://arxiv.org/pdf/1807.03819.pdf
    
    :param input_layer: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layer: list of object, tuple(object, int)
    :param head_count: The number of heads in attention.
    :type head_count: int, > 0, default=1
    :param hidden_size: The size of trainable matrices W_*. Should be a multiple of head_count.
    :type hidden_size: int, > 0     
    :param output_size: The size of output.
    :type output_size: int, > 0
    :param dropout_rate: Rate of droupout applied to the softmax. 
        A negative value means no dropout.
    :type dropout_rate: float, default=-1
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) matrix Q of shape (1 x **BatchWidth** x **ListSize_Q** x 1 x 1 x 1 x **Channels_Q**)
    (2) matrix K of shape (1 x **BatchWidth** x **ListSize_V** x 1 x 1 x 1 x **Channels_Q**)
    (3) matrix V (1 x **BatchWidth** x **ListSize_V** x 1 x 1 x 1 x **Channels_V**)
    (4) (optional): mask, can be 0.

    .. rubric:: Layer outputs:

    (1) result matrix of shape (1, **BatchWidth**, **ListSize_Q**, 1, 1, 1, output_size)
    """

    def __init__(self, input_layer, head_count, hidden_size, output_size, dropout_rate, name=None):

        if type(input_layer) is PythonWrapper.MultiheadAttention:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, (3, 4))

        if head_count < 1:
            raise ValueError('The `head_count` must be > 0.')

        if hidden_size < 1:
            raise ValueError('The `hidden_size` must be > 0.')

        if output_size < 1:
            raise ValueError('The `output_size` must be > 0.')

        internal = PythonWrapper.MultiheadAttention(str(name), layers, outputs, int(head_count), int(hidden_size), int(output_size), dropout_rate)
        super().__init__(internal)

    @property
    def head_count(self):
        """Gets the number of heads in attention.
        """
        return self._internal.get_head_count()

    @head_count.setter
    def head_count(self, head_count):
        """Sets the number of heads in attention.
        """
        if head_count < 1:
            raise ValueError('The `head_count` must be > 0.')

        self._internal.set_head_count(head_count)

    @property
    def hidden_size(self):
        """Gets the trainable matrices size.
        """
        return self._internal.get_hidden_size()

    @hidden_size.setter
    def hidden_size(self, hidden_size):
        """Sets the trainable matrices size.
        """
        if hidden_size < 1:
            raise ValueError('The `hidden_size` must be > 0.')

        self._internal.set_hidden_size(hidden_size)

    @property
    def output_size(self):
        """Gets the output size.
        """
        return self._internal.get_output_size()

    @output_size.setter
    def output_size(self, output_size):
        """Sets the output size.
        """
        if output_size < 1:
            raise ValueError('The `output_size` must be > 0.')
        self._internal.set_output_size(output_size)

    @property
    def dropout_rate(self):
        """Gets the rate of dropout applied to the softmax.
        """
        return self._internal.get_dropout_rate()

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate):
        """Sets the rate of dropout applied to the softmax.
        """
        self._internal.set_dropout_rate(dropout_rate)

    @property
    def use_mask(self):
        """Checks if the mask will be used.
        """
        return self._internal.get_use_mask()

    @use_mask.setter
    def use_mask(self, use_mask):
        """Specifies if the mask should be used.
        """
        self._internal.set_use_mask(use_mask)
