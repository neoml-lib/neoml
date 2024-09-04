""" Copyright (c) 2017-2024 ABBYY

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


class TransformerEncoder(Layer):
    """The transformer encoder layer based on the "Attention is all you need" article.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected. 
    :type input_layer: object, tuple(object, int)
    :param head_count: The number of heads in self-attention sub-layer
    :type head_count: int, default=1
    :param hidden_size: The hidden size of self-attention layer, must be a multiple of head_count
    :type hidden_size: int, default=1
    :param dropout: The rate of dropouts of transformer
    :type dropout: float, default=0.
    :param sa_dropout: The rate of dropout of self-attention sub-layer
    :type sa_dropout: float, default=0.
    :param feed_forward_size: The size of the first fully-connected layer in feed-forward
    :type feed_forward_size: int, default=1
    :param pre_norm: The place of the normalization layer: right after input or before feedForward as usual
    :type pre_norm: bool, default=False
    :param activation: activation used between fully-connected layers in feed-forward
    :type activation: str, {"linear", "elu", "relu", "leaky_relu", "abs", "sigmoid", "tanh",
        "hard_tanh", "hard_sigmoid", "power", "hswish", "gelu"}, default="relu"
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) input data: the set of vector sequences, of dimensions:

        - **BatchLength**, **Height**, **Width** and **Depth** must be equal to 1
        - **BatchWidth** - number of sequences in batch
        - **ListSize** - length of sequences
        - **Channels** - size of the elements in sequences

    (2) (optional) input mask: blob containing 1.0 and 0.0 where 1.0 means IGNORED object

        - **Width** (seq_Q) and **Channels** (seq_V) must be equal to the ListSize of the first input
        - Other dimensions must be equal to 1
    
    .. rubric:: Layer outputs:

    (1) output data: float blob, of dimensions:

        - **BatchWidth** and **ListSize** are equal to the corresponding dims of the first input
        - **BatchLength**, **Height**, **Width** and **Depth** are equal to 1
        - **Channels** is equal to the Channels of the first input

    """
    activations = ["linear", "elu", "relu", "leaky_relu", "abs", "sigmoid", "tanh", "hard_tanh", "hard_sigmoid", "power", "hswish", "gelu"]

    def __init__(self, input_layers, head_count=1, hidden_size=1, dropout=0., sa_dropout=0., feed_forward_size=1, activation='relu', pre_norm=False, name=None):

        if type(input_layers) is PythonWrapper.TransformerEncoder:
            super().__init__(input_layers)
            return

        if head_count < 1:
            raise ValueError('The `head_count` must be > 0.')

        if hidden_size < 1:
            raise ValueError('The `hidden_size` must be > 0.')
        
        if hidden_size % head_count != 0:
            raise ValueError('The `hidden_size` must be a multiple of `head_count`')

        if dropout >= 1.:
            raise ValueError('The `dropout` for transformer must be < 1.')

        if sa_dropout >= 1.:
            raise ValueError('The `dropout` for self-attention must be < 1.')

        if feed_forward_size < 1:
            raise ValueError('The `feed_forward_size` must be > 0.')

        if activation not in self.activations:
            raise ValueError('The `activation` has invalid value')

        activation = self.activations.index(activation)
 
        layers, outputs = check_input_layers(input_layers, (1, 2))

        internal = PythonWrapper.TransformerEncoder(str(name), layers, outputs,
            int(head_count), int(hidden_size), float(dropout), float(sa_dropout), int(feed_forward_size), int(activation), bool(pre_norm))
        super().__init__(internal)

    @property
    def head_count(self):
        """Gets the head count of the attention.
        """
        return self._internal.get_head_count()

    @head_count.setter
    def head_count(self, head_count):
        """Sets the head count of the attention.
        """
        if head_count < 1:
            raise ValueError('The `head_count` must be > 0.')
        self._internal.set_head_count(int(head_count))

    @property
    def hidden_size(self):
        """Gets the hidden size of the attention.
        """
        return self._internal.get_hidden_size()

    @hidden_size.setter
    def hidden_size(self, hidden_size):
        """Sets the hidden size of the attention.
        Must be a multiple of the head_count.
        """
        if hidden_size < 1:
            raise ValueError('The `hidden_size` must be > 0.')
        self._internal.set_hidden_size(int(hidden_size))

    @property
    def dropout(self):
        """Gets the dropout rate for transformer.
        """
        return self._internal.get_dropout()

    @dropout.setter
    def dropout(self, dropout):
        """Sets the dropout rate for transformer.
        """
        if dropout >= 1.0:
            raise ValueError('The `dropout` for transformer must be < 1.')
        self._internal.set_dropout(float(dropout))

    @property
    def sa_dropout(self):
        """Gets the dropout rate of self-attention sub-layer.
        """
        return self._internal.get_sa_dropout()

    @sa_dropout.setter
    def sa_dropout(self, dropout):
        """Sets the dropout rate of self-attention sub-layer.
        """
        if dropout >= 1.0:
            raise ValueError('The `dropout` for self-attention must be < 1.')
        self._internal.set_sa_dropout(float(dropout))

    @property
    def feed_forward_size(self):
        """Gets the feed forward size.
        """
        return self._internal.get_feed_forward_size()

    @feed_forward_size.setter
    def feed_forward_size(self, feed_forward_size):
        """Sets the feed forward size.
        """
        if feed_forward_size < 1:
            raise ValueError('The `feed_forward_size` must be > 0.')
        self._internal.set_feed_forward_size(int(feed_forward_size))

    @property
    def pre_norm(self):
        """Gets the place of the normalization layer.
        """
        return self._internal.get_pre_norm()

    @pre_norm.setter
    def pre_norm(self, pre_norm):
        """Sets the place of the normalization layer.
        """
        self._internal.set_pre_norm(bool(pre_norm))

