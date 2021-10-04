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
import numpy

class Transformer(Layer):
    """The transformer layer.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected. 
    :type input_layer: object, tuple(object, int)
    :param head_count: The number of heads in self-attention layer
    :type head_count: int, default=1
    :param hidden_size: The hidden size of self-attention layer, must be a multiple of head_count
    :type hidden_size: int, default=1
    :param output_size: The output size of self-attention and second fully-connected layer
    :type output_size: int, default=1
    :param attention_dropout: The rate of the dropout inside of self-attention
    :type attention_dropout: float, default=0.
    :param feed_forward_size: The size of the first fully-connected layer in feed-forward
    :type feed_forward_size: int, default=1
    :param activation: activation used between fully-connected layers in feed-forward
    :type activation: str, {"linear", "elu", "relu", "leaky_relu", "abs", "sigmoid", "tanh",
        "hard_tanh", "hard_sigmoid", "power", "hswish", "gelu"}, default="relu"
    :param feed_forward_dropout: The of the dropout after each of the fully-connected layers in feed-forward
    :type feed_forward_dropout: float, default=0.
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a data blob of ... size.
    
    .. rubric:: Layer outputs:

    (1) a blob of ... size.
    """
    activations = ["linear", "elu", "relu", "leaky_relu", "abs", "sigmoid", "tanh", "hard_tanh", "hard_sigmoid", "power", "hswish", "gelu"]

    def __init__(self, input_layer, head_count=1, hidden_size=1, output_size=1, attention_dropout=0.,
        feed_forward_size=1, activation='relu', feed_forward_dropout=0., name=None):

        if type(input_layer) is PythonWrapper.Transformer:
            super().__init__(input_layer)
            return

        if head_count < 1:
            raise ValueError('The `head_count` must be > 0.')

        if hidden_size < 1:
            raise ValueError('The `hidden_size` must be > 0.')
        
        if hidden_size % head_count != 0:
            raise ValueError('The `hidden_size` must be a multiple of `head_count`')

        if output_size < 1:
            raise ValueError('The `output_size` must be > 0.')

        if attention_dropout >= 1.:
            raise ValueError('The `attention_dropout` must be < 1.')

        if feed_forward_size < 1:
            raise ValueError('The `feed_forward_size` must be > 0.')

        if activation not in actvations:
            raise ValueError('The `activation` has invalid value'

        if feed_forward_dropout >= 1.:
            raise ValueError('The `feed_forward_dropout` must be < 1.')

        activation = acitvations.index(activation)
 
        layers, outputs = check_input_layers(input_layer, 1)

        internal = PythonWrapper.Transformer(str(name), layers[0], int(outputs[0]), int(head_count), int(hidden_size),
            int(output_size), float(attention_dropout), int(feed_forward_size), float(feed_forward_dropout), int(activation))
        super().__init__(internal)
