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
import neoml.Blob as Blob


class Qrnn(Layer):
    """The quasi-recurrent layer that can be applied to a set of vector 
    sequences.
    Unlike LSTM or GRU, the layer performs most of the calculations 
    before the recurrent part, which helps improve performance on GPU.
    We use time convolution outside of the recurrent part instead of
    fully-connected layers in the recurrent part.
    See https://arxiv.org/abs/1611.01576

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param hidden_size: The hidden layer size.    
    :type hidden_size: int, > 0
    :param window_size: The size of the window used in time convolution.    
    :type window_size: int, > 0
    :param stride: The window stride for time convolution.
    :type stride: int, > 0, default=1
    :param paddings: The additional zeros tacked to the start and end of sequence 
        before convolution.    
    :type paddings: tuple(int, int), >= 0, default=(0, 0)
    :param activation: The activation function used in the update gate.    
    :type activation: str, {"linear", "elu", "relu", "leaky_relu", "abs", "sigmoid",
                    "tanh", "hard_tanh", "hard_sigmoid", "power", "hswish", 
                    "gelu"}, default="tanh"
    :param dropout: The dropout probability in the forget gate.
    :type dropout: float, [0..1], default=0.0
    :param mode: The way of processing the input sequences.
        - bidirectional_concat means the direct and the reverse 
        sequence are concatenated and then processed as one;
        - bidirectional_sum means the direct and the reverse
        sequence are added up and then processed as one.  
    :type mode: str, {"direct", "reverse", "bidirectional_concat", 
        "bidirectional_sum"}, default="direct"
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the set of vector sequences, of dimensions:

        - **BatchLength** is the length of one sequence
        - **BatchWidth** is the number of vector sequences in the set
        - **ListSize** is 1
        - **Height** * **Width** * **Depth** * **Channels** is the vector length
    
    (2) (optional): the initial state of the recurrent part.
        If not set, the recurrent part is all zeros before the first step.
        The dimensions:

        - **BatchLength**, **ListSize**, **Height**, **Width**, **Depth** are 1
        - **BatchWidth** is the same as for the first input
        - **Channels** is hidden_size
    
    .. rubric:: Layer outputs:

    (1) the result sequence. The dimensions:

        - **BatchLength** can be calculated from the input as
          (BatchLength + paddings[0] + paddings[1] - (window_size - 1))/(stride + 1)
        - **BatchWidth** is the same as for the inputs
        - **ListSize**, **Height**, **Width**, **Depth** are 1
        - **Channels** is hidden_size for all recurrent modes 
          except bidirectional_concat, when it is 2 * hidden_size
    """

    activations = ["linear", "elu", "relu", "leaky_relu", "abs", "sigmoid", "tanh", "hard_tanh", "hard_sigmoid", "power", "hswish", "gelu"]
    recurrent_modes = ["direct", "reverse", "bidirectional_concat", "bidirectional_sum"]

    def __init__(self, input_layers, hidden_size, window_size, stride=1, paddings=(0, 0), activation="tanh", dropout=0.0, mode="direct", name=None):

        if type(input_layers) is PythonWrapper.Qrnn:
            super().__init__(input_layers)
            return

        if hidden_size < 1:
            raise ValueError('The `hidden_size` must be > 0.')

        if window_size < 1:
            raise ValueError('The `window_size` must be > 0.')

        if stride < 1:
            raise ValueError('The `stride` must be > 0.')

        if len(paddings) != 2:
            raise ValueError('The `paddings` must have two values (padding_front, padding_back).')

        padding_front = paddings[0]
        if padding_front < 0:
            raise ValueError('The `padding_front` must be >= 0.')

        padding_back = paddings[1]
        if padding_back < 0:
            raise ValueError('The `padding_back` must be >= 0.')

        activation_index = self.activations.index(activation)

        if dropout < 0 or dropout >= 1:
            raise ValueError('The `dropout` must be in [0, 1).')

        mode_index = self.recurrent_modes.index(mode)

        layers, outputs = check_input_layers(input_layers, (1, 2))

        internal = PythonWrapper.Qrnn(str(name), layers, int(hidden_size), int(window_size), int(stride), int(padding_front), int(padding_back), activation_index, float(dropout), mode_index, outputs)
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
        if hidden_size < 1:
            raise ValueError('The `hidden_size` must be > 0.')

        self._internal.set_hidden_size(int(hidden_size))

    @property
    def window_size(self):
        """Gets the window size for time convolution.
        """
        return self._internal.get_window_size()

    @window_size.setter
    def window_size(self, window_size):
        """Sets the window size for time convolution.
        """
        if window_size < 1:
            raise ValueError('The `window_size` must be > 0.')

        self._internal.set_window_size(int(window_size))

    @property
    def stride(self):
        """Gets the stride for time convolution.
        """
        return self._internal.get_stride()

    @stride.setter
    def stride(self, stride):
        """Sets the stride for time convolution.
        """
        if stride < 1:
            raise ValueError('The `stride` must be > 0.')

        self._internal.set_stride(int(stride))

    @property
    def padding_front(self):
        """Gets the size of zero padding at the sequence start.
        """
        return self._internal.get_padding_front()

    @padding_front.setter
    def padding_front(self, padding):
        """Sets the size of zero padding at the sequence start.
        """
        if padding < 0:
            raise ValueError('The `padding_front` must be >= 0.')

        self._internal.set_padding_front(int(padding))

    @property
    def padding_back(self):
        """Gets the size of zero padding at the sequence end.
        """
        return self._internal.get_padding_back()

    @padding_back.setter
    def padding_back(self, padding):
        """Sets the size of zero padding at the sequence end.
        """
        if padding < 0:
            raise ValueError('The `padding_back` must be >= 0.')

        self._internal.set_padding_back(int(padding))

    @property
    def activation(self):
        """Gets the activation function used in the update gate.
        """
        return self.activations[self._internal.get_activation()]

    @activation.setter
    def activation(self, activation):
        """Sets the activation function used in the update gate.
        """
        activation_index = self.activations.index(activation)
        self._internal.set_activation(activation_index)

    @property
    def dropout(self):
        """Gets the dropout probability for the forget gate.
        """
        return self._internal.get_dropout()

    @dropout.setter
    def dropout(self, rate):
        """Sets the dropout probability for the forget gate.
        """
        if rate < 0 or rate >= 1:
            raise ValueError('The `rate` must be in [0, 1).')

        self._internal.set_dropout(float(rate))

    @property
    def filter(self):
        """Gets the trained weights for each gate. The blob dimensions:
        - **BatchLength** is 1
        - **BatchWidth** is 3 * hidden_size
        (contains the weights for each of the three gates 
        in the order: update, forget, output)
        - **Height** is window_size
        - **Width**, **Depth** are 1
        - **Channels** is equal to the input's **Height** * **Width** * **Depth** * **Channels**
        """
        return Blob.Blob(self._internal.get_filter())

    @filter.setter
    def filter(self, blob):
        """Sets the trained weights for each gate. The blob dimensions:
        - **BatchLength** is 1
        - **BatchWidt**h is 3 * hidden_size
        (contains the weights for each of the three gates 
        in the order: update, forget, output)
        - **Height** is window_size
        - **Width**, **Depth** are 1
        - **Channels** is equal to the input's **Height** * **Width** * **Depth** * **Channels**
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_filter(blob._internal)

    @property
    def free_term(self):
        """Gets the free term for all three gates in the same order.
        The blob size is 3 * hidden_size.
        """
        return Blob.Blob(self._internal.get_free_term())

    @free_term.setter
    def free_term(self, blob):
        """Sets the free term for all three gates in the same order.
        The blob size is 3 * hidden_size.
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_free_term(blob._internal)

    @property
    def recurrent_mode(self):
        """Gets the sequence processing mode.
        """
        return self.recurrent_modes[self._internal.get_recurrent_mode()]

    @recurrent_mode.setter
    def recurrent_mode(self, mode):
        """Sets the sequence processing mode.
        """
        mode_index = self.recurrent_modes.index(mode)

        self._internal.set_recurrent_mode(mode_index)
