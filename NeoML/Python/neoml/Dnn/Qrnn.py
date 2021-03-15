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
from .Utils import check_input_layers
import neoml.Blob as Blob


class Qrnn(Layer):
    """
    
    Parameters
    ----------
    input_layers :
    count :
    size :
    name : str, default=None
        The layer name.
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
            raise ValueError('The `paddings` must be (padding_front, padding_back).')

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
        """
        """
        return self._internal.get_hidden_size()

    @hidden_size.setter
    def hidden_size(self, hidden_size):
        """
        """
        if hidden_size < 1:
            raise ValueError('The `hidden_size` must be > 0.')

        self._internal.set_hidden_size(int(hidden_size))

    @property
    def window_size(self):
        """
        """
        return self._internal.get_window_size()

    @window_size.setter
    def window_size(self, window_size):
        """
        """
        if window_size < 1:
            raise ValueError('The `window_size` must be > 0.')

        self._internal.set_window_size(int(window_size))

    @property
    def stride(self):
        """
        """
        return self._internal.get_stride()

    @stride.setter
    def stride(self, stride):
        """
        """
        if stride < 1:
            raise ValueError('The `stride` must be > 0.')

        self._internal.set_stride(int(stride))

    @property
    def padding_front(self):
        """
        """
        return self._internal.get_padding_front()

    @padding_front.setter
    def padding_front(self, padding):
        """
        """
        if padding < 0:
            raise ValueError('The `padding_front` must be >= 0.')

        self._internal.set_padding_front(int(padding))

    @property
    def padding_back(self):
        """
        """
        return self._internal.get_padding_back()

    @padding_back.setter
    def padding_back(self, padding):
        """
        """
        if padding < 0:
            raise ValueError('The `padding_back` must be >= 0.')

        self._internal.set_padding_back(int(padding))

    @property
    def activation(self):
        """
        """
        return self.activations[self._internal.get_activation()]

    @activation.setter
    def activation(self, activation):
        """
        """
        activation_index = self.activations.index(activation)
        self._internal.set_activation(activation_index)

    @property
    def dropout(self):
        """
        """
        return self._internal.get_dropout()

    @dropout.setter
    def dropout(self, rate):
        """
        """
        if rate < 0 or rate >= 1:
            raise ValueError('The `rate` must be in [0, 1).')

        self._internal.set_dropout(float(rate))

    @property
    def filter(self):
        """
        """
        return Blob.Blob(self._internal.get_filter())

    @filter.setter
    def filter(self, blob):
        """
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')

        self._internal.set_filter(blob._internal)

    @property
    def free_term(self):
        """
        """
        return Blob.Blob(self._internal.get_free_term())

    @free_term.setter
    def free_term(self, blob):
        """
        """
        if not type(blob) is Blob.Blob:
            raise ValueError('The `blob` must be neoml.Blob.')
 
        self._internal.set_free_term(blob._internal)

    @property
    def recurrent_mode(self):
        """
        """
        return self.recurrent_modes[self._internal.get_recurrent_mode()]

    @recurrent_mode.setter
    def recurrent_mode(self, mode):
        """
        """
        mode_index = self.recurrent_modes.index(mode)

        self._internal.set_recurrent_mode(mode_index)
