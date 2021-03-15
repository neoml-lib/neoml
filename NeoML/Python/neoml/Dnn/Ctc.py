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


class CtcLoss(Layer):
    """
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    loss_weight : float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, blank, skip, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.CtcLoss:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, (2, 5))

        internal = PythonWrapper.CtcLoss(str(name), layers, outputs, int(blank), bool(skip), float(loss_weight))
        super().__init__(internal)

    @property
    def blank(self):
        """
        """
        return self._internal.get_blank_label()

    @blank.setter
    def blank(self, value):
        """
        """
        self._internal.set_blank_label(int(value))

    @property
    def last_loss(self):
        """Gets the value of the loss function on the last step.
        """
        return self._internal.get_last_loss()
        
    @property
    def loss_weight(self):
        """Gets the multiplier for the loss function during training.
        """
        return self._internal.get_loss_weight()

    @loss_weight.setter
    def loss_weight(self, weight):
        """Sets the multiplier for the loss function during training.
        """
        self._internal.set_loss_weight(weight)

    @property
    def max_gradient(self):
        """Gets the upper limit for the absolute value of the function gradient.
        """
        return self._internal.get_max_gradient()

    @max_gradient.setter
    def max_gradient(self, value):
        """Sets the upper limit for the absolute value of the function gradient.
        """
        if value <= 0 :
            raise ValueError('The `max_gradient` must be > 0.')

        self._internal.set_max_gradient(value)

    @property
    def skip(self):
        """
        """
        return self._internal.get_skip()

    @skip.setter
    def skip(self, value):
        """
        """
        self._internal.set_skip(bool(value))

# ----------------------------------------------------------------------------------------------------------------------


class CtcDecoding(Layer):
    """
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, blank, blank_threshold, arc_threshold, name=None):

        if type(input_layers) is PythonWrapper.CtcDecoding:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, (1, 2))

        internal = PythonWrapper.CtcDecoding(str(name), layers, outputs, int(blank), float(blank_threshold), float(arc_threshold))
        super().__init__(internal)

    @property
    def blank(self):
        """
        """
        return self._internal.get_blank_label()

    @blank.setter
    def blank(self, value):
        """
        """
        self._internal.set_blank_label(int(value))

    @property
    def blank_threshold(self):
        """
        """
        return self._internal.get_blank_threshold()

    @blank_threshold.setter
    def blank_threshold(self, value):
        """
        """
        self._internal.set_blank_threshold(float(value))

    @property
    def arc_threshold(self):
        """
        """
        return self._internal.get_arc_threshold()

    @arc_threshold.setter
    def arc_threshold(self, value):
        """
        """
        self._internal.set_arc_threshold(float(value))

    @property
    def sequence_length(self):
        """
        """
        return self._internal.get_sequence_length()

    @property
    def batch_width(self):
        """
        """
        return self._internal.get_batch_width()

    @property
    def label_count(self):
        """
        """
        return self._internal.get_label_count()

    def get_best_sequence(self, sequence_number):
        """
        """
        return self._internal.get_best_sequence(sequence_number)
