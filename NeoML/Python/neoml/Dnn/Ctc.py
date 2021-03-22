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


class CtcLoss(Layer):
    """The layer that calculates the loss function used for connectionist
    temporal classification (CTC).
    See https://www.cs.toronto.edu/~graves/preprint.pdf

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param blank: Sets the value for the blank label that will be used as space.
    :type blank: int
    :param skip: Specifies if blank labels may be skipped when aligning.
    :type skip: bool
    :param    loss_weight:  The multiplier for the loss function value during training.
    :type loss_weight: float, default=1.0
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the network response.
    The dimensions:
    - **BatchLength** is the maximum sequence length
    - **BatchWidth** is the number of sequences in the set
    - **ListSize** is 1
    - **Height** * **Width** * **Depth** * **Channels** is the number of classes

    (2) the correct labels as a blob with int data.
    The dimensions:
    - **BatchLength** is the maximum labels sequence length
    - **BatchWidth** is the number of sequences, same as first input's **BatchWidth**
    - the other dimensions are 1

    (3) (optional): the label sequences lengths as a blob with int data.
    If this input isn't connected, the label sequences are considered to be
    the second input's **BatchLength** long.
    The dimensions:
    - **BatchWidth** is the same as for the first input
    - the other dimensions are 1

    (4) (optional): the network response sequences lengths as a blob
    with int data. If this input isn't connected, the response sequences
    are considered to be the first input's **BatchLength** long.
    The dimensions:
    - **BatchWidth** is the same as for the first input
    - the other dimensions are 1

    (5) (optional): the sequences' weights. 
    The dimensions:
    - **BatchWidth** is the same as for the first input
    - the other dimensions are 1

    .. rubric:: Layer outputs:

    The layer has no output.
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
        """Gets the value of the blank label.
        """
        return self._internal.get_blank_label()

    @blank.setter
    def blank(self, value):
        """Sets the value of the blank label.
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
        """Checks if blank labels may be skipped when aligning.
        """
        return self._internal.get_skip()

    @skip.setter
    def skip(self, value):
        """Specifies if blank labels may be skipped when aligning.
        """
        self._internal.set_skip(bool(value))

# ----------------------------------------------------------------------------------------------------------------------


class CtcDecoding(Layer):
    """The layer that is looking for the most probable sequences 
    in the response of a connectionist temporal classification (CTC) network.

    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: object, (object, int) or list of them        
    :param blank: Sets the value for the blank label that will be used as space.
    :type blank: int
    :param blank_threshold: The probability threshold for blank labels when building
        a linear division graph (LDG).
    :type blank_threshold: float, [0..1]
    :param arc_threshold: The probability threshold for cutting off arcs when building
        a linear division graph (LDG).
    :type arc_threshold: float, [0..1]
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) the network response.
    The dimensions:
    - BatchLength is the maximum sequence length
    - BatchWidth is the number of sequences in the set
    - ListSize is 1
    - Height * Width * Depth * Channels is the number of classes

    (2) (optional): the network response sequences lengths as a blob
    with int data. If this input isn't connected, the response sequences
    are considered to be the first input's BatchLength long.
    The dimensions:
    - BatchWidth is the same as for the first input
    - the other dimensions are 1

    .. rubric:: Layer outputs:

    The layer has no output.
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
        """Gets the value of the blank label.
        """
        return self._internal.get_blank_label()

    @blank.setter
    def blank(self, value):
        """Sets the value of the blank label.
        """
        self._internal.set_blank_label(int(value))

    @property
    def blank_threshold(self):
        """Gets the probability threshold for blank layers when building
        a linear division graph (LDG).
        """
        return self._internal.get_blank_threshold()

    @blank_threshold.setter
    def blank_threshold(self, value):
        """Sets the probability threshold for blank layers when building
        a linear division graph (LDG).
        """
        self._internal.set_blank_threshold(float(value))

    @property
    def arc_threshold(self):
        """Gets the probability threshold for cutting off arcs when building
        a linear division graph (LDG).
        """
        return self._internal.get_arc_threshold()

    @arc_threshold.setter
    def arc_threshold(self, value):
        """Sets the probability threshold for cutting off arcs when building
        a linear division graph (LDG).
        """
        self._internal.set_arc_threshold(float(value))

    @property
    def sequence_length(self):
        """Returns the sequence length.
        """
        return self._internal.get_sequence_length()

    @property
    def batch_width(self):
        """Returns the number of sequences.
        """
        return self._internal.get_batch_width()

    @property
    def label_count(self):
        """Returns the number of classes.
        """
        return self._internal.get_label_count()

    def get_best_sequence(self, sequence_number):
        """Retrieves the most probable sequence for the object
        with sequence_number index in the set.
        """
        return self._internal.get_best_sequence(sequence_number)
