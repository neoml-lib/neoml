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


class Crf(Layer):
    """The layer that trains and calculates transitional probabilities
    in a conditional random field (CRF).
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int) of list of them
    :param class_count: The number of classes in the CRF.
    :type class_count: int
    :param padding: The number of empty class used to fill the sequence end.
    :type padding: int, default=0
    :param dropout_rate: Variational dropout.
    :type dropout_rate: float, default=0.0
    :param name: The layer name.
    :type name: str, default=None

    .. rubric:: Layer inputs:

    (1) a blob with the object sequences.
        The dimensions:

        - **BatchLength** is the sequence length
        - **BatchWidth** is the number of sequences in the set
        - **ListSize** is 1
        - **Height** * **Width** * **Depth** * **Channels** is the object size
    
    (2) (optional): a blob with integer data that contains the correct class
        sequences. It is required only for training.
        The dimensions:

        - **BatchLength**, **BatchWidth** equal to the first input's
        - the other dimensions are equal to 1
    
    .. rubric:: Layer outputs:

    (1) (optional): a blob with integer data that contains optimal class 
        sequences. 
        This output will be returned only if you set calc_prev_best_class to True.
        During training, this output usually isn't needed 
        and is switched off by default.
        The dimensions:

        - **BatchLength**, **BatchWidth** equal to the first input's
        - **Channels** equal to the number of classes
        - the other dimensions are equal to 1
    
    (2) a blob with float data that contains non-normalized logarithm of 
        optimal class sequences probabilities.
        The dimensions are the same as for the first output.
    
    (3) (optional): a blob with non-normalized logarithm of the correct class
        sequences probabilities. 
        This output will be there only if the layer has two inputs.
        The dimensions are equal to the second input's:

        - **BatchLength**, **BatchWidth** equal to the first input's
        - the other dimensions are equal to 1
    """

    def __init__(self, input_layer, class_count, padding=0, dropout_rate=0.0, name=None):

        if type(input_layer) is PythonWrapper.Crf:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, (1, 2))

        if class_count < 1:
            raise ValueError('The `class_count` must be > 0.')

        if padding < 0:
            raise ValueError('The `padding` must be >= 0.')

        if float(dropout_rate) < 0 or float(dropout_rate) >= 1:
            raise ValueError('The `dropout_rate` must be in [0, 1).')

        internal = PythonWrapper.Crf(str(name), layers, outputs, class_count, padding, dropout_rate)
        super().__init__(internal)

    @property
    def class_count(self):
        """Gets the number of classes in the CRF.
        """
        return self._internal.get_class_count()

    @class_count.setter
    def class_count(self, class_count):
        """Sets the number of classes in the CRF.
        """
        if int(class_count) < 1:
            raise ValueError('The `class_count` must be > 0.')

        self._internal.set_class_count(int(class_count))

    @property
    def padding(self):
        """Gets the number of the empty class.
        """
        return self._internal.get_padding()

    @padding.setter
    def padding(self, padding):
        """Sets the number of the empty class.
        """
        if int(padding) < 0:
            raise ValueError('The `padding` must be >= 0.')

        self._internal.set_padding(int(padding))

    @property
    def dropout_rate(self):
        """Gets the variational dropout rate.
        """
        return self._internal.get_dropout_rate()

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate):
        """Sets the variational dropout rate.
        """
        if float(dropout_rate) < 0 or float(dropout_rate) >= 1:
            raise ValueError('The `dropout_rate` must be in [0, 1).')

        self._internal.set_dropout_rate(float(dropout_rate))

    @property
    def calc_best_prev_class(self):
        """Checks if the first output will be returned.
        """
        return self._internal.get_calc_best_prev_class()

    @calc_best_prev_class.setter
    def calc_best_prev_class(self, calc_best_prev_class):
        """Specifies if the first output should be returned.
        """
        self._internal.set_calc_best_prev_class(bool(calc_best_prev_class))

    @property
    def hidden_weights(self):
        """Gets the hidden layer weights. The dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** is the number of classes
        - **Height** * **Width** * **Depth** * **Channels** the same as for the first input
        """
        return self._internal.get_hidden_weights()

    @hidden_weights.setter
    def hidden_weights(self, hidden_weights):
        """Sets the hidden layer weights. The dimensions:

        - **BatchLength** * **BatchWidth** * **ListSize** is class_count
        - **Height** * **Width** * **Depth** * **Channels** the same as for the first input
        """
        self._internal.set_hidden_weights(hidden_weights)

    @property
    def free_terms(self):
        """Gets the hidden layer free terms. The blob size is class_count.
        """
        return self._internal.get_free_terms()

    @free_terms.setter
    def free_terms(self, free_terms):
        """Sets the hidden layer free terms. The blob size is class_count.
        """
        self._internal.set_free_terms(free_terms)

    @property
    def transitions(self):
        """Gets the transition probability matrix. The dimensions:

        - **BatchLength**, **BatchWidth** are class_count
        - the other dimensions are 1
        """
        return self._internal.get_transitions()

    @transitions.setter
    def transitions(self, transitions):
        """Sets the transition probability matrix. The dimensions:

        - **BatchLength**, **BatchWidth** are class_count
        - the other dimensions are 1
        """
        self._internal.set_transitions(transitions)

# ----------------------------------------------------------------------------------------------------------------------


class CrfLoss(Layer):
    """The layer that calculates the loss function used for training a CRF.
    The value is -log(probability of the correct class sequence)
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: array of (object, int) tuples or objects
    :param loss_weight: The multiplier for the loss function value during training.
    :type loss_weight:  float, default=1.0
    :param name: The layer name.
    :type name: str, default=None
        
    .. rubric:: Layer inputs:

    (1) a blob with integer data that contains optimal class sequences.
        The dimensions:
        - **BatchLength**, **BatchWidth** equal to the network inputs'
        - **Channels** equal to the number of classes
        - the other dimensions are equal to 1
    
    (2) a blob with float data containing non-normalized logarithm 
        of probabilities of the optimal class sequences.
        The dimensions are the same as for the first input.
    
    (3) a blob with float data containing non-normalized logarithm
        of probability of the correct class being in this position.
        The dimensions:

        - **BatchLength**, **BatchWidth** the same as for the first input
        - the other dimensions equal to 1

    .. rubric:: Layer outputs:

    The layer has no output.
    """
    def __init__(self, input_layers, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.CrfLoss:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 3)

        internal = PythonWrapper.CrfLoss(str(name), layers, outputs, float(loss_weight))
        super().__init__(internal)

    @property
    def last_loss(self):
        """Gets the value of the loss function on the last step.
        """
        return self._internal.get_last_loss()

    @property
    def loss_weight(self):
        """Gets the multiplier for the loss function value dduring training.
        """
        return self._internal.get_loss_weight()

    @loss_weight.setter
    def loss_weight(self, loss_weight):
        """Sets the multiplier for the loss function value during training.
        """
        self._internal.set_loss_weight(float(loss_weight))

    @property
    def max_gradient(self):
        """Gets the gradient clipping threshold.
        """
        return self._internal.get_max_gradient()

    @max_gradient.setter
    def max_gradient(self, max_gradient):
        """Sets the gradient clipping threshold.
        """
        self._internal.set_max_gradient(float(max_gradient))

# ----------------------------------------------------------------------------------------------------------------------


class BestSequence(Layer):
    """The layer that finds the optimal class sequence 
    using the Crf layer output.
    
    :param input_layers: The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    :type input_layers: list of object, tuple(object, int)
    :param name: The layer name.
    :type name: str, default=None
        
    .. rubric:: Layer inputs:

    (1) first output of Crf. A blob with int data that contains the optimal
        class sequences.
        The dimensions:

        - **BatchLength** is the sequence length
        - **BatchWidth** is the number of sequences in the set
        - **Channels** is the number of classes
        - all other dimensions are 1
    
    (2) second output of Crf. A blob with float data that contains
        non-normalized logarithm of optimal class sequences probabilities.
        The dimensions are the same as for the first input.    
    
    .. rubric:: Layer outputs:

    (1) a blob with int data that contains the optimal class sequence.
        The dimensions:

        - **BatchLength**, **BatchWidth** are the same as for the inputs
        - the other dimensions are 1
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.BestSequence:
            super().__init__(input_layers)
            return

        layers, outputs = check_input_layers(input_layers, 2)

        internal = PythonWrapper.BestSequence(str(name), layers, outputs)
        super().__init__(internal)

