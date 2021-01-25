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
import neoml.Dnn as Dnn
import neoml.Utils as Utils


class Loss(Dnn.Layer):
    """The base class for layers estimating error.
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.LossLayer):
            raise ValueError('The `internal` must be PythonWrapper.LossLayer')

        super().__init__(internal)

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
    def train_labels(self):
        """Checks if gradients should also be calculated for the second input,
        which contains the class labels.
        """
        return self._internal.get_train_labels()

    @train_labels.setter
    def train_labels(self, train):
        """Specifies if gradients should also be calculated for the second input,
        which contains the class labels.
        """
        self._internal.set_train_labels(train)

    @property
    def max_gradient(self):
        """Gets the upper limit for the absolute value of the function gradient.
        """
        return self._internal.get_max_gradient()

    @max_gradient.setter
    def max_gradient(self, max_value):
        """Sets the upper limit for the absolute value of the function gradient.
        """
        self._internal.set_max_gradient(max_value)

# ----------------------------------------------------------------------------------------------------------------------


class CrossEntropyLoss(Loss):
    """The layer that calculates the loss value as cross-entropy 
    between the result and the standard:
    loss = -sum(y_i * log(z_i)),
    where for each i class 
        y_i represents the class label, 
        z_i is the network response (with softmax applied or not)
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    softmax : bool, default=True
        Specifies if softmax function should be applied to the result.
    loss_weight : float, default=1.0
        The multiplier for the loss function value during training.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, softmax=True, loss_weight=1.0, name=None):

        if type(input_layers) is PythonWrapper.CrossEntropyLoss:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.CrossEntropyLoss(str(name), layers, outputs, bool(softmax), float(loss_weight))
        super().__init__(internal)

    @property
    def apply_softmax(self):
        """Checks is softmax function should be applied to the result.
        """
        return self._internal.get_apply_softmax()

    @apply_softmax.setter
    def apply_softmax(self, value):
        """Specifies if softmax function should be applied to the result.
        """
        self._internal.set_apply_softmax(int(value))

# ----------------------------------------------------------------------------------------------------------------------


class BinaryCrossEntropyLoss(Loss):
    """The layer that calculates the cross-entropy loss function
    for binary classification:
    loss = y * -log(sigmoid(x)) + (1 - y) * -log(1 - sigmoid(x)), where
        x is the network response,
        y is the correct class label (can be -1 or 1)
    
    Parameters
    ---------------
    input_layers : array of (object, int) tuples or objects
        The input layers to be connected. 
        The integer in each tuple specifies the number of the output.
        If not set, the first output will be used.
    name : str, default=None
        The layer name.
    """
    def __init__(self, input_layers, name=None):

        if type(input_layers) is PythonWrapper.BinaryCrossEntropyLoss:
            super().__init__(input_layers)
            return

        layers, outputs = Utils.check_input_layers(input_layers, (2, 3))

        internal = PythonWrapper.BinaryCrossEntropyLoss(str(name), layers, outputs)
        super().__init__(internal)

    @property
    def positive_weight(self):
        """Gets the multiplier for the term that corresponds 
        to the correct results.
        """
        return self._internal.get_positive_weight()

    @positive_weight.setter
    def positive_weight(self, weight):
        """Sets the multiplier for the term that corresponds
        to the correct results. Tune this value to prioritize 
        precision (positive_weight < 1) or accuracy (positive_weight > 1).
        """
        self._internal.set_positive_weight(weight)
