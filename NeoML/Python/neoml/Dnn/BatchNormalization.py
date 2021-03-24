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


class BatchNormalization(Layer):
    """The layer that performs normalization using the formula:
    bn(x)[i][j] = ((x[i][j] - mean[j]) / sqrt(var[j])) * gamma[j] + beta[j]
    
    - gamma and beta are the trainable parameters
    - mean and var depend on whether the layer is being trained:

        - If the layer is being trained, mean[j] and var[j] are 
          the mean value and the variance of x data with j 
          coordinate across all i.
        - If the layer is not being trained, mean[j] and var[j] are
          the exponential moving mean and the unbiased variance
          estimate calculated during training.
    
    :param input_layer: The input layer and the number of its output. If no number
        is specified, the first output will be connected.
    :type input_layer: object, tuple(object, int)
    :param channel_based: Turns on and off channel-based statistics:

        - If True, mean, var, gamma, and beta in the formula will be 
          vectors of the input **Channels** length. 
          The i coordinate will iterate over all values from 0 to 
          **BatchLength** * **BatchWidth** * **ListSize** * **Height** * **Width** * **Depth** - 1.
        - If False, the mean, var, gamma, and beta vectors 
          will have the **Height** * **Width** * **Depth** * **Channels** length. 
          The i coordinate will iterate over all values from 0 to 
          **BatchLength** * **BatchWidth** * **ListSize** - 1.
    :type channel_based: bool, default=True
    :param zero_free_term: Specifies if the free term (beta) should be trained or filled with zeros.
    :type zero_free_term: bool, default=False
    :param slow_convergence_rate: The coefficient for calculating the exponential moving mean and variance.
    :type slow_convergence_rate: float, default=1.0
    :param name: The layer name.
    :type name: str, default=None
    """

    def __init__(self, input_layer, channel_based, zero_free_term=False, slow_convergence_rate=1.0, name=None):

        if type(input_layer) is PythonWrapper.BatchNormalization:
            super().__init__(input_layer)
            return

        layers, outputs = check_input_layers(input_layer, 1)

        if slow_convergence_rate <= 0 or slow_convergence_rate > 1:
            raise ValueError('The `slow_convergence_rate` must be in (0, 1].')

        internal = PythonWrapper.BatchNormalization(str(name), layers[0], int(outputs[0]), bool(channel_based), bool(zero_free_term), float(slow_convergence_rate))
        super().__init__(internal)

    @property
    def channel_based(self):
        """Sets the channel-based mode.
        """
        return self._internal.get_channel_based()

    @channel_based.setter
    def channel_based(self, channel_based):
        """Gets the channel-based mode.
        """
        self._internal.set_channel_based(bool(channel_based))

    @property
    def slow_convergence_rate(self):
        """Sets the coefficient for calculating 
        the exponential moving mean and variance.
        """
        return self._internal.get_slow_convergence_rate()

    @slow_convergence_rate.setter
    def slow_convergence_rate(self, slow_convergence_rate):
        """Gets the coefficient for calculating 
        the exponential moving mean and variance.
        """
        if slow_convergence_rate <= 0 or slow_convergence_rate > 1:
            raise ValueError('The `slow_convergence_rate` must be in (0, 1].')

        self._internal.set_slow_convergence_rate(float(slow_convergence_rate))

    @property
    def zero_free_term(self):
        """Specifies if the free term should be zero.
        """
        return self._internal.get_zero_free_term()

    @zero_free_term.setter
    def zero_free_term(self, zero_free_term):
        """Indicates if the free term will be zero or trained.
        """
        self._internal.set_zero_free_term(bool(zero_free_term))

    @property
    def final_params(self):
        """Gets the trained parameters as a blob of the dimensions:

        - **BatchLength**, **ListSize**, **Channels** equal to 1
        - **BatchWidth** is 2
        - **Height**, **Width**, **Depth** equal to 1 if in channel-based mode,
          otherwise the same as the dimensions of the input
        """
        return Blob( self._internal.get_final_params() )

    @final_params.setter
    def final_params(self, final_params):
        """Sets the trainable parameters as a blob of the dimensions:

        - **BatchLength**, **ListSize**, **Channels** equal to 1
        - **BatchWidth** is 2
        - **Height**, **Width**, **Depth** equal to 1 if in channel-based mode,
          otherwise the same as the dimensions of the input
        """
        self._internal.set_final_params(final_params._internal)

    @property
    def use_final_params_for_init(self):
        """Checks if the final parameters should be used for initialization.
        """
        return self._internal.get_use_final_params_for_init()

    @use_final_params_for_init.setter
    def use_final_params_for_init(self, use_final_params_for_init):
        """Specifies if the final parameters should be used for initialization.
        """
        self._internal.set_use_final_params_for_init(bool(use_final_params_for_init))
