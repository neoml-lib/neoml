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


class Solver:
    """The base optimizer class. Sets the rules to update weights
    when training a neural network.
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.Solver):
            raise ValueError('The `internal` must be PythonWrapper.Solver')

        self._internal = internal

    def train(self):
        """Modifies the trainable parameters of the network layers, using
        the accumulated gradients and previous steps' history (moment, etc.).
        """
        self._internal.train()

    def reset(self):
        """Resets to initial state.
        """
        self._internal.reset()

    @property
    def learning_rate(self):
        """Gets the learning rate.
        """
        return self._internal.get_learning_rate()

    @learning_rate.setter
    def learning_rate(self, value):
        """Sets the learning rate.
        """
        self._internal.set_learning_rate(value)

    @property
    def l2(self):
        """Gets the L2 regularization parameter.
        """
        return self._internal.get_l2()

    @l2.setter
    def l2(self, value):
        """Sets the L2 regularization parameter.
        """
        self._internal.set_l2(value)

    @property
    def l1(self):
        """Gets the L1 regularization parameter.
        """
        return self._internal.get_l1()

    @l1.setter
    def l1(self, value):
        """Sets the L1 regularization parameter.
        """
        self._internal.set_l1(value)

    @property
    def max_gradient_norm(self):
        """Gets the upper limit for gradient norm.
        """
        return self._internal.get_max_gradient_norm()

    @max_gradient_norm.setter
    def max_gradient_norm(self, value):
        """Sets the upper limit for gradient norm.
        Set to a negative value to have no limit.
        """
        self._internal.set_max_gradient_norm(value)

# -------------------------------------------------------------------------------------------------------------


class SimpleGradient(Solver):
    """Stochastic gradient descent with moment.
    
    :param math_engine: The math engine to be used for calculations.
    :type math_engine: object
    :param learning_rate: The learning rate.
    :type learning_rate: float, default=0.01
    :param l1: The L1 regularization parameter.    
    :type l1: float, default=0
    :param l2: The L2 regularization parameter.    
    :type l2: float, default=0
    :param max_gradient_norm: The upper limit for gradient norm.
        A negative value means no limit, which is also the default setting.  
    :type max_gradient_norm: float, default=-1.0
    :param moment_decay_rate: The moment decay rate. Moment is a weighted sum of previous gradients.
    :type moment_decay_rate: float, default=0.9
        
    """
    def __init__(self, math_engine, learning_rate=0.01, l1=0, l2=0, max_gradient_norm=-1.0, moment_decay_rate=0.9):
        if isinstance(math_engine, PythonWrapper.SimpleGradient):
            super().__init__(math_engine)
            return

        internal = PythonWrapper.SimpleGradient(math_engine._internal, float(learning_rate), float(l1), float(l2),
                                                float(max_gradient_norm), float(moment_decay_rate))
        super().__init__(internal)

    @property
    def moment_decay_rate(self):
        """Gets the moment decay rate. 
        Moment is a weighted sum of previous gradients.
        """
        return self._internal.get_moment_decay_rate()

    @moment_decay_rate.setter
    def moment_decay_rate(self, value):
        """Sets the moment decay rate. 
        Moment is a weighted sum of previous gradients.
        """
        self._internal.set_moment_decay_rate(value)

# -------------------------------------------------------------------------------------------------------------


class AdaptiveGradient(Solver):
    """Gradient descent with adaptive momentum (Adam).
    
    :param math_engine: The math engine to be used for calculations.
    :type math_engine: object
    :param learning_rate: The learning rate.
    :type learning_rate: float, default=0.01
    :param l1: The L1 regularization parameter.
    :type l1: float, default=0
    :param l2: The L2 regularization parameter.
    :type l2: float, default=0
    :param max_gradient_norm: The upper limit for gradient norm.
        A negative value means no limit, which is also the default setting.
    :type max_gradient_norm: float, default=-1.0
    :param moment_decay_rate: The moment decay rate. Moment is a weighted sum of previous gradients.   
    :type moment_decay_rate: float, default=0.9
    :param second_moment_decay_rate: The decay rate for the weighted sum of previous gradients, squared,
        also called the second moment.  
    :type second_moment_decay_rate: float, default=0.99
    :param epsilon: The small value used to avoid division by zero 
        when calculating second moment.
    :type epsilon: float, default=1e-6
    :param ams_grad: Turns AMSGrad mode on or off.
        AMSGrad helps against divergence and rapid vanishing of previous states
        memory, which may become a problem for the optimizers that use 
        the moving mean for squared gradient history (Adam, NAdam, RMSprop).
        See https://openreview.net/pdf?id=ryQu7f-RZ.   
    :type ams_grad: bool, default=False
    """
    def __init__(self, math_engine, learning_rate=0.01, l1=0, l2=0, max_gradient_norm=-1.0, moment_decay_rate=0.9,
                 second_moment_decay_rate=0.99, epsilon=1e-6, ams_grad=False):
        if isinstance(math_engine, PythonWrapper.AdaptiveGradient):
            super().__init__(math_engine)
            return

        internal = PythonWrapper.AdaptiveGradient(math_engine._internal, float(learning_rate), float(l1), float(l2),
                                                  float(max_gradient_norm), float(moment_decay_rate),
                                                  float(second_moment_decay_rate), float(epsilon), float(ams_grad))

        super().__init__(internal)

    @property
    def moment_decay_rate(self):
        """Gets the moment decay rate. 
        Moment is a weighted sum of previous gradients.
        """
        return self._internal.get_moment_decay_rate()

    @moment_decay_rate.setter
    def moment_decay_rate(self, value):
        """Sets the moment decay rate. 
        Moment is a weighted sum of previous gradients.
        """
        self._internal.set_moment_decay_rate(value)

    @property
    def second_moment_decay_rate(self):
        """Gets the decay rate for the weighted sum of previous gradients, 
        squared - that is, the second moment.
        """
        return self._internal.get_second_moment_decay_rate()

    @second_moment_decay_rate.setter
    def second_moment_decay_rate(self, value):
        """Sets the decay rate for the weighted sum of previous gradients, 
        squared - that is, the second moment.
        """
        self._internal.set_second_moment_decay_rate(value)

    @property
    def epsilon(self):
        """Gets the small value used to avoid division by zero when
        calculating second moment.
        """
        return self._internal.get_epsilon()

    @epsilon.setter
    def epsilon(self, value):
        """Sets the small value used to avoid division by zero when
        calculating second moment.
        """
        self._internal.set_epsilon(value)

    @property
    def ams_grad(self):
        """Checks if AMSGrad mode is on.
        """
        return self._internal.get_ams_grad()

    @ams_grad.setter
    def ams_grad(self, value):
        """Turns AMSGrad mode on. May be called only before training starts.
        AMSGrad helps against divergence and rapid vanishing of previous states
        memory, which may become a problem for the optimizers that use 
        the moving mean for squared gradient history (Adam, NAdam, RMSprop).
        See https://openreview.net/pdf?id=ryQu7f-RZ.
        """
        self._internal.set_ams_grad(value)

# -------------------------------------------------------------------------------------------------------------


class NesterovGradient(Solver):
    """The optimizer that uses Nesterov moment.
    See http://cs229.stanford.edu/proj2015/054_report.pdf (Algo 8).
    
    :param math_engine: The math engine to be used for calculations.
    :type math_engine: object
    :param learning_rate: The learning rate.    
    :type learning_rate: float, default=0.01
    :param l1: The L1 regularization parameter.
    :type l1: float, default=0
    :param l2: The L2 regularization parameter.
    :type l2: float, default=0
    :param max_gradient_norm: The upper limit for gradient norm.
        A negative value means no limit, which is also the default setting.
    :type max_gradient_norm: float, default=-1.0
    :param moment_decay_rate: The moment decay rate. Moment is a weighted sum of previous gradients.   
    :type moment_decay_rate: float, default=0.9
    :param second_moment_decay_rate: The decay rate for the weighted sum of previous gradients, squared,
        also called the second moment.
    :type second_moment_decay_rate: float, default=0.99
    :param epsilon: The small value used to avoid division by zero 
        when calculating second moment.    
    :type epsilon: float, default=1e-6
    :param ams_grad: Turns AMSGrad mode on or off.
        AMSGrad helps against divergence and rapid vanishing of previous states
        memory, which may become a problem for the optimizers that use 
        the moving mean for squared gradient history (Adam, NAdam, RMSprop).
        See https://openreview.net/pdf?id=ryQu7f-RZ.
    :type ams_grad: bool, default=False 
    """
    def __init__(self, math_engine, learning_rate=0.01, l1=0, l2=0, max_gradient_norm=-1.0, moment_decay_rate=0.9,
                 second_moment_decay_rate=0.99, epsilon=1e-6, ams_grad=False):
        if isinstance(math_engine, PythonWrapper.NesterovGradient):
            super().__init__(math_engine)
            return

        internal = PythonWrapper.NesterovGradient(math_engine._internal, float(learning_rate), float(l1), float(l2),
                                                  float(max_gradient_norm), float(moment_decay_rate),
                                                  float(second_moment_decay_rate), float(epsilon), float(ams_grad))
        super().__init__(internal)

    @property
    def moment_decay_rate(self):
        """Gets the moment decay rate. 
        Moment is a weighted sum of previous gradients.
        """
        return self._internal.get_moment_decay_rate()

    @moment_decay_rate.setter
    def moment_decay_rate(self, value):
        """Sets the moment decay rate. 
        Moment is a weighted sum of previous gradients.
        """
        self._internal.set_moment_decay_rate(value)

    @property
    def second_moment_decay_rate(self):
        """Gets the decay rate for the weighted sum of previous gradients, 
        squared - that is, the second moment.
        """
        return self._internal.get_second_moment_decay_rate()

    @second_moment_decay_rate.setter
    def second_moment_decay_rate(self, value):
        """Sets the decay rate for the weighted sum of previous gradients, 
        squared - that is, the second moment.
        """
        self._internal.set_second_moment_decay_rate(value)

    @property
    def epsilon(self):
        """Gets the small value used to avoid division by zero when
        calculating second moment.
        """
        return self._internal.get_epsilon()

    @epsilon.setter
    def epsilon(self, value):
        """Sets the small value used to avoid division by zero when
        calculating second moment.
        """
        self._internal.set_epsilon(value)

    @property
    def ams_grad(self):
        """Checks if AMSGrad mode is on.
        """
        return self._internal.get_ams_grad()

    @ams_grad.setter
    def ams_grad(self, value):
        """Turns AMSGrad mode on. May be called only before training starts.
        AMSGrad helps against divergence and rapid vanishing of previous states
        memory, which may become a problem for the optimizers that use 
        the moving mean for squared gradient history (Adam, NAdam, RMSprop).
        See https://openreview.net/pdf?id=ryQu7f-RZ.
        """
        self._internal.set_ams_grad(value)
