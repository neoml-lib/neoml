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
    """
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.Solver):
            raise ValueError('The `internal` must be PythonWrapper.Solver')

        self._internal = internal

    def train(self):
        """
        """
        self._internal.train()

    def reset(self):
        """
        """
        self._internal.reset()

    @property
    def learning_rate(self):
        """
        """
        return self._internal.get_learning_rate()

    @learning_rate.setter
    def learning_rate(self, value):
        """
        """
        self._internal.set_learning_rate(value)

    @property
    def l2(self):
        """
        """
        return self._internal.get_l2()

    @l2.setter
    def l2(self, value):
        """
        """
        self._internal.set_l2(value)

    @property
    def l1(self):
        """
        """
        return self._internal.get_l1()

    @l1.setter
    def l1(self, value):
        """
        """
        self._internal.set_l1(value)

    @property
    def max_gradient_norm(self):
        """
        """
        return self._internal.get_max_gradient_norm()

    @max_gradient_norm.setter
    def max_gradient_norm(self, value):
        """
        """
        self._internal.set_max_gradient_norm(value)

# -------------------------------------------------------------------------------------------------------------


class SimpleGradient(Solver):
    """
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
        """
        """
        return self._internal.get_moment_decay_rate()

    @moment_decay_rate.setter
    def moment_decay_rate(self, value):
        """
        """
        self._internal.set_moment_decay_rate(value)

# -------------------------------------------------------------------------------------------------------------


class AdaptiveGradient(Solver):
    """
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
        """
        """
        return self._internal.get_moment_decay_rate()

    @moment_decay_rate.setter
    def moment_decay_rate(self, value):
        """
        """
        self._internal.set_moment_decay_rate(value)

    @property
    def second_moment_decay_rate(self):
        """
        """
        return self._internal.get_second_moment_decay_rate()

    @second_moment_decay_rate.setter
    def second_moment_decay_rate(self, value):
        """
        """
        self._internal.set_second_moment_decay_rate(value)

    @property
    def epsilon(self):
        """
        """
        return self._internal.get_epsilon()

    @epsilon.setter
    def epsilon(self, value):
        """
        """
        self._internal.set_epsilon(value)

    @property
    def ams_grad(self):
        """
        """
        return self._internal.get_ams_grad()

    @ams_grad.setter
    def ams_grad(self, value):
        """
        """
        self._internal.set_ams_grad(value)

# -------------------------------------------------------------------------------------------------------------


class NesterovGradient(Solver):
    """
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
        """
        """
        return self._internal.get_moment_decay_rate()

    @moment_decay_rate.setter
    def moment_decay_rate(self, value):
        """
        """
        self._internal.set_moment_decay_rate(value)

    @property
    def second_moment_decay_rate(self):
        """
        """
        return self._internal.get_second_moment_decay_rate()

    @second_moment_decay_rate.setter
    def second_moment_decay_rate(self, value):
        """
        """
        self._internal.set_second_moment_decay_rate(value)

    @property
    def epsilon(self):
        """
        """
        return self._internal.get_epsilon()

    @epsilon.setter
    def epsilon(self, value):
        """
        """
        self._internal.set_epsilon(value)

    @property
    def ams_grad(self):
        """
        """
        return self._internal.get_ams_grad()

    @ams_grad.setter
    def ams_grad(self, value):
        """
        """
        self._internal.set_ams_grad(value)
