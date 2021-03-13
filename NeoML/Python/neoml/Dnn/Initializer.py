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

import neoml
import neoml.PythonWrapper as PythonWrapper
import neoml.Random as Random


class Initializer:
    """The base class for initializing trainable weights of a layer 
    before the first training iteration.
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.Initializer):
            raise ValueError('The `internal` must be PythonWrapper.Initializer')

        self._internal = internal

    @property
    def random(self):
        """Gets the random numbers generator used in the initializer.
        """
        return Random.Random(self._internal.get_random())

# -------------------------------------------------------------------------------------------------------------


class Xavier(Initializer):
    """Initializes a blob using the Xavier algorithm: with random values
    from a normal distribution over (0, 1 / input_size).
    
    Parameters
    ---------
    random_generator : object, default=None
        Sets the random numbers generator to be used.
        By default, the standard NeoML randomizer with 42 seed is used.
    """
    def __init__(self, random_generator=None):
        if isinstance(random_generator, PythonWrapper.Xavier):
            super().__init__(random_generator)
            return

        if random_generator is None:
            random_generator = neoml.Random.Random(42)

        if not isinstance(random_generator, neoml.Random.Random):
            raise ValueError('The `random_generator` must be a neoml.Random.Random.')

        internal = PythonWrapper.Xavier(random_generator)
        super().__init__(internal)

# -------------------------------------------------------------------------------------------------------------


class Uniform(Initializer):
    """Initializes a blob using uniform distribution between the set bounds.
    
    Parameters
    ---------
    lower_bound : float, default=-1.0
        The lower bound of the distribution.
    upper_bound : float, default=1.0
        The upper bound of the distribution.
    random_generator : object, default=None
        Sets the random numbers generator to be used.
        By default, the standard NeoML randomizer with 42 seed is used.
    """
    def __init__(self, lower_bound=-1.0, upper_bound=1.0, random_generator=None):
        if isinstance(lower_bound, PythonWrapper.Uniform):
            super().__init__(lower_bound)
            return

        if random_generator is None:
            random_generator = neoml.Random.Random(42)

        if not isinstance(random_generator, neoml.Random.Random):
            raise ValueError('The `random_generator` must be a neoml.Random.Random.')

        internal = PythonWrapper.Uniform(float(lower_bound), float(upper_bound), random_generator)
        super().__init__(internal)

    @property
    def lower_bound(self):
        """Gets the distribution lower bound.
        """
        return self._internal.get_lower_bound()

    @lower_bound.setter
    def lower_bound(self, value):
        """Sets the distribution lower bound.
        """
        self._internal.set_lower_bound(value)

    @property
    def upper_bound(self):
        """Gets the distibution upper bound.
        """
        return self._internal.get_upper_bound()

    @upper_bound.setter
    def upper_bound(self, value):
        """Sets the distribution upper bound.
        """
        self._internal.set_upper_bound(value)
