""" Copyright (c) 2017-2021 ABBYY Production LLC

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

import random
from abc import ABCMeta, abstractmethod
import neoml.PythonWrapper as PythonWrapper


class BaseTraits(metaclass=ABCMeta):
    """Base class for working with traits in differential evolution.
    """

    @abstractmethod
    def generate(self, min_value, max_value):
        """Generates a random trait value in the specified bounds.
        
        Parameters
        ---------
        min_value : any type
            The lower bound for the interval.
        max_value : any type
            The upper bound for the interval.
        """

    @abstractmethod
    def less(self, first, second):
        """Checks if the first trait value is smaller than the second.
        
        Parameters
        ---------
        first : any type
            The first value to be compared.
        second : any type
            The second value to be compared.
        """

    @abstractmethod
    def mutate(self, base, left, right, fluctuation, min_value, max_value):
        """Performs mutation for the differential evolution algorithm.
        
        Parameters
        ---------
        base : any type
            A member of the original population.
        left : any type
            Another member of the original population.
        right : any type
            Another member of the original population.
        fluctuation : any type
            The coefficient for mutation.
        min_value : any type
            The lower bound for the mutated value.
        max_value : any type
            The upper bound for the mutated value.
        """

# -------------------------------------------------------------------------------------------------------------


class DoubleTraits(BaseTraits):
    """The implementation of a double parameter.
    """

    def generate(self, min_value, max_value):
        """Generates a random trait value in the specified bounds.
        
        Parameters
        ---------
        min_value : double
            The lower bound for the interval.
        max_value : double
            The upper bound for the interval.
        """
        return random.uniform(min_value, max_value)

    def less(self, first, second):
        """Checks if the first trait value is smaller than the second.
        
        Parameters
        ---------
        first : double
            The first value to be compared.
        second : double
            The second value to be compared.
        """
        return first < second

    def mutate(self, base, left, right, fluctuation, min_value, max_value):
        """Performs mutation for the differential evolution algorithm.
        
        Parameters
        ---------
        base : double
            A member of the original population.
        left : double
            Another member of the original population.
        right : double
            Another member of the original population.
        fluctuation : double
            The coefficient for mutation.
        min_value : double
            The lower bound for the mutated value.
        max_value : double
            The upper bound for the mutated value.
        """
        mute = base + fluctuation * (left - right)
        if mute < min_value:
            mute = min_value + random.random() * (base - min_value)
        elif mute > max_value:
            mute = max_value - random.random() * (max_value - base)

        return min(max(mute, min_value), max_value)

# -------------------------------------------------------------------------------------------------------------


class IntTraits(BaseTraits):
    """The implementation of an integer parameter.
    """

    def generate(self, min_value, max_value):
        """Generates a random trait value in the specified bounds.

        Parameters
        ---------
        min_value : int
            The lower bound for the interval.
        max_value : int
            The upper bound for the interval.
        """
        return int(random.uniform(min_value, max_value))

    def less(self, first, second):
        """Checks if the first trait value is smaller than the second.

        Parameters
        ---------
        first : int
            The first value to be compared.
        second : int
            The second value to be compared.
        """
        return first < second

    def mutate(self, base, left, right, fluctuation, min_value, max_value):
        """Performs mutation for the differential evolution algorithm.

        Parameters
        ---------
        base : int
            A member of the original population.
        left : int
            Another member of the original population.
        right : int
            Another member of the original population.
        fluctuation : double
            The coefficient for mutation.
        min_value : int
            The lower bound for the mutated value.
        max_value : int
            The upper bound for the mutated value.
        """
        mute = base + int(fluctuation * (left - right))
        if mute < min_value:
            mute = min_value + int(random.random() * (base - min_value))
        elif mute > max_value:
            mute = max_value - int(random.random() * (max_value - base))

        return min(max(mute, min_value), max_value)

# -------------------------------------------------------------------------------------------------------------


class DifferentialEvolution:
    """Optimizing function value implementation based on differential evolution
       The purpose of the algorithm is to find the optimal system parameters 
       (represented by a vector of real values X by default) for which 
       the specified function value f(X) will be the closest to the reference value.
       The Evaluate function to calculate f(X) and compare it with the reference
       is provided by the user.
    """
    def __init__(self, f, lower_bounds, upper_bounds, fluctuation=0.5, cross_probability=0.5, population=100,
                 param_traits=None, result_traits=None, max_generation_count=None, max_non_growing_count=None):

        if len(lower_bounds) != len(upper_bounds):
            raise ValueError('`lower_bounds` and `upper_bounds` inputs must be the same length.')

        if param_traits is None:
            param_traits = []
            for i in range(len(lower_bounds)):
                if type(lower_bounds[i]) is float:
                    if type(upper_bounds[i]) is float or type(upper_bounds[i]) is int:
                        param_traits.append(DoubleTraits())
                    else:
                        raise ValueError('You should define `param_traits` for given `bounds`.')
                if type(lower_bounds[i]) is int:
                    if type(upper_bounds[i]) is float:
                        param_traits.append(DoubleTraits())
                    elif type(upper_bounds[i]) is int:
                        param_traits.append(IntTraits())
                    else:
                        raise ValueError('You should define `param_traits` for given `bounds`.')
        else:
            if len(lower_bounds) != len(param_traits):
                raise ValueError('`bounds` and `param_traits` inputs must be the same length.')
            if any(not isinstance(p, BaseTraits) for p in param_traits):
                raise ValueError('`param_traits` must be a list of neoml.BaseTraits implementations.')

        if fluctuation <= 0 or fluctuation >= 1:
            raise ValueError('`fluctuation` must be in (0, 1).')

        if cross_probability <= 0 or cross_probability >= 1:
            raise ValueError('`cross_probability` must be in (0, 1).')

        if population < 5:
            raise ValueError('`population` must be at least 5.')

        if result_traits is None:
            result_traits = DoubleTraits()

        if max_generation_count is None:
            max_generation_count = -1

        if max_non_growing_count is None:
            max_non_growing_count = -1

        if not isinstance(result_traits, BaseTraits):
            raise ValueError('`result_traits` must implement neoml.BaseTraits.')

        self.internal = PythonWrapper.DifferentialEvolution(f, lower_bounds, upper_bounds, param_traits,
                                                            result_traits, float(fluctuation), float(cross_probability),
                                                            int(population), int(max_generation_count),
                                                            int(max_non_growing_count))

    def build_next_generation(self):
        """Builds the next generation.

        Return values
        -------
        success : bool
           Returns True if any of the stop conditions was fulfilled.
        """
        return self.internal.build_next_generation()

    def run(self):
        """Runs optimization until one of the stop conditions is fulfilled.
        """
        return self.internal.run()

    @property
    def population(self):
        """Gets the resulting population.
        """
        return self.internal.get_population()

    @property
    def population_function_values(self):
        """Gets the function values on the resulting population.
        """
        return self.internal.get_population_function_values()

    @property
    def optimal_vector(self):
        """Gets the "best vector".
        """
        return self.internal.get_optimal_vector()
