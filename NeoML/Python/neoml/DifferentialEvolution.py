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
        
        :param min_value: the lower bound for the interval.
        :type min_value: any type

        :param max_value: the upper bound for the interval.
        :type max_value: any type
        """

    @abstractmethod
    def less(self, first, second):
        """Checks if the first trait value is smaller than the second.
        
        :param first: the first value to be compared.
        :type first: any type

        :param second: the second value to be compared.
        :type second: any type
        """

    @abstractmethod
    def mutate(self, base, left, right, fluctuation, min_value, max_value):
        """Performs mutation for the differential evolution algorithm.

        :param base: a member of the original population.
        :type base: any type

        :param left: another member of the original population.
        :type left: any type

        :param right: another member of the original population.
        :type right: any type

        :param fluctuation: the coefficient for mutation.
        :type fluctuation: any type

        :param min_value: the lower bound for the mutated value.
        :type min_value: any type

        :param max_value: the upper bound for the mutated value.
        :type max_value: any type
        """

# -------------------------------------------------------------------------------------------------------------


class DoubleTraits(BaseTraits):
    """The implementation of a double parameter.
    """

    def generate(self, min_value, max_value):
        """Generates a random trait value in the specified bounds.
        
        :param min_value: the lower bound for the interval.
        :type min_value: double

        :param max_value: the upper bound for the interval.
        :type max_value: double
        """
        return random.uniform(min_value, max_value)

    def less(self, first, second):
        """Checks if the first trait value is smaller than the second.

        :param first: the first value to be compared.
        :type first: double

        :param second: the second value to be compared.
        :type second: double
        """
        return first < second

    def mutate(self, base, left, right, fluctuation, min_value, max_value):
        """Performs mutation for the differential evolution algorithm.

        :param base: a member of the original population.
        :type base: double

        :param left: another member of the original population.
        :type left: double

        :param right: another member of the original population.
        :type right: double

        :param fluctuation: the coefficient for mutation.
        :type fluctuation: double

        :param min_value: the lower bound for the mutated value.
        :type min_value: double

        :param max_value: the upper bound for the mutated value.
        :type max_value: double
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

        :param min_value: the lower bound for the interval.
        :type min_value: int

        :param max_value: the upper bound for the interval.
        :type max_value: int
        """
        return int(random.uniform(min_value, max_value))

    def less(self, first, second):
        """Checks if the first trait value is smaller than the second.

        :param first: the first value to be compared.
        :type first: int

        :param second: the second value to be compared.
        :type second: int
        """
        return first < second

    def mutate(self, base, left, right, fluctuation, min_value, max_value):
        """Performs mutation for the differential evolution algorithm.

        :param base: a member of the original population.
        :type base: int

        :param left: another member of the original population.
        :type left: int

        :param right: another member of the original population.
        :type right: int

        :param fluctuation: the coefficient for mutation.
        :type fluctuation: double

        :param min_value: the lower bound for the mutated value.
        :type min_value: int

        :param max_value: the upper bound for the mutated value.
        :type max_value: int
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
       
       :param f: the function to be optimized.
       :type f: object
       
       :param lower_bounds: the minimum values for the parameters.
       :type lower_bounds: array of the same length as upper_bounds
       
       :param upper_bounds: the maximum values for the parameters.
       :type upper_bounds: array of the same length as lower_bounds
       
       :param fluctuation: the fluctuation coefficient.
       :type fluctuation: float, default=0.5
       
       :param cross_probability: the mutation probability.
       :type cross_probability: float, default=0.5
       
       :param population: the number of elements in each generation.
       :type population: int, >=5, default=100
       
       :param param_traits: the parameter types.
       :type param_traits: array of ``BaseTraits``-implementing objects, default=None
       
       :param result_traits: the result types.
       :type result_traits: array of ``BaseTraits``-implementing objects, default=None
       
       :param max_generation_count: the maximum number of generations after which the algorithm stops.
       :type max_generation_count: int, default=None
       
       :param max_non_growing_count: the maximum number of iterations for which the function minimum has been unchanged.
       :type max_non_growing_count: int, default=None
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

        :return: if the stop condition was reached.
        :rtype: *bool*
        """
        return self.internal.build_next_generation()

    def run(self):
        """Runs optimization until one of the stop conditions is fulfilled.
        """
        return self.internal.run()

    @property
    def population(self):
        """Gets the resulting population.

        :return: the population on the current step.
        :rtype: *array-like of shape {population, vector_length}*
        """
        return self.internal.get_population()

    @property
    def population_function_values(self):
        """Gets the function values on the resulting population.

        :return: the function values
        :rtype: *array-like of shape {population,}*
        """
        return self.internal.get_population_function_values()

    @property
    def optimal_vector(self):
        """Gets the "best vector."

        :return: the best fector found
        :rtype: *array-like of shape {vector_length,}*
        """
        return self.internal.get_optimal_vector()
