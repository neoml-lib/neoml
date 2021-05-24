.. _py-submodule-misc:

###################
neoml.Algorithms
###################

NeoML library also provides some other algorithms.

Differential evolution
######################

This method finds the global minimum (or maximum) of a non-differentiable, non-linear, multimodal function of many variables F(x1, x2, ..., xn).

It uses mutation, crossover and selection to transform the current population (that is, the function parameters) into the next generation so that the function values on the new population "improve." The process is repeated until the stop criteria are met.

Class description
*****************

.. autoclass:: neoml.DifferentialEvolution.DifferentialEvolution
   :members:

Parameter traits
****************

Define your own parameter types:

.. autoclass:: neoml.DifferentialEvolution.BaseTraits
   :members:

Predefined types for integer and double parameters:

.. autoclass:: neoml.DifferentialEvolution.IntTraits
   :members:

.. autoclass:: neoml.DifferentialEvolution.DoubleTraits
   :members: