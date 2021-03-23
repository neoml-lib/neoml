.. _py-submodule-dnn:

#########
neoml.Dnn
#########

- :ref:`py-dnn-network`
- :ref:`py-dnn-layers`

   - :ref:`py-dnn-recurrent`
   - :ref:`py-dnn-activation`

.. _py-dnn-network:

Neural network
##############

.. _py-dnn-layers:

Layers
######

.. _py-dnn-recurrent:

Recurrent layers
****************

Lstm
====

.. autoclass:: neoml.Dnn.Lstm
   :members:

Qrnn
====

.. autoclass:: neoml.Dnn.Qrnn
   :members:

Gru
===

.. autoclass:: neoml.Dnn.Gru
   :members:

.. _py-dnn-activation:

Activation layers
*****************

Linear
======

.. autoclass:: neoml.Dnn.Linear
   :members:

ELU
===

.. autoclass:: neoml.Dnn.ELU
   :members:

ReLU
====

.. autoclass:: neoml.Dnn.ReLU
   :members:

LeakyReLU
=========

.. autoclass:: neoml.Dnn.LeakyReLU
   :members:

HSwish
======

.. autoclass:: neoml.Dnn.HSwish
   :members:

GELU
====

.. autoclass:: neoml.Dnn.GELU
   :members:

Abs
===
.. autoclass:: neoml.Dnn.Abs
   :members:

Sigmoid
=======

.. autoclass:: neoml.Dnn.Sigmoid
   :members:

Tanh
====

.. autoclass:: neoml.Dnn.Tanh
   :members:

HardTanh
========
.. autoclass:: neoml.Dnn.HardTanh
   :members:

HardSigmoid
===========

.. autoclass:: neoml.Dnn.HardSigmoid
   :members:

Power
=====

.. autoclass:: neoml.Dnn.Power
   :members:

Binarization layers
*******************

.. autoclass:: neoml.Dnn.EnumBinarization
   :members:
.. autoclass:: neoml.Dnn.BitSetVectorization
   :members:

AccumulativeLookup layer
************************

.. autoclass:: neoml.Dnn.AccumulativeLookup
   :members:

Accuracy layers
***************

.. autoclass:: neoml.Dnn.Accuracy
   :members:
.. autoclass:: neoml.Dnn.ConfusionMatrix
   :members:

AddToObject layer
*****************
