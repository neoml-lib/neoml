.. _py-submodule-dnn:

#########
neoml.Dnn
#########

- :ref:`py-dnn-network`
- :ref:`py-dnn-layers`

   - :ref:`py-dnn-recurrent`

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
~~~~

.. autoclass:: neoml.Dnn.Lstm
   :members:

Qrnn
~~~~

.. autoclass:: neoml.Dnn.Qrnn
   :members:

Gru
~~~

.. autoclass:: neoml.Dnn.Gru
   :members:

.. _py-dnn-activation:

Activation layers
*****************

.. autoclass:: neoml.Dnn.Linear
   :members:
.. autoclass:: neoml.Dnn.ELU
   :members:
.. autoclass:: neoml.Dnn.ReLU
   :members:
.. autoclass:: neoml.Dnn.LeakyReLU
   :members:
.. autoclass:: neoml.Dnn.HSwish
   :members:
.. autoclass:: neoml.Dnn.GELU
   :members:
.. autoclass:: neoml.Dnn.Abs
   :members:
.. autoclass:: neoml.Dnn.Sigmoid
   :members:
.. autoclass:: neoml.Dnn.Tanh
   :members:
.. autoclass:: neoml.Dnn.HardTanh
   :members:
.. autoclass:: neoml.Dnn.HardSigmoid
   :members:
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
