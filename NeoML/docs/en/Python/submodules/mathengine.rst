.. _py-submodule-mathengine:

#################
neoml.MathEngine
#################

The purpose of the MathEngine class is to isolate the algorithms library from the implementation of the low-level platform-dependent operations. It is used in :ref:`blob <py-dnn-blob>`, :ref:`layers <py-dnn-layers>`, and :ref:`neural network<py-dnn-network>`.

.. autoclass:: neoml.MathEngine.MathEngine
   :members:

.. autoclass:: neoml.MathEngine.CpuMathEngine
   :members:

.. autoclass:: neoml.MathEngine.GpuMathEngine
   :members:

.. automethod:: neoml.MathEngine.enum_gpu

.. automethod:: neoml.MathEngine.default_math_engine