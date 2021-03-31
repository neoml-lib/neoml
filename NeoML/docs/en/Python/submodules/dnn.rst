.. _py-submodule-dnn:

#########
neoml.Dnn
#########

- :ref:`py-dnn-network`
- :ref:`py-dnn-layers`
   - :ref:`py-dnn-recurrent`
   - :ref:`py-dnn-activation`
   - :ref:`py-dnn-binarization`
   - :ref:`py-dnn-qualitycontrol`
   - :ref:`py-dnn-attention`
   - :ref:`py-dnn-normalization`
   - :ref:`py-dnn-concat`
   - :ref:`py-dnn-convolution`
   - :ref:`py-dnn-crf`
   - :ref:`py-dnn-ctc`
   - :ref:`py-dnn-eltwise`
   - :ref:`py-dnn-imageconversion`
   - :ref:`py-dnn-loss`
   - :ref:`py-dnn-pooling`
   - :ref:`py-dnn-inputoutput`
   - :ref:`py-dnn-split`
   - :ref:`py-dnn-sequences`
- :ref:`py-dnn-initializer`
- :ref:`py-dnn-solver`

.. _py-dnn-network:

Neural network
##############

.. autoclass:: neoml.Dnn.Dnn
   :members:

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

Irnn
====

.. autoclass:: neoml.Dnn.Irnn
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

.._py-dnn-binarization:

Binarization layers
*******************

EnumBinarization
=================
.. autoclass:: neoml.Dnn.EnumBinarization
   :members:

BitSetVectorization
====================
.. autoclass:: neoml.Dnn.BitSetVectorization
   :members:

AccumulativeLookup
************************

.. autoclass:: neoml.Dnn.AccumulativeLookup
   :members:

.._py-dnn-qualitycontrol:

Quality control layers
**********************

Accuracy
========

.. autoclass:: neoml.Dnn.Accuracy
   :members:

PrecisionRecall
========================

.. autoclass:: neoml.Dnn.PrecisionRecall
   :members:

ConfusionMatrix
================

.. autoclass:: neoml.Dnn.ConfusionMatrix
   :members:

Argmax
************

.. autoclass:: neoml.Dnn.Argmax
   :members:

.._py-dnn-attention:

Attention layers
****************

AttentionDecoder
======================

.. autoclass:: neoml.Dnn.AttentionDecoder 
   :members:

MultiheadAttention
========================

.. autoclass:: neoml.Dnn.MultiheadAttention
   :members:

.._py-dnn-normalization:

Normalization layers
********************

BatchNormalization
==================

.. autoclass:: neoml.Dnn.BatchNormalization 
   :members:

ObjectNormalization
===================

.. autoclass:: neoml.Dnn.ObjectNormalization
   :members:

.._py-dnn-concat:

Concatenation layers
************************
The layers that concatenate blobs along one of the dimensions.

ConcatChannels
==============

.. autoclass:: neoml.Dnn.ConcatChannels 
   :members:

ConcatDepth
===========

.. autoclass:: neoml.Dnn.ConcatDepth 
   :members:

ConcatWidth
===========

.. autoclass:: neoml.Dnn.ConcatWidth 
   :members:

ConcatHeight
============

.. autoclass:: neoml.Dnn.ConcatHeight 
   :members:

ConcatBatchWidth
================

.. autoclass:: neoml.Dnn.ConcatBatchWidth
   :members:

ConcatObject
============

.. autoclass:: neoml.Dnn.ConcatObject
   :members:

.._py-dnn-conv:

Convolutional layers
********************
The layers that perform various types of convolution.

Conv
====

.. autoclass:: neoml.Dnn.Conv
   :members:

Conv3D
======

.. autoclass:: neoml.Dnn.Conv3D
   :members:

TransposedConv3D
================

.. autoclass:: neoml.Dnn.TransposedConv3D
   :members:

TransposedConv
==============

.. autoclass:: neoml.Dnn.TransposedConv
   :members:

ChannelwiseConv
===============

.. autoclass:: neoml.Dnn.ChannelwiseConv
   :members:

TimeConv
========

.. autoclass:: neoml.Dnn.TimeConv
   :members:

.._py-dnn-crf:

CRF layers
**********


Crf
===

.. autoclass:: neoml.Dnn.Crf
   :members:

CrfLoss
=======

.. autoclass:: neoml.Dnn.CrfLoss
   :members:

BestSequence
============

.. autoclass:: neoml.Dnn.BestSequence
   :members:

.._py-dnn-ctc:

CTC layers
**********

CtcLoss
=======

.. autoclass:: neoml.Dnn.CtcLoss
   :members:

CtcDecoding
===========

.. autoclass:: neoml.Dnn.CtcDecoding
   :members:

DotProduct
****************

.. autoclass:: neoml.Dnn.DotProduct 
   :members:

Dropout
*************

.. autoclass:: neoml.Dnn.Dropout 
   :members:

.._py-dnn-eltwise:

Elementwise operation layers
****************************

EltwiseSum
==========

.. autoclass:: neoml.Dnn.EltwiseSum
   :members:

EltwiseMul
==========

.. autoclass:: neoml.Dnn.EltwiseMul
   :members:

EltwiseNegMul
=============

.. autoclass:: neoml.Dnn.EltwiseNegMul
   :members:

EltwiseMax
==========

.. autoclass:: neoml.Dnn.EltwiseMax
   :members:

FullyConnected
*********************

.. autoclass:: neoml.Dnn.FullyConnected
   :members:

.._py-dnn-imageconversion:

Image conversion layers
***********************

ImageResize
===========

.. autoclass:: neoml.Dnn.ImageResize
   :members:

PixelToImage
============

.. autoclass:: neoml.Dnn.PixelToImage
   :members:

ImageToPixel
============

.. autoclass:: neoml.Dnn.ImageToPixel
   :members:

.._py-dnn-loss:

Loss layers
***********

CrossEntropyLoss
================

.. autoclass:: neoml.Dnn.CrossEntropyLoss
   :members:

BinaryCrossEntropyLoss
======================

.. autoclass:: neoml.Dnn.BinaryCrossEntropyLoss
   :members:

EuclideanLoss
======================

.. autoclass:: neoml.Dnn.EuclideanLoss
   :members:

HingeLoss
==========

.. autoclass:: neoml.Dnn.HingeLoss
   :members:

SquaredHingeLoss
================

.. autoclass:: neoml.Dnn.SquaredHingeLoss
   :members:

FocalLoss
================

.. autoclass:: neoml.Dnn.FocalLoss
   :members:

BinaryFocalLoss
================

.. autoclass:: neoml.Dnn.BinaryFocalLoss
   :members:

CenterLoss
================

.. autoclass:: neoml.Dnn.CenterLoss
   :members:

MultiHingeLoss
================

.. autoclass:: neoml.Dnn.MultiHingeLoss
   :members:

MultiSquaredHingeLoss
=====================

.. autoclass:: neoml.Dnn.MultiSquaredHingeLoss
   :members:

MatrixMultiplication
**************************

.. autoclass:: neoml.Dnn.MatrixMultiplication
   :members:

MultichannelLookup
**************************

.. autoclass:: neoml.Dnn.MultichannelLookup
   :members:

.._py-dnn-pooling:

Pooling layers
**************

MaxPooling
==========

.. autoclass:: neoml.Dnn.MaxPooling
   :members:

MeanPooling
===========

.. autoclass:: neoml.Dnn.MeanPooling
   :members:

GlobalMaxPooling
================

.. autoclass:: neoml.Dnn.GlobalMaxPooling
   :members:

GlobalMeanPooling
================

.. autoclass:: neoml.Dnn.GlobalMeanPooling
   :members:

MaxOverTimePooling
==================

.. autoclass:: neoml.Dnn.MaxOverTimePooling
   :members:

ProjectionPooling
=================

.. autoclass:: neoml.Dnn.ProjectionPooling
   :members:

MaxPooling3D
=================

.. autoclass:: neoml.Dnn.MaxPooling3D
   :members:

MeanPooling3D
=================

.. autoclass:: neoml.Dnn.MeanPooling3D
   :members:

PositionalEmbedding
*************************

.. autoclass:: neoml.Dnn.PositionalEmbedding
   :members:

Reorg
*************************

.. autoclass:: neoml.Dnn.Reorg
   :members:

.._py-dnn-inputoutput:

Passing the data to and from the network
**********************************************

Source
==============

.. autoclass:: neoml.Dnn.Source
   :members:

Sink
=============

.. autoclass:: neoml.Dnn.Sink
   :members:

Softmax
************

.. autoclass:: neoml.Dnn.Softmax
   :members:

.._py-dnn-split:

Split layers
********************
The layers that split a blob along one of the dimensions.

SplitChannels
=============

.. autoclass:: neoml.Dnn.SplitChannels
   :members:

SplitDepth
=============

.. autoclass:: neoml.Dnn.SplitDepth
   :members:

SplitWidth
=============

.. autoclass:: neoml.Dnn.SplitWidth
   :members:

SplitHeight
=============

.. autoclass:: neoml.Dnn.SplitHeight
   :members:

SplitBatchWidth
===============

.. autoclass:: neoml.Dnn.SplitBatchWidth
   :members:

.._py-dnn-sequences:

Working with sequences
*************************

SubSequence
===============

.. autoclass:: neoml.Dnn.SubSequence
   :members:

ReverseSequence
===============

.. autoclass:: neoml.Dnn.ReverseSequence
   :members:

SequenceSum
===================

.. autoclass:: neoml.Dnn.SequenceSum
   :members:

TiedEmbeddings
********************

.. autoclass:: neoml.Dnn.TiedEmbeddings
   :members:

Transform
****************

.. autoclass:: neoml.Dnn.Transform
   :members:

Transpose
****************

.. autoclass:: neoml.Dnn.Transpose
   :members:

Upsampling2D
************

.. autoclass:: neoml.Dnn.Upsampling2D
   :members:

.._py-dnn-initializer

Initializers
###############

Xavier
***********

.. autoclass:: neoml.Dnn.Xavier
   :members:

Uniform
*************

.. autoclass:: neoml.Dnn.Uniform
   :members:

.._py-dnn-solver:

Solvers
###########

SimpleGradient
******************

.. autoclass:: neoml.Dnn.SimpleGradient
   :members:

AdaptiveGradient
********************

.. autoclass:: neoml.Dnn.AdaptiveGradient
   :members:

NesterovGradient
******************

.. autoclass:: neoml.Dnn.NesterovGradient
   :members: