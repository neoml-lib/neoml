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

Binarization layers
*******************

.. autoclass:: neoml.Dnn.EnumBinarization
   :members:
.. autoclass:: neoml.Dnn.BitSetVectorization
   :members:

AccumulativeLookup
************************

.. autoclass:: neoml.Dnn.AccumulativeLookup
   :members:

Accuracy layers
***************

Accuracy
========

.. autoclass:: neoml.Dnn.Accuracy
   :members:

ConfusionMatrix
================

.. autoclass:: neoml.Dnn.ConfusionMatrix
   :members:

Argmax
************

.. autoclass:: neoml.Dnn.Argmax
   :members:

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

Concatenation layers
************************

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

Convolutional layers
********************

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

Crf layers
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

Ctc layers
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

Image Conversion layers
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

PrecisionRecall
*************************

.. autoclass:: neoml.Dnn.PrecisionRecall
   :members:

Reorg
*************************

.. autoclass:: neoml.Dnn.Reorg
   :members:

SequenceSum
********************

.. autoclass:: neoml.Dnn.SequenceSum
   :members:

Sink
*********

.. autoclass:: neoml.Dnn.Sink
   :members:

Softmax
************

.. autoclass:: neoml.Dnn.Softmax
   :members:

Source
********

.. autoclass:: neoml.Dnn.Source
   :members:

Split layers
********************

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

SubSequence layers
******************

SubSequence
===============

.. autoclass:: neoml.Dnn.SubSequence
   :members:

ReverseSequence
===============

.. autoclass:: neoml.Dnn.ReverseSequence
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

Initializers
*************

Xavier
======

.. autoclass:: neoml.Dnn.Xavier
   :members:

Uniform
=======

.. autoclass:: neoml.Dnn.Uniform
   :members:

Solvers
********

SimpleGradient
===============

.. autoclass:: neoml.Dnn.SimpleGradient
   :members:

AdaptiveGradient
=================

.. autoclass:: neoml.Dnn.AdaptiveGradient
   :members:

NesterovGradient
================

.. autoclass:: neoml.Dnn.NesterovGradient
   :members: