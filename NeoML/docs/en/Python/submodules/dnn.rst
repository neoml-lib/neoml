.. _py-submodule-dnn:

#########
neoml.Dnn
#########

- :ref:`py-dnn-network`
- :ref:`py-dnn-blob`
- :ref:`py-dnn-layers`

   - :ref:`py-dnn-baselayer`
   - :ref:`py-dnn-inputoutput`
   - :ref:`py-dnn-recurrent`
   - :ref:`py-dnn-fullyconnected`
   - :ref:`py-dnn-activation`
   - :ref:`py-dnn-conv`
   - :ref:`py-dnn-loss`
   - :ref:`py-dnn-pooling`
   - :ref:`py-dnn-softmax`
   - :ref:`py-dnn-dropout`
   - :ref:`py-dnn-cumsum`
   - :ref:`py-dnn-normalization`
   - :ref:`py-dnn-eltwise`
   - :ref:`py-dnn-qualitycontrol`
   - :ref:`py-dnn-discrete-features`
   - :ref:`py-dnn-attention`
   - :ref:`py-dnn-logical`
      - :ref:`py-dnn-not`
      - :ref:`py-dnn-less`
      - :ref:`py-dnn-equal`
      - :ref:`py-dnn-where`
   - :ref:`py-dnn-auxiliary`:
      - :ref:`py-dnn-transform`
      - :ref:`py-dnn-transpose`
      - :ref:`py-dnn-argmax`
      - :ref:`py-dnn-dotproduct`
      - :ref:`py-dnn-matrixmult`
      - :ref:`py-dnn-reorg`
      - :ref:`py-dnn-concat`
      - :ref:`py-dnn-split`
      - :ref:`py-dnn-sequences`
      - :ref:`py-dnn-imageconversion`
      - :ref:`py-dnn-scattergather`
   - :ref:`py-dnn-crf`
   - :ref:`py-dnn-ctc`

- :ref:`py-dnn-initializers`
- :ref:`py-dnn-solver`
- :ref:`py-dnn-random`
- :ref:`py-dnn-autodiff`

.. _py-dnn-network:

Neural network
##############

A neural network is a directed graph with the vertices corresponding to layers and the arcs corresponding to the connections along which the data is passed from one layer's output to another's input.

Each layer should be added to the network after you assign a unique name to it. A layer may not be connected to several networks at once.

Source layers are used to pass the data into the network. A source layer has no inputs and passes the data blob specified by the user to its only output.

Sink layers with no outputs are used to retrieve the result of the network operation. They provide a function that returns the blob with data.

After all the layers are added and connected the network may be set up for training.

.. autoclass:: neoml.Dnn.Dnn
   :members:

.. _py-dnn-blob:

Data blobs
##########

All data used in the network operation (inputs, outputs, trainable parameters) is stored in blobs. A *blob* is a 7-dimensional array, and each of its dimensions has a specific meaning:

- **BatchLength** is a "time" axis, used to denote data sequences; it is mainly used in recurrent networks
- **BatchWidth** corresponds to the batch, used to pass several independent objects together
- **ListSize** is the dimensions for the objects that are connected (for example, pixels out of one image) but do not form a sequence
- **Height** is the height of a matrix or an image
- **Width** is the width of a matrix or an image
- **Depth** is the width of a 3-dimensional image
- **Channels** corresponds to channels for multi-channel image formats and is also used to work with one-dimensional vectors.

The blobs may contain one of the two types of data: ``float`` and ``int``. Both data types are 32-bit.

If the data type is not specified directly anywhere in this documentation, that means ``float`` is used.

Class description
*******************

.. autoclass:: neoml.Blob.Blob
   :members:

Working with blobs
*******************

.. automethod:: neoml.Blob.store

.. automethod:: neoml.Blob.load

.. automethod:: neoml.Blob.asblob

Creating blobs of typical size
******************************

The auxiliary methods that create blobs for frequently used types of data.

.. automethod:: neoml.Blob.vector

.. automethod:: neoml.Blob.matrix

.. automethod:: neoml.Blob.tensor

.. automethod:: neoml.Blob.list_blob

.. automethod:: neoml.Blob.image2d

.. automethod:: neoml.Blob.image3d

.. _py-dnn-layers:

Layers
######

A layer is an element of the network that performs some operation: anything from the input data reshape or a simple math function calculation, up to convolution or LSTM (Long short-term memory).

If the operation needs input data, it will be taken from the layer input. Each layer input contains one data blob, and if several blobs are needed, the layer will have several inputs. Each layer input should be connected to another layer's output.

If the operation returns results that should be used by other layers, they will be passed to the layer outputs. Each layer output contains one data blob, so depending on the operation it performs the layer may have several outputs. Several other layer inputs may be connected to the same output, but you may not leave an output unconnected to any inputs.

In addition, the layer may have settings specified by the user before starting calculations, and trainable parameters that are optimized during network training.

The layers also have names that can be used to find a layer in the network. The name should be set at layer creation or before adding it to the network.

.. _py-dnn-baselayer:

Base layer class
*****************

All NeoML layer classes are derived from this class.

.. autoclass:: neoml.Dnn.Layer
   :members:

.. _py-dnn-inputoutput:

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

.. _py-dnn-recurrent:

Recurrent layers
****************

Lstm
====

The long short-term memory layer.

.. autoclass:: neoml.Dnn.Lstm
   :members:

Qrnn
====

.. autoclass:: neoml.Dnn.Qrnn
   :members:

Gru
===

The gated recurrent unit.

.. autoclass:: neoml.Dnn.Gru
   :members:

Irnn
====

.. autoclass:: neoml.Dnn.Irnn
   :members:

IndRnn
======

.. autoclass:: neoml.Dnn.IndRnn
   :members:

.. _py-dnn-fullyconnected:

FullyConnected
*********************

.. autoclass:: neoml.Dnn.FullyConnected
   :members:

.. _py-dnn-activation:

Activation layers
*****************

These layers calculate the value of different activation functions on their inputs.

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

Exp
===

.. autoclass:: neoml.Dnn.Exp
   :members:

Log
===

.. autoclass:: neoml.Dnn.Log
   :members:

Erf
===

.. autoclass:: neoml.Dnn.Erf
   :members:

.. _py-dnn-conv:

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

.. _py-dnn-loss:

Loss layers
***********

These layers calculate different types of loss functions on the network response. They have no output and return the loss value through a `last_loss` method.

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

CustomLoss
==========

NeoML provides an interface for user-implemented custom loss functions. They must be constructed out of simple arithmetic and :ref:`py-dnn-autodiff` functions.

.. autoclass:: neoml.Dnn.CustomLossCalculatorBase
   :members:

.. autoclass:: neoml.Dnn.CustomLoss
   :members:

.. _py-dnn-pooling:

Pooling layers
**************

The layers that perform pooling operations on the input data.

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

GlobalSumPooling
================

.. autoclass:: neoml.Dnn.GlobalSumPooling
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

.. _py-dnn-softmax:

Softmax
************

.. autoclass:: neoml.Dnn.Softmax
   :members:

.. _py-dnn-dropout:

Dropout
*************

.. autoclass:: neoml.Dnn.Dropout 
   :members:

.. _py-dnn-cumsum:

CumSum
===========

.. autoclass:: neoml.Dnn.CumSum
   :members:

.. _py-dnn-normalization:

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

.. _py-dnn-eltwise:

Lrn (Local Response Normalization)
==================================

.. autoclass:: neoml.Dnn.Lrn
   :members:

Elementwise operation layers
****************************

EltwiseSum
==========

.. autoclass:: neoml.Dnn.EltwiseSum
   :members:

EltwiseSub
==========

.. autoclass:: neoml.Dnn.EltwiseSub
   :members:

EltwiseMul
==========

.. autoclass:: neoml.Dnn.EltwiseMul
   :members:

EltwiseDiv
==========

.. autoclass:: neoml.Dnn.EltwiseDiv
   :members:

EltwiseNegMul
=============

.. autoclass:: neoml.Dnn.EltwiseNegMul
   :members:

EltwiseMax
==========

.. autoclass:: neoml.Dnn.EltwiseMax
   :members:

.. _py-dnn-qualitycontrol:

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

.. _py-dnn-discrete-features:

Working with discrete features
*******************************

AccumulativeLookup
========================

.. autoclass:: neoml.Dnn.AccumulativeLookup
   :members:

MultichannelLookup
==========================

.. autoclass:: neoml.Dnn.MultichannelLookup
   :members:

PositionalEmbedding
==========================

.. autoclass:: neoml.Dnn.PositionalEmbedding
   :members:

TiedEmbeddings
==================

.. autoclass:: neoml.Dnn.TiedEmbeddings
   :members:

EnumBinarization
=================
.. autoclass:: neoml.Dnn.EnumBinarization
   :members:

BitSetVectorization
====================
.. autoclass:: neoml.Dnn.BitSetVectorization
   :members:

.. _py-dnn-attention:

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

.. _py-dnn-logical:

Logical operations Layers
*************************

.. _py-dnn-not:

Not
===

.. autoclass:: neoml.Dnn.Not
   :members:

.. _py-dnn-less:

Less
====

.. autoclass:: neoml.Dnn.Less
   :members:

.. _py-dnn-equal:

Equal
=====

.. autoclass:: neoml.Dnn.Equal
   :members:

.. _py-dnn-where:

Where
=====

.. autoclass:: neoml.Dnn.Where
   :members:

.. _py-dnn-auxiliary:

Auxiliary operations
************************

.. _py-dnn-transform:

Transform
================

.. autoclass:: neoml.Dnn.Transform
   :members:

.. _py-dnn-transpose:

Transpose
============

.. autoclass:: neoml.Dnn.Transpose
   :members:

.. _py-dnn-argmax:

Argmax
===========

.. autoclass:: neoml.Dnn.Argmax
   :members:

.. _py-dnn-dotproduct:

DotProduct
==============

.. autoclass:: neoml.Dnn.DotProduct 
   :members:

.. _py-dnn-matrixmult:

MatrixMultiplication
========================

.. autoclass:: neoml.Dnn.MatrixMultiplication
   :members:

.. _py-dnn-reorg:

Reorg
========

.. autoclass:: neoml.Dnn.Reorg
   :members:

.. _py-dnn-concat:

Concatenation layers
============================
The layers that concatenate blobs along one of the dimensions.

ConcatChannels
----------------------

.. autoclass:: neoml.Dnn.ConcatChannels 
   :members:

ConcatDepth
--------------

.. autoclass:: neoml.Dnn.ConcatDepth 
   :members:

ConcatWidth
--------------

.. autoclass:: neoml.Dnn.ConcatWidth 
   :members:

ConcatHeight
----------------

.. autoclass:: neoml.Dnn.ConcatHeight 
   :members:

ConcatBatchWidth
--------------------

.. autoclass:: neoml.Dnn.ConcatBatchWidth
   :members:

ConcatBatchLength
--------------------

.. autoclass:: neoml.Dnn.ConcatBatchLength
   :members:

ConcatListSize
--------------------

.. autoclass:: neoml.Dnn.ConcatListSize
   :members:

ConcatObject
----------------

.. autoclass:: neoml.Dnn.ConcatObject
   :members:

.. _py-dnn-split:

Split layers
===================
The layers that split a blob along one of the dimensions.

SplitChannels
-----------------

.. autoclass:: neoml.Dnn.SplitChannels
   :members:

SplitDepth
---------------

.. autoclass:: neoml.Dnn.SplitDepth
   :members:

SplitWidth
--------------

.. autoclass:: neoml.Dnn.SplitWidth
   :members:

SplitHeight
---------------

.. autoclass:: neoml.Dnn.SplitHeight
   :members:

SplitListSize
-----------------

.. autoclass:: neoml.Dnn.SplitListSize
   :members:

SplitBatchWidth
-----------------

.. autoclass:: neoml.Dnn.SplitBatchWidth
   :members:

SplitBatchLength
-----------------

.. autoclass:: neoml.Dnn.SplitBatchLength
   :members:

.. _py-dnn-sequences:

Working with sequences
=============================

SubSequence
----------------

.. autoclass:: neoml.Dnn.SubSequence
   :members:

ReverseSequence
---------------------

.. autoclass:: neoml.Dnn.ReverseSequence
   :members:

SequenceSum
---------------

.. autoclass:: neoml.Dnn.SequenceSum
   :members:

.. _py-dnn-imageconversion:

Image conversion layers
=========================

ImageResize
--------------

.. autoclass:: neoml.Dnn.ImageResize
   :members:

PixelToImage
---------------

.. autoclass:: neoml.Dnn.PixelToImage
   :members:

ImageToPixel
---------------

.. autoclass:: neoml.Dnn.ImageToPixel
   :members:

Upsampling2D
-----------------

.. autoclass:: neoml.Dnn.Upsampling2D
   :members:

.. _py-dnn-scattergather:

Scatter & Gather Layers
=======================

ScatterND
---------

.. autoclass:: neoml.Dnn.ScatterND
   :members:

.. _py-dnn-crf:

Conditional random field (CRF)
********************************
Conditional random field is implemented in three parts: 

- the trainable layer that contains the transition probabilities of the random field
- the loss function optimized by training
- the special layer to extract highest-probability sequences from the output of the trained CRF layer

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

.. _py-dnn-ctc:

Connectionist temporal classification (CTC)
*************************************************

A connectionist temporal classification (CTC) network is trained to optimize the loss function.

After training, use the special decoding layer to extract the optimal sequences from the network output.

See also https://www.cs.toronto.edu/~graves/preprint.pdf

CtcLoss
=======

.. autoclass:: neoml.Dnn.CtcLoss
   :members:

CtcDecoding
===========

.. autoclass:: neoml.Dnn.CtcDecoding
   :members:

.. _py-dnn-initializers:

Initializers
###############

Before the first training iteration the layers' weights (trainable parameters) must be initialized. The initializer is the same for all the network trainable weights, except for the free term vectors that are initialized with zeros.

Xavier
***********

The default initializer. It generates the weights using the normal distribution ``N(0, 1/n)``, where `n` is the input size.

.. autoclass:: neoml.Dnn.Xavier
   :members:

XavierUniform
******************

The default initializer. It generates the weights using the uniform distribution ``U(-sqrt(1/n), sqrt(1/n))``, where `n` is the input size.

.. autoclass:: neoml.Dnn.XavierUniform
   :members:

Uniform
*************

Generates the weights using a uniform distribution over the specified segment.

.. autoclass:: neoml.Dnn.Uniform
   :members:

.. _py-dnn-solver:

Solvers
###########

The optimizer, or solver, sets the rules to update the weights during training.

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

.. _py-dnn-random:

Random
########

.. autoclass:: neoml.Random.Random
   :members:

.. _py-dnn-autodiff:

Autodifferentiation
####################

NeoML supports autodifferentiation for a wide set of operations. Use these operations and simple arithmetic if you'd like to create your own loss function `neoml.Dnn.CustomLoss`. Then during the backward pass, NeoML will be able to calculate gradients of your custom loss.

.. automethod:: neoml.AutoDiff.const

Simple arithmetic operations
******************************

.. automethod:: neoml.AutoDiff.add
.. automethod:: neoml.AutoDiff.sub
.. automethod:: neoml.AutoDiff.mul
.. automethod:: neoml.AutoDiff.div
.. automethod:: neoml.AutoDiff.less

Basic math functions
*********************

.. automethod:: neoml.AutoDiff.max
.. automethod:: neoml.AutoDiff.sum
.. automethod:: neoml.AutoDiff.cumsum
.. automethod:: neoml.AutoDiff.mean
.. automethod:: neoml.AutoDiff.neg
.. automethod:: neoml.AutoDiff.abs
.. automethod:: neoml.AutoDiff.log
.. automethod:: neoml.AutoDiff.exp
.. automethod:: neoml.AutoDiff.pow

Other operations
*******************

.. automethod:: neoml.AutoDiff.concat
.. automethod:: neoml.AutoDiff.reshape
.. automethod:: neoml.AutoDiff.clip
.. automethod:: neoml.AutoDiff.top_k
.. automethod:: neoml.AutoDiff.binary_cross_entropy