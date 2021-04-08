.. _py-submodule-dnn:

#########
neoml.Dnn
#########

- :ref:`py-dnn-network`
- :ref:`py-dnn-layers`

   - :ref:`py-dnn-recurrent`
   - :ref:`py-dnn-loss`
   - :ref:`py-dnn-activation`
   - :ref:`py-dnn-pooling`
   - :ref:`py-dnn-inputoutput`
   - :ref:`py-dnn-qualitycontrol`
   - :ref:`py-dnn-conv`
   - :ref:`py-dnn-binarization`
   - :ref:`py-dnn-attention`
   - :ref:`py-dnn-normalization`
   - :ref:`py-dnn-concat`
   - :ref:`py-dnn-split`
   - :ref:`py-dnn-sequences`
   - :ref:`py-dnn-eltwise`
   - :ref:`py-dnn-imageconversion`
   - :ref:`py-dnn-crf`
   - :ref:`py-dnn-ctc`
- :ref:`py-dnn-initializers`
- :ref:`py-dnn-solver`

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

.. _py-dnn-layers:

Layers
######

A layer is an element of the network that performs some operation: anything from the input data reshape or a simple math function calculation, up to convolution or LSTM (Long short-term memory).

If the operation needs input data, it will be taken from the layer input. Each layer input contains one data blob, and if several blobs are needed, the layer will have several inputs. Each layer input should be connected to another layer's output.

If the operation returns results that should be used by other layers, they will be passed to the layer outputs. Each layer output contains one data blob, so depending on the operation it performs the layer may have several outputs. Several other layer inputs may be connected to the same output, but you may not leave an output unconnected to any inputs.

In addition, the layer may have settings specified by the user before starting calculations, and trainable parameters that are optimized during network training.

The layers also have names that can be used to find a layer in the network. The name should be set at layer creation or before adding it to the network.

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

.. _py-dnn-binarization:

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

.. _py-dnn-concat:

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

.. _py-dnn-split:

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

.. _py-dnn-sequences:

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

.. _py-dnn-eltwise:

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

.. _py-dnn-imageconversion:

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

.. _py-dnn-crf:

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

.. _py-dnn-ctc:

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

AccumulativeLookup
************************

.. autoclass:: neoml.Dnn.AccumulativeLookup
   :members:

Argmax
************

.. autoclass:: neoml.Dnn.Argmax
   :members:

DotProduct
****************

.. autoclass:: neoml.Dnn.DotProduct 
   :members:

Dropout
*************

.. autoclass:: neoml.Dnn.Dropout 
   :members:


FullyConnected
*********************

.. autoclass:: neoml.Dnn.FullyConnected
   :members:

MatrixMultiplication
**************************

.. autoclass:: neoml.Dnn.MatrixMultiplication
   :members:

MultichannelLookup
**************************

.. autoclass:: neoml.Dnn.MultichannelLookup
   :members:

PositionalEmbedding
*************************

.. autoclass:: neoml.Dnn.PositionalEmbedding
   :members:

Reorg
*************************

.. autoclass:: neoml.Dnn.Reorg
   :members:

Softmax
************

.. autoclass:: neoml.Dnn.Softmax
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

.. _py-dnn-initializers

Initializers
###############

Before the first training iteration the layers' weights (trainable parameters) must be initialized. The initializer is the same for all the network trainable weights, except for the free term vectors that are initialized with zeros.

Xavier
***********

The default initializer. It generates the weights using the normal distribution ``N(0, 1/n)``, where `n` is the input size.

.. autoclass:: neoml.Dnn.Xavier
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