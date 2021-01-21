# CTC — Connectionist Temporal Classification

<!-- TOC -->

- [CTC — Connectionist Temporal Classification](#ctc-—-connectionist-temporal-classification)
    - [Implementation](#implementation)
    - [Useful links](#useful-links)

<!-- /TOC -->

## Implementation

A connectionist temporal classification (CTC) network is trained to optimize the [loss function](CtcLossLayer.md).

After training, use the special [decoding layer](CtcDecodingLayer.md) to extract the optimal sequences from the network output.

## Useful links

- [Supervised Sequence Labelling with Recurrent
Neural Networks (Ch. 7)](https://www.cs.toronto.edu/~graves/preprint.pdf)
