# Conditional Random Field (CRF)

<!-- TOC -->

- [Conditional Random Field (CRF)](#conditional-random-field-crf)
    - [Implementation](#implementation)
    - [Useful links](#useful-links)

<!-- /TOC -->

## Implementation

Conditional random field (CRF) implementation consists of three parts:

1. [The trainable layer](CrfLayer.md), which contains the transition probabilities of the random field.
2. [The loss function](CrfLossLayer.md) that should be optimized by training.
3. [The special layer](BestSequenceLayer.md) that provides the highest-probability sequences from the output of the trained CRF layer.

## Useful links

- [Presentation (slide 22)](http://www.phontron.com/slides/sdm-20131114.pdf)
- [Video](https://www.youtube.com/watch?v=fGdXkVv1qNQ)
- [A sample in another library](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
