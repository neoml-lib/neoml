# CTC - Connectionist Temporal Classification

<!-- TOC -->

- [CTC - Connectionist Temporal Classification](#ctc---connectionist-temporal-classification)
    - [Реализация](#реализация)
    - [Полезные ссылки](#полезные-ссылки)

<!-- /TOC -->

## Реализация

Задача CTC решается при помощи [функции потерь](CtcLossLayer.md), оптимизируемой во время обучения.

После обучения для извлечения оптимальных последовательностей из результата работы сети, используется [специальный слой-декодер](CtcDecodingLayer.md).

## Полезные ссылки

- [Supervised Sequence Labelling with Recurrent
Neural Networks (Ch. 7)](https://www.cs.toronto.edu/~graves/preprint.pdf)
