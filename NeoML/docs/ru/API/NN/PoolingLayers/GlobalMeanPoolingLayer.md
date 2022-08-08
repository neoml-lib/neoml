# Класс CGlobalMeanPoolingLayer

<!-- TOC -->

- [Класс CGlobalMeanPoolingLayer](#класс-cglobalmeanpoolinglayer)
    - [Обучаемые параметры](#обучаемые-параметры)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, выполняющий операцию `Mean Pooling` над размерностями `Height`, `Width`, `Depth`.

## Обучаемые параметры

Слой не имеет обучаемых параметров.

## Входы

На единственный вход подается [блоб](../DnnBlob.md) с набором изображений:

- `BatchLength * BatchWidth * ListSize` - количество изображений в наборе;
- `Height` - высота изображений;
- `Width` - ширина изображений;
- `Depth` - глубина изображений;
- `Channels` - количество каналов у изображений.

## Выходы

Единственный выход содержит блоб размера:

- `BatchLength` равен `BatchLength` входа;
- `BatchWidth` равен `BatchWidth` входа;
- `ListSize` равен `ListSize` входа;
- `Height` равен `1`;
- `Width` равен `1`;
- `Depth` равен `1`;
- `Channels` равен `Channels` входа.
