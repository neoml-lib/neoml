# Класс CMeanPoolingLayer

<!-- TOC -->

- [Класс CMeanPoolingLayer](#класс-cmeanpoolinglayer)
    - [Настройки](#настройки)
        - [Размеры фильтра](#размеры-фильтра)
        - [Шаг фильтра](#шаг-фильтра)
    - [Обучаемые параметры](#обучаемые-параметры)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, выполняющий `Mean Pooling` над набором двумерных многоканальных изображений.

## Настройки

### Размеры фильтра

```c++
void SetFilterHeight( int filterHeight );
void SetFilterWidth( int filterWidth );
```

### Шаг фильтра

```c++
void SetStrideHeight( int strideHeight );
void SetStrideWidth( int strideWidth );
```

По умолчанию равны `1`.

## Обучаемые параметры

Слой не имеет обучаемых параметров.

## Входы

На единственный вход подается блоб с набором изображений:

- `BatchLength * BatchWidth * ListSize` - количество изображений в наборе;
- `Height` - высота изображений;
- `Width` - ширина изображений;
- `Depth * Channels` - количество каналов у изображений.

## Выходы

Единственный выход содержит блоб размера:

- `BatchLength` равный `BatchLength` входа;
- `BatchWidth` равный `BatchWidth` входа;
- `ListSize` равный `ListSize` входа;
- `Height` рассчитывается относительно входа по формуле  
`(Height - FilterHeight)/StrideHeight + 1`;
- `Width` рассчитывается относительно входа по формуле  
`(Width - FilterWidth)/StrideWidth + 1`;
- `Depth` равен `Depth` входа;
- `Channels` равен `Channels` входа.
