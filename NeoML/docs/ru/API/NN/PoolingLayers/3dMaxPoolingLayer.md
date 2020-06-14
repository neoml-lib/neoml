# Класс C3dMaxPoolingLayer

<!-- TOC -->

- [Класс C3dMaxPoolingLayer](#класс-c3dmaxpoolinglayer)
    - [Настройки](#настройки)
        - [Размеры фильтра](#размеры-фильтра)
        - [Шаг фильтра](#шаг-фильтра)
    - [Обучаемые параметры](#обучаемые-параметры)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, выполняющий `Max Pooling` над набором трехмерных многоканальных изображений.

## Настройки

### Размеры фильтра

```c++
void SetFilterHeight( int filterHeight );
void SetFilterWidth( int filterWidth );
void SetFilterDepth( int filterDepth );
```

### Шаг фильтра

```c++
void SetStrideHeight( int strideHeight );
void SetStrideWidth( int strideWidth );
void SetStrideDepth( int strideDepth );
```

По умолчанию равны `1`.

## Обучаемые параметры

Слой не имеет обучаемых параметров.

## Входы

На единственный вход подается блоб с набором изображений:

- `BatchLength * BatchWidth * ListSize` - количество изображений в наборе;
- `Height` - высота изображений;
- `Width` - ширина изображений;
- `Depth` - глубина изображений;
- `Channels` - количество каналов у изображений.

## Выходы

Единственный выход содержит блоб размера:

- `BatchLength` равный `BatchLength` входа;
- `BatchWidth` равный `BatchWidth` входа;
- `ListSize` равный `ListSize` входа;
- `Height` рассчитывается относительно входа по формуле  
`(Height - FilterHeight)/StrideHeight + 1`;
- `Width` рассчитывается относительно входа по формуле  
`(Width - FilterWidth)/StrideWidth + 1`;
- `Depth` рассчитывается относительно входа по формуле  
`(Depth - FilterDepth)/StrideDepth + 1`;
- `Channels` равен `Channels` входа.
