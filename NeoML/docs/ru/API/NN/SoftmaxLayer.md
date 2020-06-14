# Класс CSoftmaxLayer

<!-- TOC -->

- [Класс CSoftmaxLayer](#класс-csoftmaxlayer)
    - [Настройки](#настройки)
        - [Область применения](#область-применения)
    - [Обучаемые параметры](#обучаемые-параметры)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, вычисляющий функцию `softmax` над набором векторов.

Формула, применяемая к каждому вектору:

```c++
softmax(x[0], ... , x[n-1])[i] = exp(x[i]) / (exp(x[0]) + ... + exp(x[n-1]))
```

## Настройки

### Область применения

```c++
// Область, над которой будет происходить нормирование.
enum TNormalizationArea {
    NA_ObjectSize = 0,
    NA_BatchLength,
    NA_ListSize,
    NA_Channel,

    NA_Count
};

void SetNormalizationArea( TNormalizationArea newArea )
```

Настраивает то, какие размерности входного [блоба](DnnBlob.md) будут считаться размером векторов в наборе.

- `NA_ObjectSize` - *[По умолчанию]* входной блоб интерпретируется как `BatchLength * BatchWidth * ListSize` векторов длины `Height * Width * Depth * Channels`
- `NA_BatchLength` - входной блоб интерпретируется как `BatchWidth * ListSize * Height * Width * Depth * Channels` векторов длины `BatchLength`
- `NA_ListSize` - входной блоб интерпретируется как `BatchLength * BatchWidth * Height * Width * Depth * Channels` векторов длины `ListSize`
- `NA_Channel` - входной блоб интерпретируется как `BatchLength * BatchWidth * ListSize * Height * Width * Depth` векторов длины `Channels`

## Обучаемые параметры

Слой не имеет обучаемых параметров.

## Входы

На единственный вход подается блоб произвольного размера. Какие размерности будут считаться за размеры векторов, а какие - за размеры набора, зависит от [`GetNormalizationArea()`](область-применения).

## Выходы

Единственный выход содержит блоб того же размера, что и блоб входа, с результатами применения `softmax` к каждому вектору из набора.
