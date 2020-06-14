# Класс CBitSetVectorizationLayer

<!-- TOC -->

- [Класс CBitSetVectorizationLayer](#класс-cbitsetvectorizationlayer)
    - [Настройки](#настройки)
        - [Размер bitset](#размер-bitset)
    - [Обучаемые параметры](#обучаемые-параметры)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, конвертирующий набор bitset в вектора из нулей и единиц.

## Настройки

### Размер bitset

```c++
void SetBitSetSize( int bitSetSize );
```

## Обучаемые параметры

Слой не имеет обучаемых параметров.

## Входы

На единственный вход подается блоб c данными типа `int` следующего размера:

- `BatchLength * BatchWidth * ListSize * Height * Width * Depth` - количество `bitset` в наборе
- `Channels` - бинарное представление `bitset`

## Выходы

Единственный выход содержит блоб размера:

- `Channels` равен `GetBitSetSize()`
- остальные размерности равны аналогичным у входа
