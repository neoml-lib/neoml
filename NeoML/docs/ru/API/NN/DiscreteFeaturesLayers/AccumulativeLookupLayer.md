# Класс CAccumulativeLookupLayer

<!-- TOC -->

- [Класс CAccumulativeLookupLayer](#класс-caccumulativelookuplayer)
    - [Настройки](#настройки)
    - [Обучаемые параметры](#обучаемые-параметры)
        - [Таблица векторов](#таблица-векторов)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, сопоставляющий значениям дискретного признака векторы фиксированной длины и обучающий эти вектора.

Например, векторные представления слов ([Word2Vec](https://en.wikipedia.org/wiki/Word2vec), [GloVe](https://en.wikipedia.org/wiki/GloVe_(machine_learning)) и т.д.).

Данный слой поддерживает работу с одним признаком, и при получении нескольких значений признака возвращает сумму векторов, соответствующих этим значениям.

## Настройки

```c++
// Размер одной таблицы.
struct CLookupDimension {
    int VectorCount; // Количество векторов.
    int VectorSize; // Размер векторов.
};

void SetDimension( const CLookupDimension& newDimension );
```

Установка размера таблицы векторов.

## Обучаемые параметры

### Таблица векторов

```c++
CPtr<CDnnBlob> GetEmbeddings() const;
```

Блоб с обученными векторами, размера:

- `BatchLength` равен `GetDimension().VectorCount`;
- `BatchWidth` равен `GetDimension().VectorSize`.

## Входы

На единственный вход подаётся блоб с данными типа `int`, содержащий наборы значений, размера:

- `BatchLength * BatchWidth * ListSize` - количество наборов значений признака;
- `Height * Width * Depth * Channels` - количество значений признака в наборе.

## Выходы

Единственный выход содержит блоб с суммами векторов из таблицы, указанных в наборах. Блоб имеет размеры:

- `BatchLength` равен `BatchLength` входа;
- `BatchWidth` равен `BatchWidth` входа;
- `ListSize` равен `ListSize` входа;
- `Height` равен `1`;
- `Width` равен `1`;
- `Depth` равен `1`;
- `Channels` равен `GetDimension().VectorSize`.
