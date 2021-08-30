# Класс CSplitBatchLengthLayer

<!-- TOC -->

- [Класс CSplitBatchLengthLayer](#класс-csplitbatchlengthlayer)
    - [Настройки](#настройки)
        - [Размеры выходов](#размеры-выходов)
    - [Обучаемые параметры](#обучаемые-параметры)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, разбивающий единственный входной блоб на несколько меньших блобов по размерности `BatchLength`.

## Настройки

### Размеры выходов

```c++
void SetOutputCounts(const CArray<int>& outputCounts);
```

Установка размерности `BatchLength` у выходов. Подробнее о количестве и размерах выходов см. [ниже](#выходы).

```c++
void SetOutputCounts2(int count0);
void SetOutputCounts3(int count0, int count1);
void SetOutputCounts4(int count0, int count1, int count2);
```

Установка размерности `BatchLength` у выходов, без необходимости конструировать массив. Аналогично вызовам `SetOutputCounts(const CArray<int>&)` с массивом из `1`, `2` и `3` элементов соответственно. Подробнее о количестве и размерах выходов см. [ниже](#выходы).

## Обучаемые параметры

Слой не имеет обучаемых параметров.

## Входы

На единственный вход подается блоб размера:

- `BatchLength`, `Height`, `ListSize`, `Width`, `Depth`, `Channels` могут быть произвольными;
- `BatchLength` не меньше суммы элементов `GetOutputCounts()`.

## Выходы

Слой имеет как минимум `GetOutputCounts().Size()` выходов, каждый из которых содержит блоб размера:

- `BatchLength` равен соответствующему элементу `GetOutputCount()`, например, у первого выхода `BatchLength` равен `GetOutputCount()[0]` и т.д;
- `BatchWidth`, `ListSize`, `Height`, `Width`, `Depth`, `Channels` равные соответствующим размерам входа.

Однако, если `BatchLength` входа **больше** суммы элементов `GetOutputCounts()`, то у слоя будет еще один выход размера:

- `BatchLength` равен разности `BatchLength` входа и суммы элементов `GetOutputCount()`;
- `BatchWidth`, `ListSize`, `Height`, `Width`, `Depth`, `Channels` равные соответствующим размерам входа.
