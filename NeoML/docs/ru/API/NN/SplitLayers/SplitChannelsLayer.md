# Класс CSplitChannelsLayer

<!-- TOC -->

- [Класс CSplitChannelsLayer](#класс-csplitchannelslayer)
    - [Настройки](#настройки)
        - [Размеры выходов](#размеры-выходов)
    - [Обучаемые параметры](#обучаемые-параметры)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, разбивающий единственный входной блоб на несколько меньших блобов по размерности `Channels`.

## Настройки

### Размеры выходов

```c++
void SetOutputCounts(const CArray<int>& outputCounts);
```

Установка размерности `Channels` у выходов. Подробнее о количестве и размерах выходов см. [ниже](#выходы).

```c++
void SetOutputCounts2(int count0);
void SetOutputCounts3(int count0, int count1);
void SetOutputCounts4(int count0, int count1, int count2);
```

Установка размерности `Channels` у выходов, без необходимости конструировать массив. Аналогично вызовам `SetOutputCounts(const CArray<int>&)` с массивом из `1`, `2` и `3` элементов соответственно. Подробнее о количестве и размерах выходов см. [ниже](#выходы).

## Обучаемые параметры

Слой не имеет обучаемых параметров.

## Входы

На единственный вход подается блоб размера:

- `BatchLength`, `BatchWidth`, `ListSize`, `Height`, `Width`, `Depth` могут быть произвольными;
- `Channels` не меньше суммы элементов `GetOutputCounts()`.

## Выходы

Слой имеет как минимум `GetOutputCounts().Size()` выходов, каждый из которых содержит блоб размера:

- `BatchLength`, `BatchWidth`, `ListSize`, `Height`, `Width`, `Depth` равны соответствующим размерам входа;
- `Channels` равен соответствующему элементу `GetOutputCount()`, например, у первого выхода `Channels` равен `GetOutputCount()[0]` и т.д.

Однако, если `Channels` входа **больше** суммы элементов `GetOutputCounts()`, то у слоя будет еще один выход размера:

- `BatchLength`, `BatchWidth`, `ListSize`, `Height`, `Width`, `Depth` равны соответствующим размерам входа;
- `Channels` равен разности `Channels` входа и суммы элементов `GetOutputCount()`.
