# Класс CCumSumLayer

<!-- TOC -->

- [Класс CCumSumLayer](#класс-ccumsumlayer)
    - [Настройки](#настройки)
        - [Используемая размерность](#используемая-размерность)
        - [Подсчет в обратную сторону](#подсчет-в-обратную-сторону)
    - [Обучаемые параметры](#обучаемые-параметры)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, который считает кумулятивную сумму вдоль одной из размерностей блоба.

## Настройки

### Используемая размерность

```c++
void SetDimension(TBlobDim d);
```

Установка размерности блоба, вдоль которой будет посчитана сумма.

### Подсчет в обратную сторону

```c++
void SetReverse(bool newReverse);
```

Подсчет кумулятивных сум в обратном порядке (вдоль той же размерности).

## Обучаемые параметры

Слой не имеет обучаемых параметров.

## Входы

На единственный вход подается блоб произвольного размера и типа данных.

## Выходы

Единственный выход содержит блоб того же размера и типа данных, что и на входе.