# Функции потерь

- Бинарная классификация:
  - [CBinaryCrossEntropyLossLayer](BinaryCrossEntropyLossLayer.md) - перекрёстная энтропия;
  - [CHingeLossLayer](HingeLossLayer.md) - функция `Hinge`;
  - [CSquaredHingeLossLayer](SquaredHingeLossLayer.md) - модифицированная функция `SquaredHinge`;
  - [CBinaryFocalLossLayer](BinaryFocalLossLayer.md) - функция `Focal` (модифицированная кросс-энтропия);
- Многоклассовая классификация:
  - [CCrossEntropyLossLayer](CrossEntropyLossLayer.md) - перекрёстная энтропия;
  - [CMultiHingeLossLayer](MultiHingeLossLayer.md) - функция `Hinge`;
  - [CMultiSquaredHingeLossLayer](MultiSquaredHingeLossLayer.md) - модифицированная функция `SquaredHinge`;
  - [CFocalLossLayer](FocalLossLayer.md) - функция `Focal` (модифицированная кросс-энтропия);
- Регрессия:
  - [CEuclideanLossLayer](EuclideanLossLayer.md) - евклидово расстояние;
- Дополнительно:
  - [CCenterLossLayer](CenterLossLayer.md) - вспомогательная функция `Center`, штрафующая дисперсию внутри классов.