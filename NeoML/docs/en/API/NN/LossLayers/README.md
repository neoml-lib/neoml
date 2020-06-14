# Loss Functions

- For binary classification:
  - [CBinaryCrossEntropyLossLayer](BinaryCrossEntropyLossLayer.md) - cross-entropy
  - [CHingeLossLayer](HingeLossLayer.md) - hinge loss
  - [CSquaredHingeLossLayer](SquaredHingeLossLayer.md) - modified squared hinge
  - [CBinaryFocalLossLayer](BinaryFocalLossLayer.md) - focal loss (modified cross-entropy)
- For multi-class classification:
  - [CCrossEntropyLossLayer](CrossEntropyLossLayer.md) - cross-entropy
  - [CMultiHingeLossLayer](MultiHingeLossLayer.md) - hinge loss
  - [CMultiSquaredHingeLossLayer](MultiSquaredHingeLossLayer.md) - modified squared hinge
  - [CFocalLossLayer](FocalLossLayer.md) - focal loss (modified cross-entropy)
- For regression:
  - [CEuclideanLossLayer](EuclideanLossLayer.md) - Euclidean distance
- Additionally:
  - [CCenterLossLayer](CenterLossLayer.md) - the auxiliary *center loss* function that penalizes large variance inside a class