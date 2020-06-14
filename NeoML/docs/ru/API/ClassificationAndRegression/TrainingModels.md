# Интерфейсы для обучения

<!-- TOC -->

- [Интерфейсы для обучения](#интерфейсы-для-обучения)
	- [Для классификации](#для-классификации)
	- [Для регрессии](#для-регрессии)

<!-- /TOC -->

Все алгоритмы классификации реализуют интерфейс `ITrainingModel`; все алгоритмы регрессии — интерфейс `IRegressionTrainingModel`.

## Для классификации

Интерфейс `ITrainingModel` предоставляет метод `Train`, который принимает на вход набор данных в виде объекта, реализующего интерфейс `IProblem`, и возвращает модель, реализующую интерфейс `IModel`.

```c++
class NEOML_API ITrainingModel {
public:
	virtual ~ITrainingModel() = 0;

	// Построить классификатор, обученный на заданных данных
	virtual CPtr<IModel> Train( const IProblem& trainingClassificationData ) = 0;
};
```

## Для регрессии

Интерфейс `IRegressionTrainingModel` предоставляет метод `TrainRegression`, который принимает на вход набор данных в виде объекта, реализующего интерфейс `IRegressionProblem`, и возвращает модель, реализующую интерфейс `IRegressionModel`.

```c++
class NEOML_API IRegressionTrainingModel {
public:
	virtual ~RegressionITrainingModel() = 0;

	// Построить модель регрессии, используя заданные значения
	virtual CPtr<IRegressionModel> TrainRegression( const IRegressionProblem& problem ) = 0;
};
```
