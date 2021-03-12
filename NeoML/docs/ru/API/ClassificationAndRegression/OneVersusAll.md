# Многоклассовая классификация методом один против всех COneVersusAll

<!-- TOC -->

- [Многоклассовая классификация методом один против всех COneVersusAll](#многоклассовая-классификация-методом-один-против-всех-coneversusall)
	- [Параметры построения модели](#параметры-построения-модели)
	- [Модель](#модель)
	- [Результат классификации](#результат-классификации)
	- [Пример](#пример)

<!-- /TOC -->

Данный метод позволяет выполнить многоклассовую классификацию, имея только бинарный классификатор.

Метод заключается в преобразовании исходной многоклассовой задачи в набор бинарных задач путем противопоставления каждого из классов данных всем остальным. На этом наборе обучается соответствующее количество бинарных классификаторов. Классификация же выполняется путем отнесения объекта к классу, бинарный классификатор для которого показал максимальную уверенность.

В **NeoML** алгоритм реализован классом `COneVersusAll`. Он предоставляет метод `Train` для обучения модели классификации.

## Параметры построения модели

Алгоритм имеет только один параметр — указатель на базовый метод бинарной классификации, который должен быть представлен объектом, реализующим [ITrainingModel](TrainingModels.md).

## Модель

Модель, обученная данным методом, представляет собой ансамбль бинарных моделей. Построенная модель описывается интерфейсом `IOneVersusAllModel`:

```c++
class NEOML_API IOneVersusAllModel : public IModel {
public:
	virtual ~IOneVersusAllModel() = 0;

	// получить базовые IModel классификаторов
	virtual const CObjectArray<IModel>& GetModels() const = 0;

	// получить результат классификации с информацией о нормализации вероятностей
	virtual bool ClassifyEx( const CSparseFloatVector& data,
		COneVersusAllClassificationResult& result ) const = 0;
	virtual bool ClassifyEx( const CSparseFloatVectorDesc& data,
		COneVersusAllClassificationResult& result ) const = 0;

	// Сериализация.
	virtual void Serialize( CArchive& ) = 0;
};
```

## Результат классификации

Помимо стандартного метода `Classify` модель классификации "один против всех" предоставляет метод `ClassifyEx`, который возвращает результат типа `COneVersusAllClassificationResult`.

```c++
struct NEOML_API COneVersusAllClassificationResult : public CClassificationResult {
public:
	double SigmoidSum;
};
```
- *SigmoidSum* — сумма сигмоид, по которой можно восстановить ненормированные вероятности, возвращаемые бинарными классификаторами.

## Пример

Ниже представлен простой пример обучения модели на базе линейного бинарного классификатора.

```c++
CLinear linear( EF_LogReg );
	
COneVersusAll oneVersusAll( linear );
CPtr<IModel> model = oneVersusAll.Train( *trainData );
```